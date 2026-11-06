"""本地 Web 服务：加载 models/ 下已下载的模型，对同一段音频做多模型对比。

用法:
    python server.py                 # 加载所有已下载模型，打开 http://127.0.0.1:8000
    python server.py --models sensevoice,paraformer-zh-small --threads 4
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

import numpy as np
import uvicorn
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

import asr_engine
import registry
import live_engine
import speaker_owner
import semantic_seg

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"

app = FastAPI(title="端侧 ASR 对比 Demo")
STATE: dict = {"models": {}, "load_errors": {}, "diarizer": None, "diar_error": None, "threads": 2,
               "save_audio": True, "owner_profile": None, "owner_model": None}
LOCK = threading.Lock()  # 串行推理，保证各模型耗时互不干扰


def load_models(selected: list[str] | None, threads: int):
    STATE["threads"] = threads
    downloaded = asr_engine.find_downloaded()
    for spec in registry.ASR_MODELS:
        key = spec["key"]
        if selected and key not in selected:
            continue
        if key not in downloaded:
            continue
        m = asr_engine.AsrModel(spec, downloaded[key], num_threads=threads)
        try:
            print(f"[加载] {spec['name']} ...", flush=True)
            m.load()
            STATE["models"][key] = m
            print(f"       完成，用时 {m.load_seconds:.1f}s", flush=True)
        except Exception as e:  # noqa: BLE001 —— 单个模型失败不影响其他
            STATE["load_errors"][key] = f"{type(e).__name__}: {e}"
            print(f"       失败: {e}", flush=True)

    seg = registry.resolve_files(registry.get_spec("pyannote-seg"))
    emb = registry.resolve_files(registry.get_spec("speaker-embedding"))
    if seg and emb:
        try:
            STATE["diarizer"] = asr_engine.Diarizer(seg["model"], emb["model"], num_threads=threads)
            print("[加载] 说话人分离模型就绪", flush=True)
        except Exception as e:  # noqa: BLE001
            STATE["diar_error"] = str(e)
    else:
        STATE["diar_error"] = "未下载说话人分离模型（python download_models.py 默认会下载）"

    # 声纹主人识别：注册文件存在（data/owner_profile.json，不提交）才启用；没有就是普通模式，无 owner 字段
    profile_path = speaker_owner.DEFAULT_PROFILE
    if profile_path.exists() and emb:
        try:
            STATE["owner_profile"] = speaker_owner.OwnerProfile.load(profile_path)
            STATE["owner_model"] = speaker_owner.load_model(emb["model"])
            print(f"[加载] 主人声纹注册就绪（{len(STATE['owner_profile'])} 条向量）", flush=True)
        except Exception as e:  # noqa: BLE001
            STATE["owner_profile"] = None
            print(f"[加载] 主人声纹注册失败: {e}", flush=True)

    if not STATE["models"]:
        print("\n!!! 没有可用模型。请先运行: python download_models.py\n", flush=True)


def model_info(spec: dict) -> dict:
    files = registry.resolve_files(spec)
    key = spec["key"]
    if key in STATE["models"]:
        status = "loaded"
    elif key in STATE["load_errors"]:
        status = "error"
    elif files:
        status = "downloaded"  # 已下载但本次未选择加载
    else:
        status = "missing"
    info = {k: v for k, v in spec.items() if k not in ("files", "hf_allow")}
    info.update({
        "status": status,
        "error": STATE["load_errors"].get(key),
        "disk_size_mb": registry.disk_size_mb(files),
        "files": {k: v.name for k, v in files.items()} if files else None,
        "load_seconds": round(STATE["models"][key].load_seconds, 2) if key in STATE["models"] else None,
    })
    return info


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/live")
def live_page():
    return FileResponse(WEB_DIR / "live.html")


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    """持续监听：首条消息为 JSON 配置，之后是 16kHz 单声道 Int16 小端二进制音频帧。

    配置：{"models": ["sensevoice", "paraformer-zh-small"], "min_silence": 0.5, "max_chunk": 15,
          "partial_interval": 0.4, "piece_gap": 0.2, "piece_max": 8, "save_audio": true}
    partial_interval：说话中每隔几秒把当前这一段重新识别一次（实时出字），0 或不给 = 不出；
    piece_gap / piece_max：句内停顿多久切段 / 一段最长几秒（实时出字按段识别，不影响正式结果）；
    min_speech / neg_threshold：VAD 开口要多久 / 开口后概率低于多少计静音（可选，默认见 LiveSession，页面不暴露）；
    save_audio：false = 这次不存录音（回放已有文件时用；服务端开了 --no-save-audio 时一律不存）。
    推送：live_engine.LiveSession 的事件（JSON）。
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()
    sess = None
    try:
        cfg = json.loads(await ws.receive_text())
        keys = [k for k in cfg.get("models") or list(STATE["models"]) if k in STATE["models"]]
        if not keys:
            await ws.send_json({"type": "error", "message": "没有可用模型，请先 python download_models.py"})
            return
        sess = live_engine.LiveSession(
            {k: STATE["models"][k] for k in keys},
            on_event=lambda ev: loop.call_soon_threadsafe(out.put_nowait, ev),
            min_silence=float(cfg.get("min_silence", 0.5)), max_chunk=float(cfg.get("max_chunk", 15)),
            partial_interval=min(max(float(cfg.get("partial_interval") or 0), 0.0), 10.0),
            piece_gap=min(max(float(cfg.get("piece_gap", 0.2)), 0.0), 2.0),
            piece_max=min(max(float(cfg.get("piece_max", 8.0)), 1.0), 30.0),
            min_speech=min(max(float(cfg.get("min_speech", 0.15)), 0.05), 1.0),
            neg_threshold=min(max(float(cfg.get("neg_threshold", 0.5)), 0.1), 0.5),
            lock=LOCK, log_dir=ROOT_DIR / "logs" / "live",
            owner_profile=STATE["owner_profile"], owner_model=STATE["owner_model"],
            save_audio=STATE["save_audio"] and cfg.get("save_audio", True) is not False)
        await ws.send_json({"type": "ready", "models": [{"key": k, "name": STATE["models"][k].spec["name"]}
                                                        for k in keys],
                            "min_silence": sess.min_silence, "max_chunk": sess.max_chunk,
                            "partial_interval": sess.partial_interval, "piece_gap": sess.piece_gap,
                            "piece_max": sess.piece_max, "min_speech": sess.min_speech,
                            "neg_threshold": sess.neg_threshold,
                            "audio_file": sess.audio_path.relative_to(ROOT_DIR).as_posix() if sess.audio_path else None})

        done = asyncio.Event()

        async def pump():  # 唯一的发送方：避免多个协程同时往同一个 WebSocket 写
            while True:
                ev = await out.get()
                await ws.send_json(ev)
                if ev.get("type") == "stopped":
                    done.set()
                    return

        sender = asyncio.create_task(pump())
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    pcm = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768.0
                    await run_in_threadpool(sess.accept, pcm)
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                    s, sess = sess, None
                    await run_in_threadpool(s.close)  # 冲出最后一句并等转写完成
                    loop.call_soon_threadsafe(out.put_nowait, {"type": "stopped"})
                    await asyncio.wait_for(done.wait(), 10)
                    break
        finally:
            sender.cancel()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        except Exception:  # noqa: BLE001
            pass
    finally:
        if sess is not None:
            await run_in_threadpool(sess.close, 5.0)


@app.get("/api/models")
def api_models():
    return {
        "models": [model_info(s) for s in registry.ASR_MODELS],
        "diarization": {"ready": STATE["diarizer"] is not None, "error": STATE["diar_error"]},
        "threads": STATE["threads"],
    }


def _diar_asr_model():
    for key in ("sensevoice", "paraformer-zh-small"):
        if key in STATE["models"]:
            return STATE["models"][key]
    return next(iter(STATE["models"].values()), None)


@app.post("/api/transcribe")
async def api_transcribe(request: Request, ref: str = "", diarize: int = 0, num_speakers: int = -1,
                         threshold: float = 0.5, norm_num: int = 1, vad: int = 1,
                         seg: str = "", decider: str = ""):
    data = await request.body()
    if not data:
        raise HTTPException(400, "空音频")
    try:
        samples, sr = asr_engine.read_wav(data)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"无法解析 WAV（仅支持 PCM WAV）: {e}") from e
    samples = asr_engine.resample(samples, sr, asr_engine.SAMPLE_RATE)
    sr = asr_engine.SAMPLE_RATE
    duration = len(samples) / sr
    if duration < 0.2:
        raise HTTPException(400, "录音太短（<0.2s）")
    # seg: semantic（默认）| vad | window；旧参数 vad=0 等价于 window
    mode = seg if seg in ("semantic", "vad", "window") else ("semantic" if vad else "window")
    results, diar, vad_info = await run_in_threadpool(_run_all, samples, sr, ref, diarize, num_speakers,
                                                     threshold, bool(norm_num), mode, decider)
    return JSONResponse({"duration": round(duration, 3), "results": results, "diarization": diar,
                         "ref": ref, "vad": mode != "window", "seg_mode": mode, "vad_info": vad_info,
                         "time": time.strftime("%H:%M:%S")})


def _semantic_decider(prefer: str = ""):
    """语义切段的决策模型。

    prefer 指定时用它（在网页上切换决策模型，好对比两种模型的切段差异）；
    留空则优先 SenseVoice（自带标点，切得最准），否则 Paraformer（只靠规则）。
    """
    order = (prefer,) if prefer in STATE["models"] else ("sensevoice", "paraformer-zh-small")
    for key in order:
        if key in STATE["models"]:
            m = STATE["models"][key]
            return m, (lambda x, m=m: m._offline_stream(m.recognizer, asr_engine.SAMPLE_RATE, x)["text"])
    return None, None


def _run_all(samples: np.ndarray, sr: int, ref: str, diarize: int, num_speakers: int, threshold: float,
             norm_num: bool, mode: str = "semantic", decider: str = ""):
    """注意：切段只算一次，供所有模型共用；各模型耗时里不含 VAD 和语义决策。"""
    results = []
    vad_info = {"enabled": mode != "window", "mode": mode, "ms": 0.0, "n_segments": 0}
    with LOCK:
        segs: list = []
        if mode == "vad":
            segs, vad_ms = asr_engine.compute_vad(samples, sr)
            vad_info.update({"ms": vad_ms, "n_segments": len(segs)})
        elif mode == "semantic":
            frags, vad_ms = asr_engine.compute_vad(samples, sr, min_silence=0.1)
            dm, fn = _semantic_decider(decider)
            vad_info.update({"ms": vad_ms, "n_frags": len(frags)})
            if frags and fn:
                utts, st = semantic_seg.segment(samples, frags, fn, sr)
                segs = semantic_seg.spans(utts)
                vad_info.update({"n_segments": len(segs), "decide_ms": st["total_ms"], "decider": dm.spec["name"],
                                 "utterances": [{"start": round(u.start, 2), "end": round(u.end, 2),
                                                 "text": "".join(u.frag_texts), "n_frags": len(u.frag_ids),
                                                 "reason": u.reason, "wait": round(u.wait, 2)} for u in utts]})
        if mode != "window" and not segs:
            # 没有 VAD 模型或没检到人声：保持原行为（长音频退回固定窗口）
            vad_info["note"] = "未检到人声段，退回固定窗口切分"
        for key, m in STATE["models"].items():
            try:
                r = m.transcribe(samples, sr, segs=segs)
                if ref.strip():
                    c = asr_engine.cer(ref, r["text"], norm_num)
                    r["cer"] = None if c is None else round(c, 4)
            except Exception as e:  # noqa: BLE001
                r = {"key": key, "error": f"{type(e).__name__}: {e}"}
            results.append(r)

        # VAD / 固定窗口这两个模式没有“文字判断”这一步，但也要把切点列出来，
        # 才能和语义模式对比。文字直接取自模型自己刚算好的逐段结果（不多跑一遍识别）。
        # 用哪个模型的文字：跟「决策」选择一致（VAD/窗口下它就是“逐段文字取自谁”）；
        # 不再默认取第一个模型，否则选了 Paraformer 却看到 SenseVoice 的标点。
        if mode != "semantic":
            want = decider if decider in STATE["models"] else ("sensevoice" if "sensevoice" in STATE["models"] else "")
            pick = next((r for r in results if r.get("key") == want and r.get("cuts")), None)
            if pick is None:
                pick = next((r for r in results if r.get("cuts")), None)
            cuts = (pick or {}).get("cuts") or []
            why = "gap" if mode == "vad" else "window"
            vad_info["utterances"] = [
                {"start": c["start"], "end": c["end"], "text": c.get("text", ""),
                 "n_frags": 1, "reason": why, "wait": None}
                for c in cuts]
            vad_info["n_segments"] = len(cuts)
            vad_info["text_from"] = (pick or {}).get("key")

        diar = None
        if diarize:
            if STATE["diarizer"] is None:
                diar = {"error": STATE["diar_error"]}
            else:
                try:
                    segs, cost = STATE["diarizer"].run(samples, num_speakers=num_speakers, threshold=threshold)
                    asr = _diar_asr_model()
                    for s in segs:
                        a, b = int(s["start"] * sr), int(s["end"] * sr)
                        s["text"] = asr.transcribe(samples[a:b], sr)["text"] if asr and b - a > sr * 0.2 else ""
                    diar = {"segments": segs, "elapsed_ms": round(cost, 1),
                            "asr_model": asr.spec["name"] if asr else None,
                            "num_speakers": len({s["speaker"] for s in segs})}
                except Exception as e:  # noqa: BLE001
                    diar = {"error": f"{type(e).__name__}: {e}"}
    return results, diar, vad_info


def main():
    ap = argparse.ArgumentParser(description="端侧 ASR 多模型对比 Demo")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--threads", type=int, default=2, help="每个模型的推理线程数（端侧建议 1~4）")
    ap.add_argument("--models", default="", help="逗号分隔的模型 key，默认加载全部已下载模型")
    ap.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    ap.add_argument("--no-save-audio", action="store_true",
                    help="持续监听时不把录音存到 logs/live/（默认存一份，只在本机，不提交）")
    args = ap.parse_args()
    STATE["save_audio"] = not args.no_save_audio

    selected = [s.strip() for s in args.models.split(",") if s.strip()] or None
    load_models(selected, args.threads)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\n打开浏览器访问: {url}   （Ctrl+C 退出）\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    if sys.platform == "win32":
        try:  # 防止旧版控制台打印中文报错
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    main()
