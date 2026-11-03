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
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

import asr_engine
import registry
import semantic_seg

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="端侧 ASR 对比 Demo")
STATE: dict = {"models": {}, "load_errors": {}, "diarizer": None, "diar_error": None, "threads": 2}
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
                         seg: str = ""):
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
                                                     threshold, bool(norm_num), mode)
    return JSONResponse({"duration": round(duration, 3), "results": results, "diarization": diar,
                         "ref": ref, "vad": mode != "window", "seg_mode": mode, "vad_info": vad_info,
                         "time": time.strftime("%H:%M:%S")})


def _semantic_decider():
    """语义切段的决策模型：优先 SenseVoice（自带标点，切得最准），否则 Paraformer（只靠规则）。"""
    for key in ("sensevoice", "paraformer-zh-small"):
        if key in STATE["models"]:
            m = STATE["models"][key]
            return m, (lambda x, m=m: m._offline_stream(m.recognizer, asr_engine.SAMPLE_RATE, x)["text"])
    return None, None


def _run_all(samples: np.ndarray, sr: int, ref: str, diarize: int, num_speakers: int, threshold: float,
             norm_num: bool, mode: str = "semantic"):
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
            dm, fn = _semantic_decider()
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
    args = ap.parse_args()

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
