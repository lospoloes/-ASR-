"""按真实速度把一段 WAV 推给正在运行的 server.py /ws/live（和网页推流一样），记下每个事件到达的时刻，
对比不同的实时出字间隔：屏幕上的字落后多少、最长多久没有新字、正式结果慢了多少、多花了多少算力。

用法:
    python server.py --port 8001 --no-browser          # 先起服务（另开一个终端）
    python tools/replay_live.py test_audio/ramc/<名字>.wav --url ws://127.0.0.1:8001/ws/live
        --partial 0 0.5 1   依次跑：不出临时结果 / 每 0.5 秒（页面默认）/ 每 1 秒（默认）
        --piece-gap 0.2     句内停顿多久切段（默认 0.2 秒）；--piece-max 8：一段最长几秒
        --speed 2           两倍速推流（只用来调试，结论以 1 倍速为准）
    python tools/replay_live.py --load logs/replay/*.jsonl     # 不重跑，重新统计之前存下的事件（可以混着旧版的）
WAV 旁边有同名 .tsv（起、止、说话人、文本；tools/fetch_ramc.py 生成）时，按人工标注的说话时段统计“屏幕落后”，并算 CER。
输出 benchmark/results/live_realtime.md；每次跑的事件存到 logs/replay/（不提交）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import asr_engine  # noqa: E402
from live_sim import machine  # noqa: E402

SR = 16000
FRAME = 1600  # 0.1s 一块，和网页推流粒度一致
NAMES = {"sensevoice": "SenseVoice", "paraformer-zh-small": "Paraformer"}


# ------------------------------------------------------------ 推流
async def stream(url: str, pcm: np.ndarray, cfg: dict, speed: float):
    import websockets

    events = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps(cfg))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise RuntimeError(f"服务端没有就绪：{ready}")
        t0 = time.perf_counter()

        async def reader():
            async for msg in ws:
                ev = json.loads(msg)
                ev["_t"] = round((time.perf_counter() - t0) * speed, 3)  # 收到的时刻（换算成音频时间，秒）
                events.append(ev)
                if ev["type"] == "chunk":
                    print(f"  {ev['_t']:6.1f}s  chunk #{ev['id']} {ev['start']:.1f}-{ev['end']:.1f}s", flush=True)
                if ev["type"] == "stopped":
                    return

        rd = asyncio.create_task(reader())
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME].tobytes())
            delay = (i + FRAME) / SR / speed - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(rd, 120)
    return ready, events


# ------------------------------------------------------------ 统计
def prefix_edits(a: str, b: str) -> int:
    """把 a（旧的临时结果）改成 b（后来的结果）的某个开头，最少要改几个字。"""
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return min(prev)


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def analyze(run: dict, spans, ref: str | None) -> dict:
    events, models = run["events"], [m["key"] for m in run["ready"]["models"]]
    chunks = {e["id"]: e for e in events if e["type"] == "chunk"}
    if not spans:  # 没有人工标注：用这次 VAD 切出来的段当“有人在说话”
        spans = [(c["start"], c["end"]) for c in chunks.values()]
    norm = lambda s: asr_engine.normalize_text(s, True)  # noqa: E731
    res = {}
    for k in models:
        texts = [e for e in events if e["type"] == "text" and e["model"] == k]
        parts = [e for e in events if e["type"] == "partial" and e["model"] == k]
        # 每次临时识别送进去多少秒音频（旧版事件没有 audio 字段：那时每次都从这句开头 − 0.2 秒识别到 end）
        audio = [e.get("audio", e["end"] - e["start"] + 0.2) for e in parts]
        ups = sorted((e["_t"], e["end"]) for e in texts + parts)  # (出字时刻, 这次出字识别到了音频的哪儿)
        ts = np.array([u[0] for u in ups])
        cov = np.maximum.accumulate(np.array([u[1] for u in ups]))
        # 说话时屏幕落后：每 0.1s 看一眼，屏幕上的字识别到了多久之前
        lag = []
        for s, e in spans:
            for t in np.arange(s, e, 0.1):
                i = int(np.searchsorted(ts, t, side="right")) - 1
                lag.append(max(0.0, t - (cov[i] if i >= 0 else 0.0)))
        gaps = np.diff(ts) if len(ts) > 1 else np.zeros(1)
        g = int(np.argmax(gaps))
        # 开口后多久见到这句的第一个字（临时或正式）
        first = {}
        for e in texts + parts:
            first[e["chunk_id"]] = min(first.get(e["chunk_id"], 1e9), e["_t"])
        see = [first[c] - chunks[c]["start"] for c in chunks if c in first]
        # 临时结果被改掉的字：相邻两次之间（含最后一次临时 → 正式），旧的灰字有多少后来被改了
        edits = shown = 0
        final_of = {e["chunk_id"]: e["text"] for e in texts}
        for c in chunks:
            seq = [norm(p["text"]) for p in parts if p["chunk_id"] == c] + [norm(final_of.get(c, ""))]
            for a, b in zip(seq, seq[1:]):
                edits += prefix_edits(a, b)
                shown += len(a)
        hyp = "".join(e["text"] for e in sorted(texts, key=lambda e: e["chunk_id"]))
        dec_f, dec_p = sum(e["decode_ms"] for e in texts), sum(e["decode_ms"] for e in parts)
        shown_ms = [e["shown_ms"] for e in texts]
        res[k] = {
            "see": (float(np.mean(see)), float(np.max(see))),
            "lag": (float(np.mean(lag)), pct(lag, 90), float(np.max(lag))),
            "gap": (float(gaps[g]), float(ts[g]) if len(ts) > 1 else 0.0),
            "shown": (float(np.mean(shown_ms)), pct(shown_ms, 90), float(np.max(shown_ms))),
            "n_partial": len(parts), "n_lock": sum(e.get("kind") == "lock" for e in parts), "dec_f": dec_f, "dec_p": dec_p,
            "audio": (float(np.mean(audio)), float(np.max(audio))) if audio else None,
            "duty": (dec_f + dec_p) / 1000 / run["dur"], "rewrite": edits / shown if shown else None,
            "cer": asr_engine.edit_distance(norm(ref), norm(hyp)) / len(norm(ref)) if ref else None,
            "finals": [(e["chunk_id"], e["text"]) for e in sorted(texts, key=lambda e: e["chunk_id"])],
        }
    res["_chunks"] = len(chunks)
    return res


def report(runs: list[dict], stats: list[dict], wav: Path, n_spans: int, threads) -> str:
    def base_of(r):   # 和同一版实现（同一个 VAD、同样的切句）下不开实时出字的那一次比正式结果
        new = "piece_gap" in r["cfg"]
        return next((s for rr, s in zip(runs, stats)
                     if not rr["cfg"]["partial_interval"] and ("piece_gap" in rr["cfg"]) == new), None)

    def label(r):
        c = r["cfg"]
        if not c["partial_interval"]:
            return "不开" if "piece_gap" in c else "不开（旧版）"
        if "piece_gap" not in c:
            return f"整句重识别，每 {c['partial_interval']:g} 秒（旧版）"
        return f"按段，每 {c['partial_interval']:g} 秒"
    r0 = next((r for r in runs if "piece_gap" in r["cfg"]), runs[0])
    lines = [
        "# 实时出字：真人双人对话按真实速度回放", "",
        f"- 数据：`{wav.name}`（{r0['dur']:.0f}s，{n_spans} 句人工标注）按真实速度推给 `/ws/live`，和网页推流一样 0.1s 一块；"
        "记下每个事件到达的时刻（`python tools/replay_live.py`）",
        f"- 静音阈值 {r0['cfg']['min_silence']}s，最长 chunk {r0['cfg']['max_chunk']}s"
        + (f"，句内停顿 {r0['cfg']['piece_gap']:g}s 切段、一段最长 {r0['cfg'].get('piece_max', 8):g}s"
           if "piece_gap" in r0["cfg"] else "")
        + (f"，VAD 最短开口 {r0['ready']['min_speech']:g}s、静音阈值 {r0['ready']['neg_threshold']:g}"
           if "min_speech" in r0.get("ready", {}) else "") + "；"
        f"两个模型在同一个转写线程里依次跑（和网页 demo 一样）；机器：{machine()}，每个模型 {threads} 线程",
        "- **开口后见字**：一句话（VAD 切出的 chunk）开始 → 屏幕上出现这句的第一个字（临时或正式）",
        "- **说话时屏幕落后**：有人在说话的每个时刻（按人工标注，每 0.1s 取一点），屏幕上的字只识别到了多少秒之前的话",
        "- **最长没有新字**：这一栏相邻两次更新之间的最长间隔（和 3.1 节的 16.9 秒同一口径）",
        "- **正式结果出字**：VAD 判定说完 → 正式结果到达（含排队；开实时出字时可能要等手上那次临时识别做完）",
        "- **旧版** = 2026-11-05 上午的实现：sherpa VAD，每次把这句从头整句重识别（同一台机器、同一段音频，事件存档重新统计）；"
        "**按段** = 现在的实现：只重识别当前这一段，句内停顿处锁定，说完整句重识别",
        "- **额外解码**：所有临时识别的解码耗时之和，是正式结果解码耗时之和的几倍；**转写占用**：全部解码耗时 / 音频时长",
        "- **灰字被改**：临时结果里的字，后来被下一次临时结果或正式结果改掉的比例（去标点后按字比）",
        "- **单次识别音频**：每次临时识别送进模型的音频长度（平均 / 最长），决定单次临时识别要算多久", "",
        "| 实时出字 | 模型 | 开口后见字（平均 / 最长） | 说话时屏幕落后（平均 / P90 / 最大） | 最长没有新字 | 正式结果出字（平均 / P90 / 最大） |",
        "|---|---|---|---|---|---|",
    ]
    for r, s in zip(runs, stats):
        for k in (m["key"] for m in r["ready"]["models"]):
            v = s[k]
            lines.append(f"| {label(r)} | {NAMES.get(k, k)} | {v['see'][0]:.1f} / {v['see'][1]:.1f} s | "
                         f"{v['lag'][0]:.1f} / {v['lag'][1]:.1f} / {v['lag'][2]:.1f} s | "
                         f"{v['gap'][0]:.1f} s（{v['gap'][1]:.0f}s 起） | "
                         f"{v['shown'][0]:.0f} / {v['shown'][1]:.0f} / {v['shown'][2]:.0f} ms |")
    lines += ["", "| 实时出字 | 模型 | 临时识别次数（其中锁定） | 单次识别音频（平均 / 最长） | 临时识别解码合计 | 额外解码 | 转写占用 | 灰字被改 | 正式结果 CER |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r, s in zip(runs, stats):
        for k in (m["key"] for m in r["ready"]["models"]):
            v = s[k]
            same, base = "", base_of(r)
            if base is not None and s is not base and k in base:
                same = "（与不开时逐字相同）" if v["finals"] == base[k]["finals"] else "（与不开时不同）"
            lines.append(f"| {label(r)} | {NAMES.get(k, k)} | {v['n_partial']}（{v['n_lock']}） | "
                         + (f"{v['audio'][0]:.1f} / {v['audio'][1]:.1f} s" if v["audio"] else "—") + " | "
                         f"{v['dec_p'] / 1000:.1f} s | "
                         + (f"{v['dec_p'] / v['dec_f']:.1f} 倍" if v["n_partial"] else "—") + " | "
                         f"{v['duty'] * 100:.1f}% | " + (f"{v['rewrite'] * 100:.1f}%" if v["rewrite"] is not None else "—")
                         + " | " + (f"{v['cer'] * 100:.2f}%{same}" if v["cer"] is not None else "—") + " |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ 入口
def load_ref(wav: Path):
    tsv = wav.with_suffix(".tsv")
    if not tsv.exists():
        return None, None
    rows = [ln.split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [(float(r[0]), float(r[1])) for r in rows], "".join(r[3] for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", nargs="?", type=Path)
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/live")
    ap.add_argument("--partial", type=float, nargs="*", default=[0, 0.5, 1], help="实时出字间隔（秒），0 = 不出")
    ap.add_argument("--piece-gap", type=float, default=0.2, help="句内停顿多久切段（秒）")
    ap.add_argument("--piece-max", type=float, default=8.0, help="一段最长几秒")
    ap.add_argument("--silence", type=float, default=0.5)
    ap.add_argument("--max-chunk", type=float, default=15.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--load", type=Path, nargs="*", help="不推流，重新统计之前存下的事件文件")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmark" / "results" / "live_realtime.md")
    args = ap.parse_args()

    runs = []
    if args.load:
        for f in args.load:
            lines = f.read_text(encoding="utf-8").splitlines()
            head = json.loads(lines[0])
            runs.append({**head, "events": [json.loads(ln) for ln in lines[1:]]})
        wav = Path(runs[0]["wav"])
    else:
        if args.wav is None:
            ap.error("要给 WAV 文件，或者用 --load")
        wav = args.wav.resolve()
        x, sr = asr_engine.read_wav(wav.read_bytes())
        pcm = np.clip(np.round(asr_engine.resample(x, sr) * 32768.0), -32768, 32767).astype("<i2")
        out_dir = ROOT / "logs" / "replay"
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in args.partial:
            cfg = {"min_silence": args.silence, "max_chunk": args.max_chunk, "partial_interval": p,
                   "piece_gap": args.piece_gap, "piece_max": args.piece_max,
                   "save_audio": False}  # 回放的是现成文件，服务端不用再存一份录音
            print(f"[实时出字 {p:g}s] 推流 {len(pcm) / SR:.0f}s 音频 …", flush=True)
            ready, events = asyncio.run(stream(args.url, pcm, cfg, args.speed))
            run = {"type": "_run", "wav": str(wav), "cfg": cfg, "ready": ready, "dur": len(pcm) / SR,
                   "speed": args.speed, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            f = out_dir / f"{wav.stem}_p{p:g}_g{args.piece_gap:g}_{time.strftime('%H%M%S')}.jsonl"
            f.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in [run] + events) + "\n", encoding="utf-8")
            print(f"  事件存到 {f.relative_to(ROOT)}", flush=True)
            runs.append({**run, "events": events})

    spans, ref = load_ref(wav)
    stats = [analyze(r, spans, ref) for r in runs]
    try:
        api = args.url.replace("ws://", "http://").replace("wss://", "https://").split("/ws/")[0] + "/api/models"
        threads = json.load(urllib.request.urlopen(api, timeout=5))["threads"]
    except Exception:  # noqa: BLE001 —— 只读存档时服务可能没开
        threads = "?"
    text = report(runs, stats, wav, len(spans) if spans else 0, threads)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print("\n" + text)
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
