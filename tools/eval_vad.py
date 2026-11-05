"""VAD 评测：在 MagicData-RAMC 整段对话的人工标注上比较 VAD 参数，按数据定默认值。

先下载：python tools/fetch_ramc.py --count 3（整段对话在 test_audio/ramc/full/，每段约 30 分钟；许可证不允许提交）。

用法:
    python tools/eval_vad.py                                   # full/ 里全部对话 × 默认几组配置 → benchmark/results/vad_eval.md
    python tools/eval_vad.py --try min_speech=0.15,threshold=0.45 --try agc=0   # 另外加几组（相对现状改哪几项）
    python tools/eval_vad.py --only-try --try piece_gap=0.25   # 只跑现状、新默认和 --try 的几组

每段对话的逐帧人声概率只算一遍（自动增益开 / 关各一遍），各组参数拿同一串概率重放切分（vad_stream.Segmenter），
所以参数随便加，一组几秒。报告里只有统计数字，没有 RAMC 的文本。

指标（都是音频时间；人工标注的起止有零点几秒误差，只看中位数、P90 和各组之间的差）:
  开口延迟    前面 ≥1 秒没有任何声音标注的句子：标注起点 → VAD 确认开口（发出 start 事件）
  判停延迟    后面 ≥1 秒没有任何声音标注的句子：标注终点 → VAD 判定说完（发出 end 事件）
  漏检        标注句被 VAD 的句覆盖不到一半；<0.8 秒的短句（"嗯""对""好"这类）单独计
  误检        VAD 判为说话、离所有声音标注（含笑声、听不清）都超过 0.3 秒的时长，折算成每小时多少秒
  混说话人    VAD 的一句里有两个说话人各 >0.1 秒的话（与 live_sim.py --real 的"一段含两人"同一口径）
  切段 / 算力 每句几段、段长；"句尾空切" = 停顿切完这句就结束了（那次段锁定白做）；
              额外解码 = 每 0.5 秒（页面默认，--interval 可改）蹦字一次时（蹦字 + 段锁定）/ 正式结果 的 SenseVoice 解码量，
              按实时回放实测拟合的 28ms + 16ms/秒音频（M4 Pro）估，假设转写线程跟得上（跟不上时引擎丢过期蹦字，只会更少）
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import asr_engine  # noqa: E402
import fetch_ramc  # noqa: E402
import vad_stream  # noqa: E402
from vad_stream import SR  # noqa: E402

FULL = ROOT / "test_audio" / "ramc" / "full"
BASE = {"threshold": 0.5, "neg_threshold": None, "min_speech": 0.25, "min_silence": 0.5, "sentence_max": 15.0,
        "piece_gap": 0.2, "piece_min": 1.0, "piece_max": 8.0, "agc": True}
CONFIGS = [  # (名称, 相对现状改了哪几项)
    ("现状（= sherpa 默认）", {}),
    ("阈值 0.4", {"threshold": 0.4}),
    ("阈值 0.6", {"threshold": 0.6}),
    ("最短开口 0.1s", {"min_speech": 0.1}),
    ("最短开口 0.15s", {"min_speech": 0.15}),
    ("静音阈值 0.45", {"neg_threshold": 0.45}),
    ("静音阈值 0.5", {"neg_threshold": 0.5}),
    ("句末静音 0.3s", {"min_silence": 0.3}),
    ("句末静音 0.8s", {"min_silence": 0.8}),
    ("段停顿 0.15s", {"piece_gap": 0.15}),
    ("段停顿 0.3s", {"piece_gap": 0.3}),
    ("不加自动增益", {"agc": False}),
    (NEW := "新默认：最短开口 0.15s + 静音阈值 0.5", {"min_speech": 0.15, "neg_threshold": 0.5}),
]
GAP = 1.0        # 开口 / 判停延迟只统计前 / 后这么多秒内没有任何声音标注的句子
COLLAR = 0.3     # 误检：离声音标注超过这么远才算
SHORT = 0.8      # 短句
PAD = 0.2        # 和 live_engine 一样：送去识别的句子前后各补 0.2 秒（超长硬切的切点处不补）
INTERVAL = 0.5   # 估算力用的蹦字间隔（页面默认；--interval 可改）
RES = 100        # 覆盖率按 10ms 一格算


def dec_ms(n: int) -> float:
    """SenseVoice-Small int8 一次解码的耗时估计：M4 Pro、2 线程，tools/replay_live.py 实时回放里 366 次蹦字识别的线性拟合
    （离线连续跑时固定开销只有约 11ms；实时链路里识别是零散的，每次多出十几毫秒）。"""
    return 28.0 + 16.0 * max(n, 0) / SR


def probs_of(wav: str, agc: bool) -> np.ndarray:
    x, sr = asr_engine.read_wav(Path(wav).read_bytes())
    return vad_stream.frame_probs(asr_engine.resample(x, sr), agc=agc)


# ---------------------------------------------------------------- 重放：句、段、估算力
def run(probs: np.ndarray, cfg: dict, interval: float = INTERVAL) -> dict:
    seg = vad_stream.Segmenter(**{k: v for k, v in cfg.items() if k != "agc"})
    thr, pad = seg.threshold, int(PAD * SR)
    sents: list[dict] = []
    cur: dict | None = None
    cost = {"live": 0.0, "lock": 0.0, "final": 0.0}
    n_live = live_max = 0
    piece_lo = sent_lo = last_partial = 0
    last_voiced = -1

    def handle(e: vad_stream.Event, t: int) -> None:
        nonlocal cur, piece_lo, sent_lo, last_partial
        if e.kind == "start":
            cur = {"start": e.at, "open": t, "how": e.reason, "cuts": []}
            piece_lo = sent_lo = max(0, e.at - pad) if e.reason == "speech" else e.at
            last_partial = e.at
        elif e.kind == "cut":
            cur["cuts"].append((e.at, e.reason, t))
            cost["lock"] += dec_ms(e.hi - piece_lo)
            piece_lo = last_partial = e.at
        else:
            last = cur["cuts"][-1] if cur["cuts"] else None
            trailing = e.reason == "silence" and last is not None and last[1] == "pause" and last_voiced <= last[2]
            cur.update(end=e.at, close=t, reason=e.reason, trailing=trailing)
            cost["final"] += dec_ms((e.at if e.reason == "max_len" else e.at + pad) - sent_lo)
            sents.append(cur)
            cur = None

    for p in probs:
        p = float(p)
        evs = seg.step(p)
        t = seg.cur
        if p > thr:
            last_voiced = t
        for e in evs:
            handle(e, t)
        # 和 LiveSession._tick / _queue_partial 同一条件：这段已有 0.2 秒人声、离上次 ≥ 间隔、音频 ≥ 0.5 秒
        if (seg.in_speech and seg.voiced_from != -1 and t - seg.voiced_from >= SR // 5
                and t - last_partial >= interval * SR and t - piece_lo >= SR // 2):
            cost["live"] += dec_ms(t - piece_lo)
            n_live += 1
            live_max = max(live_max, t - piece_lo)
            last_partial = t
    for e in seg.flush():
        handle(e, seg.cur)
    return {"sents": sents, "cost": cost, "n_live": n_live, "live_max": live_max / SR, "frames": len(probs)}


# ---------------------------------------------------------------- 对照人工标注
def score(r: dict, speech: list, voice: list) -> dict:
    dur = r["frames"] * vad_stream.WIN / SR
    n = int(dur * RES) + 1

    def mask(iv, ext: float = 0.0) -> np.ndarray:
        m = np.zeros(n, bool)
        for s, e in iv:
            m[max(0, int((s - ext) * RES)):max(0, min(n, int(np.ceil((e + ext) * RES))))] = True
        return m

    S = [(x["start"] / SR, x["end"] / SR, x["open"] / SR, x["close"] / SR, x["how"], x["reason"]) for x in r["sents"]]
    vad = mask([(s, e) for s, e, *_ in S])
    sp = mask([(s, e) for s, e, *_ in speech])
    near = mask([(s, e) for s, e, *_ in voice], COLLAR)
    out = {"dur": dur, "n_speech": len(speech), "vad_frac": vad.sum() / n, "lab_frac": mask([(s, e) for s, e, *_ in voice]).sum() / n,
           "recall": (vad & sp).sum() / max(sp.sum(), 1), "fa": (vad & ~near).sum() / RES, "n_sent": len(S)}

    Ss, Se = np.array([x[0] for x in S]), np.array([x[1] for x in S])
    ps, pe = np.array([r[0] for r in speech]), np.array([r[1] for r in speech])
    spk = np.array([r[2] for r in speech])
    missed = missed_short = n_short = split = 0
    for s, e, *_ in speech:
        a = int(s * RES)
        short = e - s < SHORT
        n_short += short
        if vad[a:max(a + 1, int(e * RES))].mean() < 0.5:
            missed += 1
            missed_short += short
        split += int(((np.minimum(e, Se) - np.maximum(s, Ss)) > 0.1).sum() > 1)
    out.update(missed=missed, missed_short=missed_short, n_short=n_short, split=split)
    # 误触发：VAD 的一句有一半以上离声音标注 >COLLAR 秒（白跑一次识别，可能出乱字）
    out["false_trig"] = sum(near[int(s * RES):max(int(s * RES) + 1, int(e * RES))].mean() < 0.5 for s, e, *_ in S)
    out["mixed"] = sum(len(set(spk[(np.minimum(x[1], pe) - np.maximum(x[0], ps)) > 0.1])) > 1 for x in S)

    vs = np.array([v[0] for v in voice])
    ve = np.array([v[1] for v in voice])
    opens = np.array([x[2] for x in S])
    closes = np.array([x[3] for x in S])
    first_open = np.array([x[2] for x in S if x[4] == "speech"])

    def active(t: float) -> bool:   # t 时刻 VAD 已经确认开口、还没判停
        return bool(((opens <= t) & (t < closes)).any())

    on, on_pre, on_miss, n_on = [], 0, 0, 0
    off, off_early, off_merged, off_cap, n_off = [], 0, 0, 0, 0
    for s, e, spk, _ in speech:
        me = (vs == s) & (ve == e)
        if not ((vs < s) & (ve > s - GAP) & ~me).any():          # 前面 GAP 秒内没人出声：算一次"开口"
            n_on += 1
            if active(s - 0.3):
                on_pre += 1                                         # VAD 早就开着（上一句没收 / 噪声 / 标注起点偏晚）
            else:
                c = first_open[(first_open >= s - 0.3) & (first_open <= e + 0.3)]
                if len(c):
                    on.append(c.min() - s)
                else:
                    on_miss += 1
        if not ((vs < e + GAP) & (ve > e) & ~me).any():            # 后面 GAP 秒内没人出声：算一次"说完"
            n_off += 1
            ov = np.flatnonzero((Ss < e) & (Se > s))
            if not len(ov):
                off_early += 1
                continue
            x = S[ov[-1]]                                           # 盖住这句的最后一个 VAD 句
            nxt = vs[vs >= e]
            if x[5] == "max_len":
                off_cap += 1
            elif x[3] < e - 0.3:
                off_early += 1                                      # VAD 在标注终点之前就判停了（句尾没收全）
            elif len(nxt) and x[3] > nxt.min():
                off_merged += 1                                     # 空隙里没判停，和下一句连成一句
            else:
                off.append(x[3] - e)
    out.update(on=on, on_pre=on_pre, on_miss=on_miss, n_on=n_on,
               off=off, off_early=off_early, off_merged=off_merged, off_cap=off_cap, n_off=n_off)

    pieces, pause_cuts, cap_cuts, trailing, cap_sents = [], 0, 0, 0, 0
    for x in r["sents"]:
        cuts = x["cuts"]
        pause_cuts += sum(c[1] == "pause" for c in cuts)
        cap_cuts += sum(c[1] == "max_len" for c in cuts)
        cap_sents += x["reason"] == "max_len"
        bounds = [x["start"]] + [c[0] for c in cuts] + [x["end"]]
        lens = [(b - a) / SR for a, b in zip(bounds, bounds[1:])]
        if x["trailing"]:
            trailing += 1
            lens = lens[:-1]                                         # 句尾空切留下的那截静音不算一段
        pieces.append(lens)
    out.update(pieces=pieces, pause_cuts=pause_cuts, cap_cuts=cap_cuts, trailing=trailing, cap_sents=cap_sents,
               cost=r["cost"], n_live=r["n_live"], live_max=r["live_max"])
    return out


# ---------------------------------------------------------------- 报告
def pct(xs, q) -> float:
    return float(np.percentile(xs, q)) if len(xs) else float("nan")


def merge(parts: list[dict]) -> dict:
    m: dict = {}
    for k in parts[0]:
        v = [p[k] for p in parts]
        if isinstance(v[0], list):
            m[k] = sum(v, [])
        elif isinstance(v[0], dict):
            m[k] = {kk: sum(p[k][kk] for p in parts) for kk in v[0]}
        elif k in ("vad_frac", "lab_frac", "recall"):
            m[k] = float(np.average(v, weights=[p["dur"] for p in parts]))
        elif k == "live_max":
            m[k] = max(v)
        else:
            m[k] = sum(v)
    return m


def report(results: list[tuple[str, dict, dict]], per_conv: list[tuple[str, str, dict]], convs: list[str],
           interval: float = INTERVAL) -> list[str]:
    base = results[0][2]
    hours = base["dur"] / 3600
    lines = ["# VAD 评测：真人双人对话（MagicData-RAMC 整段）", "",
             f"- 数据：{len(convs)} 段 dev 集对话，共 {base['dur'] / 60:.0f} 分钟、{base['n_speech']} 句人工标注"
             f"（其中 <{SHORT}s 的短句 {base['n_short']} 句）；标注里有声音的时间占 {base['lab_frac']:.0%}。"
             "音频和文本只在本机（CC BY-NC-ND 4.0），这里只有统计数字",
             "- 方法：`tools/eval_vad.py`。逐帧人声概率用 `vad_stream`（与 sherpa-onnx 逐帧等价）算一遍，各组参数重放切分；"
             "现状 = 阈值 0.5、静音阈值 0.35（= 阈值 − 0.15）、最短开口 0.25s、句末静音 0.5s、句长上限 15s、"
             "段停顿 0.2s / 段 ≥1s / 段长上限 8s、自动增益开",
             f"- 开口延迟只统计前面 ≥{GAP:.0f}s 没人出声的句子（{base['n_on']} 句），判停延迟只统计后面 ≥{GAP:.0f}s 没人出声的句子"
             f"（{base['n_off']} 句）；标注起止本身有零点几秒误差，看中位数、P90 和组间差",
             f"- 误检：VAD 判为说话、离所有声音标注（含笑声、听不清）都 >{COLLAR}s 的时长；漏检：标注句被覆盖不到一半",
             f"- 估算算力：蹦字每 {interval}s 一次，SenseVoice 每次解码按 28ms + 16ms/秒音频（M4 Pro 实时回放实测拟合），"
             "额外解码 =（蹦字 + 段锁定）/ 正式结果；假设转写线程跟得上", "",
             "## 检测", "",
             "| 配置 | 开口延迟 中位 / P90 | 判停延迟 中位 / P90 | 漏检句（其中短句） | 标注语音覆盖 | 误检（秒/小时） "
             "| 误触发句 | 判为说话 | VAD 句数 | 混说话人句 | 一句被切开 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, _cfg, m in results:
        lines.append(
            f"| {name} | {pct(m['on'], 50):.2f} / {pct(m['on'], 90):.2f} s | {pct(m['off'], 50):.2f} / {pct(m['off'], 90):.2f} s "
            f"| {m['missed']}（{m['missed_short']}） | {m['recall']:.1%} | {m['fa'] / hours:.0f} | {m['false_trig']} | {m['vad_frac']:.0%} "
            f"| {m['n_sent']} | {m['mixed']}（{m['mixed'] / max(m['n_sent'], 1):.0%}） | {m['split']} |")
    lines += ["", "开口 / 判停没法量的句子（不计入上表延迟）：", "",
              "| 配置 | 开口：VAD 已经开着 | 开口：没检测到 | 判停：标注终点前就判停 | 判停：没判停、并进下一句 | 判停：超长硬切 |",
              "|---|---|---|---|---|---|"]
    for name, _cfg, m in results:
        lines.append(f"| {name} | {m['on_pre']} | {m['on_miss']} | {m['off_early']} | {m['off_merged']} | {m['off_cap']} |")
    lines += ["", "## 切段与算力（实时出字）", "",
              "| 配置 | 每句段数（平均） | 段长 中位 / P90 / 最长 | 停顿切 / 超长切 | 句尾空切 | 句长到上限 | 蹦字单次最长音频 "
              "| 估算额外解码 | 估算转写占用 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, _cfg, m in results:
        lens = sum(m["pieces"], [])
        c = m["cost"]
        lines.append(
            f"| {name} | {np.mean([len(p) for p in m['pieces']]):.2f} | {pct(lens, 50):.1f} / {pct(lens, 90):.1f} / {max(lens):.1f} s "
            f"| {m['pause_cuts']} / {m['cap_cuts']} | {m['trailing']}（{m['trailing'] / max(m['pause_cuts'], 1):.0%}） "
            f"| {m['cap_sents']} | {m['live_max']:.1f} s | {(c['live'] + c['lock']) / c['final']:.1f} 倍 "
            f"| {sum(c.values()) / (m['dur'] * 1000):.1%} |")
    lines += ["", "## 各段对话：现状 → 新默认", "",
              "| 对话 | 配置 | VAD 句 | 漏检（短句） | 误检（秒/小时） | 混说话人句 | 并进下一句 / 超长硬切 "
              "| 开口延迟 中位 | 判停延迟 中位 / P90 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for conv, name, m in sorted(per_conv, key=lambda x: x[0]):
        lines.append(f"| {conv}（{m['dur'] / 60:.0f} 分钟，{m['n_speech']} 句） | {name.split('：')[0]} | {m['n_sent']} "
                     f"| {m['missed']}（{m['missed_short']}） | {m['fa'] / (m['dur'] / 3600):.0f} | {m['mixed']} "
                     f"| {m['off_merged']} / {m['off_cap']} | {pct(m['on'], 50):.2f} s "
                     f"| {pct(m['off'], 50):.2f} / {pct(m['off'], 90):.2f} s |")
    return lines


def parse_try(s: str) -> tuple[str, dict]:
    cfg = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip()
        if k not in BASE:
            raise SystemExit(f"未知参数 {k}：可选 {', '.join(BASE)}")
        cfg[k] = v.strip().lower() in ("1", "true", "on", "yes") if k == "agc" else float(v)
    return s, cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", nargs="*", help="对话 ID（默认 full/ 里全部）")
    ap.add_argument("--try", dest="tries", action="append", default=[], help="另加一组：k=v[,k=v]，相对现状")
    ap.add_argument("--only-try", action="store_true", help="只跑现状、新默认和 --try 的几组")
    ap.add_argument("--interval", type=float, default=INTERVAL, help="估算力用的蹦字间隔（秒）")
    ap.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1), help="算人声概率的进程数")
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "results" / "vad_eval.md"))
    args = ap.parse_args()

    wavs = [FULL / f"{c}.wav" for c in args.conv] if args.conv else sorted(FULL.glob("*.wav"))
    wavs = [w for w in wavs if w.exists() and w.with_suffix(".txt").exists()]
    if not wavs:
        print("没有数据：先 python tools/fetch_ramc.py --count 3")
        return 1
    configs = ([CONFIGS[0], CONFIGS[-1]] if args.only_try else CONFIGS) + [parse_try(t) for t in args.tries]
    agcs = sorted({BASE["agc"] if "agc" not in c else c["agc"] for _, c in configs}, reverse=True)

    t0 = time.time()
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {(w.stem, agc): ex.submit(probs_of, str(w), agc) for w in wavs for agc in agcs}
        probs = {k: f.result() for k, f in futs.items()}
    print(f"人声概率：{len(wavs)} 段 × 自动增益 {agcs}，{time.time() - t0:.0f}s", flush=True)

    labels = {w.stem: (fetch_ramc.parse(w.with_suffix(".txt")), fetch_ramc.parse(w.with_suffix(".txt"), nonspeech=True))
              for w in wavs}
    results, per_conv = [], []
    for name, over in configs:
        cfg = {**BASE, **over}
        parts = []
        for w in wavs:
            m = score(run(probs[(w.stem, cfg["agc"])], cfg, args.interval), *labels[w.stem])
            parts.append(m)
            if not over or name == NEW:
                per_conv.append((w.stem, name, m))
        m = merge(parts)
        results.append((name, cfg, m))
        print(f"{name}: 句 {m['n_sent']}，漏检 {m['missed']}（短 {m['missed_short']}），误检 {m['fa']:.0f}s，"
              f"开口 {pct(m['on'], 50):.2f}s，判停 {pct(m['off'], 50):.2f}s，额外解码 "
              f"{(m['cost']['live'] + m['cost']['lock']) / m['cost']['final']:.1f} 倍", flush=True)

    lines = report(results, per_conv, [w.stem for w in wavs], args.interval)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
