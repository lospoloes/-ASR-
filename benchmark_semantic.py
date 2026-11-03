"""语义切段 vs 纯声学切段 对比评测。

数据:
    benchmark/semantic/session*.wav + gt.json  （make_semantic_testset.py 生成，句子边界已知）
    test_audio/lei-jun-test.wav + benchmark/long/refs.tsv（可选，真实长音频，只有文本没有句子边界）
用法:
    python benchmark_semantic.py
    python benchmark_semantic.py --leijun test_audio/lei-jun-test.wav
结果写到 benchmark/results/semantic.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import asr_engine
import registry
import semantic_seg as ss

ROOT = Path(__file__).resolve().parent


def load_punct():
    files = registry.resolve_files(registry.get_spec("punct-ct-transformer"))
    if not files:
        return None
    import sherpa_onnx as so
    cfg = so.OfflinePunctuationConfig(model=so.OfflinePunctuationModelConfig(ct_transformer=str(files["model"]),
                                                                             num_threads=1))
    p = so.OfflinePunctuation(cfg)
    return p.add_punctuation


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def score(utts, gt):
    """切断：一句被切成多段；粘连：一段里含多句；完全正确：一句恰好对应一段且该段只含这一句。"""
    spans = ss.spans(utts)
    split = merged_utts = exact = 0
    for g in gt:
        hit = [u for u in spans if overlap(u, (g["start"], g["end"])) > 0.1]
        if len(hit) > 1:
            split += 1
        if len(hit) == 1 and sum(overlap(hit[0], (h["start"], h["end"])) > 0.1 for h in gt) == 1:
            exact += 1
    for u in spans:
        if sum(overlap(u, (g["start"], g["end"])) > 0.1 for g in gt) > 1:
            merged_utts += 1
    # 句末等待：每个真实句尾，系统在断句前要等多久静音（只统计在该句尾断开的情况）
    # 真实等待 = 系统决定断句的时刻 − 用户实际说完的时刻
    # （决定时刻 = VAD 段尾 + 断句阈值；VAD 段尾本身就比实际说完晚 0.15~0.5s）
    waits = []
    for g in gt:
        ends = [u for u in utts if abs(u.end - g["end"]) < 0.8 and u.start < g["end"]]
        if ends:
            u = min(ends, key=lambda u: abs(u.end - g["end"]))
            waits.append(u.end + u.wait - g["end"])
    return {"split": split, "merged": merged_utts, "exact": exact, "n": len(gt),
            "wait": float(np.mean(waits)) if waits else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "benchmark" / "semantic"))
    ap.add_argument("--leijun", default=str(ROOT / "test_audio" / "lei-jun-test.wav"))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    downloaded = asr_engine.find_downloaded()
    models = {}
    for key in ("sensevoice", "paraformer-zh-small"):
        if key in downloaded:
            m = asr_engine.AsrModel(registry.get_spec(key), downloaded[key], args.threads)
            m.load()
            models[key] = m
    if len(models) < 2:
        print("需要 sensevoice 和 paraformer-zh-small：python download_models.py")
        return 1
    if not registry.resolve_files(registry.get_spec("silero-vad")):
        print("需要 silero-vad：python download_models.py")
        return 1
    punct = load_punct()
    sv, pf = models["sensevoice"], models["paraformer-zh-small"]
    sv_fn = lambda x: sv._offline_stream(sv.recognizer, 16000, x)["text"]  # noqa: E731
    pf_fn = lambda x: pf._offline_stream(pf.recognizer, 16000, x)["text"]  # noqa: E731

    def methods(x, frags, fine):
        out = {
            "VAD 原样（0.25s 即切）": (ss.fixed_gap_merge(frags, 0.0), None),
            "固定静音 0.6s": (ss.fixed_gap_merge(frags, 0.6), None),
            "固定静音 1.2s": (ss.fixed_gap_merge(frags, 1.2), None),
        }
        # 语义模式用细切的 VAD（min_silence=0.1），合并交给语义层
        out["语义·SenseVoice"] = ss.segment(x, fine, sv_fn)
        out["语义·Paraformer 规则"] = ss.segment(x, fine, pf_fn)
        if punct:
            out["语义·Paraformer+标点模型"] = ss.segment(x, fine, pf_fn, punct_fn=punct)
        return out

    gt_all = json.loads((Path(args.dir) / "gt.json").read_text(encoding="utf-8"))
    agg: dict = {}
    examples = []
    total_dur = 0.0
    for name, gt in gt_all.items():
        x, sr = asr_engine.read_wav((Path(args.dir) / name).read_bytes())
        x = asr_engine.resample(x, sr)
        total_dur += len(x) / 16000
        frags, _ = asr_engine.compute_vad(x)
        fine, _ = asr_engine.compute_vad(x, min_silence=0.1)
        ref = "".join(g["text"] for g in gt)
        for mname, (utts, stats) in methods(x, frags, fine).items():
            sc = score(utts, gt)
            a = agg.setdefault(mname, {"split": 0, "merged": 0, "exact": 0, "n": 0, "waits": [], "n_utts": 0,
                                       "dec_ms": 0.0, "err": {k: 0 for k in models}, "chars": 0})
            for k in ("split", "merged", "exact", "n"):
                a[k] += sc[k]
            if sc["wait"] is not None:
                a["waits"].append(sc["wait"])
            a["n_utts"] += len(utts)
            a["dec_ms"] += (stats or {}).get("total_ms", 0.0)
            rn = asr_engine.normalize_text(ref, True)
            a["chars"] += len(rn)
            for k, m in models.items():
                hyp = m.transcribe(x, segs=ss.spans(utts))["text"]
                a["err"][k] += asr_engine.edit_distance(rn, asr_engine.normalize_text(hyp, True))
            if name == "session1.wav" and mname in ("VAD 原样（0.25s 即切）", "固定静音 1.2s", "语义·Paraformer 规则"):
                for u in utts[:6]:
                    t = "".join(u.frag_texts) if u.frag_texts else ""
                    examples.append((mname, f"{u.start:.1f}–{u.end:.1f}s", u.reason or "gap", round(u.wait, 2), t))
        print(f"{name}: VAD 片段 {len(frags)} 个（细切 {len(fine)} 个）", flush=True)

    lines = ["# 语义切段评测", "",
             f"- 合成测试集 `benchmark/semantic/`：3 段对话共 {sum(len(v) for v in gt_all.values())} 句、"
             f"{sum(g['n_pauses'] for v in gt_all.values() for g in v)} 处句内思考停顿（0.4–0.9s），"
             f"句间停顿 0.35–1.6s，总时长 {total_dur:.1f}s",
             "- 切断：一句被切成多段（机器人会在用户没说完时抢话）；粘连：一段里含多句（响应变慢、意图混在一起）",
             "- 句末等待：从用户实际说完，到系统决定断句的真实时间（含 VAD 自身判停的延迟；越短响应越快）",
             "- VAD 原样/固定静音用 min_silence=0.25 的 VAD；语义切段用 min_silence=0.1 细切，再由语义层合并",
             f"- 标点模型：{'已加载（CT-Transformer int8，72MB）' if punct else '未下载，跳过该组'}", "",
             "| 切段方式 | 完全切对 | 切断 | 粘连 | 句末平均等待 | 决策耗时 | SenseVoice CER | Paraformer CER |",
             "|---|---|---|---|---|---|---|---|"]
    for mname, a in agg.items():
        w = f"{np.mean(a['waits']):.2f}s" if a["waits"] else "—"
        dec = f"{a['dec_ms'] / total_dur:.0f} ms/s音频" if a["dec_ms"] else "—"
        lines.append(f"| {mname} | {a['exact']}/{a['n']} | {a['split']} | {a['merged']} | {w} | {dec} | "
                     f"{a['err']['sensevoice'] / a['chars'] * 100:.2f}% | "
                     f"{a['err']['paraformer-zh-small'] / a['chars'] * 100:.2f}% |")
    lines += ["", "## session1 切段示例（前 6 段）", "", "| 方式 | 时间 | 断句依据 | 等待(s) | 片段文字 |", "|---|---|---|---|---|"]
    lines += [f"| {m} | {t} | {r} | {w} | {txt} |" for m, t, r, w, txt in examples]

    # 真实长音频：没有句子边界，只看段数/段长/CER
    lj = Path(args.leijun)
    refs_f = ROOT / "benchmark" / "long" / "refs.tsv"
    if lj.exists() and refs_f.exists():
        x, sr = asr_engine.read_wav(lj.read_bytes())
        x = asr_engine.resample(x, sr)
        ref = refs_f.read_text(encoding="utf-8").split("\t", 1)[1]
        rn = asr_engine.normalize_text(ref, True)
        frags, _ = asr_engine.compute_vad(x)
        fine, _ = asr_engine.compute_vad(x, min_silence=0.1)
        lines += ["", f"## 真实长音频（lei-jun-test.wav，{len(x) / 16000:.0f}s，VAD 片段 {len(frags)} 个）", "",
                  "| 切段方式 | 段数 | 平均段长 | 最长段 | SenseVoice CER | Paraformer CER |", "|---|---|---|---|---|---|"]
        for mname, (utts, _) in methods(x, frags, fine).items():
            lens = [u.end - u.start for u in utts]
            cells = []
            for k, m in models.items():
                hyp = m.transcribe(x, segs=ss.spans(utts))["text"]
                cells.append(f"{asr_engine.edit_distance(rn, asr_engine.normalize_text(hyp, True)) / len(rn) * 100:.2f}%")
            lines.append(f"| {mname} | {len(utts)} | {np.mean(lens):.1f}s | {max(lens):.1f}s | {cells[0]} | {cells[1]} |")
            print(f"lei-jun {mname}: {len(utts)} 段", flush=True)

    out = ROOT / "benchmark" / "results" / "semantic.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:16]))
    print(f"\n结果: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
