"""持续监听场景下分别评测 SenseVoice 和 Paraformer：把音频按流式方式喂给 LiveSession，统计每个模型的 CER 和延迟。

和单句评测的区别：这里的 chunk 是 VAD 在“持续聊天”里实时切出来的（和网页 /live 完全同一套逻辑），
两个模型拿到完全相同的 chunk；延迟按“单独部署”计算。

数据（有哪个跑哪个）:
    benchmark/semantic/session*.wav + gt.json        合成对话，句子边界已知 → 可算真实延迟
    benchmark/testset/dialog_2spk.wav + .tsv          双人交替对话，句子边界已知
    test_audio/lei-jun-test.wav + benchmark/long/refs.tsv   真实演讲 272s，只有文本
    test_audio/ramc/*.wav + 同名 .tsv                 真人双人对话（--real，先 python tools/fetch_ramc.py）
用法:
    python live_sim.py                     # 静音阈值 0.3 / 0.5 / 0.8 各跑一遍 → benchmark/results/live.md
    python live_sim.py --silence 0.5
    python live_sim.py --real              # 只跑真人双人对话 → benchmark/results/live_real.md
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import asr_engine
import live_engine
import registry

ROOT = Path(__file__).resolve().parent
SR = 16000


def load_real_sets():
    """真人双人对话：tools/fetch_ramc.py 截出的 wav + tsv（起、止、说话人、文本；时间为人工标注）。"""
    sets = []
    for wav in sorted((ROOT / "test_audio" / "ramc").glob("*.wav")):
        tsv = wav.with_suffix(".tsv")
        if not tsv.exists():
            continue
        rows = [ln.split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines() if ln.strip()]
        sets.append({"name": f"真人双人对话 {wav.stem}", "wav": wav, "ref": "".join(r[3] for r in rows),
                     "ends": [float(r[1]) for r in rows], "spans": [(float(r[0]), float(r[1])) for r in rows],
                     "spk": [r[2] for r in rows]})
    return sets


def machine() -> str:
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or cpu
        except Exception:  # noqa: BLE001
            pass
    return f"{platform.system()} {platform.machine()}（{cpu}）"


def load_sets():
    sets = []
    sem = ROOT / "benchmark" / "semantic"
    if (sem / "gt.json").exists():
        gt = json.loads((sem / "gt.json").read_text(encoding="utf-8"))
        for name, items in gt.items():
            sets.append({"name": f"合成对话 {name}", "wav": sem / name, "ref": "".join(i["text"] for i in items),
                         "ends": [i["end"] for i in items], "spans": [(i["start"], i["end"]) for i in items]})
    d = ROOT / "benchmark" / "testset"
    if (d / "dialog_2spk.tsv").exists():
        rows = [ln.split("\t") for ln in (d / "dialog_2spk.tsv").read_text(encoding="utf-8").splitlines() if ln.strip()]
        # 该文件的句子时间按 TTS 原始输出计（含首尾静音，未裁剪），句尾比实际说完晚 → 不用于延迟统计
        sets.append({"name": "双人交替对话 dialog_2spk", "wav": d / "dialog_2spk.wav", "ref": "".join(r[3] for r in rows),
                     "ends": None, "spans": [(float(r[0]), float(r[1])) for r in rows]})
    lj, refs = ROOT / "test_audio" / "lei-jun-test.wav", ROOT / "benchmark" / "long" / "refs.tsv"
    if lj.exists() and refs.exists():
        sets.append({"name": "雷军演讲 272s（真人）", "wav": lj, "ref": refs.read_text(encoding="utf-8").split("\t", 1)[1],
                     "ends": None, "spans": None})
    return sets


def simulate(models, x, silence, max_chunk):
    events = []
    sess = live_engine.LiveSession(models, events.append, min_silence=silence, max_chunk=max_chunk)
    for i in range(0, len(x), 1600):  # 0.1s 一块，和网页推流粒度一致；不 sleep，按最快速度跑
        sess.accept(x[i:i + 1600])
    sess.close()
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--silence", type=float, nargs="*", default=[0.3, 0.5, 0.8])
    ap.add_argument("--max-chunk", type=float, default=15.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--real", action="store_true",
                    help="只跑真人双人对话（test_audio/ramc/，先 python tools/fetch_ramc.py），结果写到 live_real.md")
    args = ap.parse_args()

    downloaded = asr_engine.find_downloaded()
    models = {}
    for key in ("sensevoice", "paraformer-zh-small"):
        if key in downloaded:
            m = asr_engine.AsrModel(registry.get_spec(key), downloaded[key], args.threads)
            m.load()
            models[key] = m
    if not models or asr_engine.vad_model_path() is None:
        print("需要模型：python download_models.py")
        return 1
    names = {k: m.spec["name"] for k, m in models.items()}
    sets = load_real_sets() if args.real else load_sets()
    if not sets:
        print("没有数据：先 python tools/fetch_ramc.py" if args.real else "没有数据")
        return 1
    common = ["- 音频以 0.1s 为单位流式喂给 `live_engine.LiveSession`（与网页 /live 同一套逻辑），VAD 实时切 chunk，"
              "两个模型拿到完全相同的 chunk"]
    if args.real:
        lines = ["# 持续监听场景：真人双人对话（MagicData-RAMC）", "",
                 "- 数据：MagicData-RAMC dev 集，两人面对面自由聊天、手机录制（16kHz 单声道），人工转写到句，"
                 "带起止时间和说话人。`python tools/fetch_ramc.py` 下载并截取；许可证 CC BY-NC-ND 4.0，音频和文本只在本机，不提交",
                 *common,
                 "- **真实延迟** = 人工标注的句子结束时刻 → 该模型出字（= VAD 判停等待 + 本模型解码，按单独部署计）；"
                 "只统计 VAD 正好在这句话结尾切开的句子（括号里是句数），和下一句挨得太近、被并进同一个 chunk 的不计",
                 "- **一段含两人**：一个 chunk 里同时有两个说话人的话（交给大模型时分不清是谁说的）"]
    else:
        lines = ["# 持续监听场景：SenseVoice vs Paraformer", "", *common,
                 "- **真实延迟** = 用户实际说完 → 该模型出字（= VAD 判停等待 + 本模型解码，按单独部署计）；"
                 "只在句尾精确已知的合成对话（已裁掉 TTS 首尾静音）上统计"]
    lines += ["- CER：字级编辑距离，去标点空格，数字归一化",
              f"- 最长 chunk {args.max_chunk}s；机器：{machine()}，每个模型 {args.threads} 线程", ""]

    for silence in args.silence:
        lines += [f"## 静音阈值 {silence}s", "",
                  "| 数据 | chunk 数 | 平均/最长 chunk | 一句被切成多段 | 一段含多句 | " + ("一段含两人 | " if args.real else "")
                  + " | ".join(f"{names[k]} CER | {names[k]} 真实延迟（平均/P90）" for k in models) + " |",
                  "|---|---|---|---|---|" + ("---|" if args.real else "") + "---|---|" * len(models)]
        tot = {k: [0, 0, []] for k in models}
        for ds in sets:
            x, sr = asr_engine.read_wav(ds["wav"].read_bytes())
            x = asr_engine.resample(x, sr)
            evs = simulate(models, x, silence, args.max_chunk)
            chunks = [e for e in evs if e["type"] == "chunk"]
            durs = [c["dur"] for c in chunks]
            split = merged = mixed = "—"
            if ds["spans"]:
                ov = lambda a, b: max(0.0, min(a[1], b[1]) - max(a[0], b[0]))  # noqa: E731
                split = sum(sum(ov((c["start"], c["end"]), s) > 0.1 for c in chunks) > 1 for s in ds["spans"])
                merged = sum(sum(ov((c["start"], c["end"]), s) > 0.1 for s in ds["spans"]) > 1 for c in chunks)
                if ds.get("spk"):
                    mixed = sum(len({p for s, p in zip(ds["spans"], ds["spk"]) if ov((c["start"], c["end"]), s) > 0.1}) > 1
                                for c in chunks)
            rn = asr_engine.normalize_text(ds["ref"], True)
            cells = []
            for k in models:
                texts = {e["chunk_id"]: e for e in evs if e["type"] == "text" and e["model"] == k}
                hyp = "".join(texts[c["id"]]["text"] for c in chunks if c["id"] in texts)
                err = asr_engine.edit_distance(rn, asr_engine.normalize_text(hyp, True))
                tot[k][0] += err
                tot[k][1] += len(rn)
                lat = []
                if ds["ends"]:
                    for g_end in ds["ends"]:
                        # 收尾这句话的 chunk：段尾在真实句尾附近
                        cand = [c for c in chunks if -0.3 <= c["end"] - g_end <= 1.0 and c["id"] in texts]
                        if cand:
                            c = min(cand, key=lambda c: abs(c["end"] - g_end))
                            judged = c["end"] + c["endpoint_wait_ms"] / 1000   # VAD 判定说完的音频时刻
                            lat.append((judged - g_end) * 1000 + texts[c["id"]]["decode_ms"])
                tot[k][2] += lat
                lat_s = (f"{np.mean(lat):.0f} / {np.percentile(lat, 90):.0f} ms" + (f"（{len(lat)} 句）" if args.real else "")
                         if lat else "—")
                cells.append(f"{err / max(len(rn), 1) * 100:.2f}% | {lat_s}")
            lines.append(f"| {ds['name']} | {len(chunks)} | {np.mean(durs):.1f}s / {max(durs):.1f}s | {split} | {merged} | "
                         + (f"{mixed} | " if args.real else "") + " | ".join(cells) + " |")
            print(f"[{silence}s] {ds['name']}: {len(chunks)} chunks", flush=True)
        lines += ["", "合计：" + "；".join(
            f"{names[k]} CER **{tot[k][0] / max(tot[k][1], 1) * 100:.2f}%**、真实延迟平均 **{np.mean(tot[k][2]):.0f} ms**"
            if tot[k][2] else f"{names[k]} CER **{tot[k][0] / max(tot[k][1], 1) * 100:.2f}%**" for k in models), ""]

    out = ROOT / "benchmark" / "results" / ("live_real.md" if args.real else "live.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
