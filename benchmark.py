"""批量 benchmark：所有已下载 ASR 模型 × 一个目录里的 wav，输出 CER / RTF / 数字格式汇总。

用法:
    python benchmark.py                                # 默认用 benchmark/testset（make_test_audio.py 生成）
    python benchmark.py --dir D:/my_wavs               # 自己的真人录音，目录里放 refs.tsv（文件名<TAB>参考文本）
    python benchmark.py --dir test_audio --refs benchmark/long/refs.tsv              # 长音频，VAD 切段
    python benchmark.py --dir test_audio --refs benchmark/long/refs.tsv --no-vad     # 对照：30s 固定窗口
    python benchmark.py --threads 1 --models sensevoice,paraformer-zh-small
结果写到 benchmark/results/<时间>.md 和 .csv（--name 可指定固定文件名）

注意：VAD 对每段音频只算一次，所有模型共用；各模型的 RTF 只含 ASR 本身耗时，VAD 耗时单独列出。
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import asr_engine
import registry

ROOT = Path(__file__).resolve().parent


def pct(v) -> str:
    return "—" if v is None else f"{v * 100:.2f}%"


def read_tsv(f: Path) -> dict[str, str]:
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if "\t" in line and not line.startswith("#"):
            k, v = line.split("\t", 1)
            out[k.strip()] = v.strip()
    return out


def read_refs(d: Path, refs_path: str = "") -> dict[str, str]:
    """--refs 指定文件时直接读它；否则读目录下的 refs.tsv。"""
    return read_tsv(Path(refs_path) if refs_path else d / "refs.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "benchmark" / "testset"))
    ap.add_argument("--refs", default="", help="refs.tsv 路径（默认读 --dir 下的 refs.tsv）")
    ap.add_argument("--models", default="")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-vad", action="store_true",
                    help="关闭 VAD，用 30s 固定窗口切段（对照用）；RTF 只含 ASR 耗时")
    ap.add_argument("--no-num-norm", action="store_true", help="CER 不做数字归一化（阿拉伯数字与汉字数字算不同）")
    ap.add_argument("--name", default="", help="结果文件的固定前缀（默认用时间戳）")
    args = ap.parse_args()

    d = Path(args.dir)
    wavs = sorted(p for p in d.glob("*.wav") if not p.name.startswith("dialog_"))
    if not wavs:
        print(f"{d} 下没有 wav。先运行 python make_test_audio.py，或用 --dir 指定目录")
        return 1
    refs = read_refs(d, args.refs)
    if args.refs and not refs:
        print(f"警告: --refs 指定的文件读不到参考文本: {args.refs}")
    sel = [k.strip() for k in args.models.split(",") if k.strip()]
    downloaded = asr_engine.find_downloaded()
    models = []
    for spec in registry.ASR_MODELS:
        if spec["key"] in downloaded and (not sel or spec["key"] in sel):
            m = asr_engine.AsrModel(spec, downloaded[spec["key"]], args.threads)
            print(f"加载 {spec['name']} ...", flush=True)
            m.load()
            models.append(m)
    if not models:
        print("没有已下载的模型，先运行 python download_models.py")
        return 1

    # 读音频；VAD 每段音频只算一次，所有模型共用
    audio = []
    vad_total_ms, vad_segs = 0.0, 0
    vad_ok = False
    for w in wavs:
        x, sr = asr_engine.read_wav(w.read_bytes())
        x = asr_engine.resample(x, sr)
        segs: list = []
        if not args.no_vad:
            segs, vad_ms = asr_engine.compute_vad(x)
            vad_total_ms += vad_ms
            vad_segs += len(segs)
            vad_ok = vad_ok or bool(segs)
        audio.append((w.name, x, refs.get(w.name, ""), segs))
    total_dur = sum(len(x) for _, x, _, _ in audio) / 16000

    rows, summary = [], []
    for m in models:
        # 预热一次，避免首次推理的初始化开销计入
        m.transcribe(audio[0][1][:16000], segs=[])
        errs = chars = 0
        elapsed = 0.0
        arabic = False
        first = []
        for name, x, ref, segs in audio:
            r = m.transcribe(x, segs=segs)
            elapsed += r["elapsed_ms"] / 1000
            arabic |= r["has_arabic_digits"]
            if r.get("first_partial_at") is not None:
                first.append(r["first_partial_at"])
            c = None
            if ref:
                rn = asr_engine.normalize_text(ref, not args.no_num_norm)
                hn = asr_engine.normalize_text(r["text"], not args.no_num_norm)
                e = asr_engine.edit_distance(rn, hn)
                errs, chars, c = errs + e, chars + len(rn), e / max(len(rn), 1)
            rows.append({"model": m.key, "file": name, "ref": ref, "hyp": r["text"],
                         "cer": "" if c is None else f"{c:.4f}", "rtf": r["rtf"],
                         "elapsed_ms": r["elapsed_ms"]})
        summary.append({
            "model": m.spec["name"], "key": m.key, "size": registry.disk_size_mb(m.files),
            "streaming": m.spec["streaming"], "cer": errs / chars if chars else None,
            "rtf": elapsed / total_dur, "arabic": arabic,
            "first": sum(first) / len(first) if first else None, "load": m.load_seconds,
        })
        s = summary[-1]
        print(f"  {m.key:30} CER {pct(s['cer']):>7}  RTF {s['rtf']:.4f}", flush=True)

    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = args.name or time.strftime("%Y%m%d-%H%M%S")
    with open(out_dir / f"{stamp}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    mode = "30s 固定窗口（--no-vad）" if args.no_vad else "VAD 切段"
    env = f"{platform.system()} {platform.machine()} · {platform.processor() or 'CPU'} · Python {platform.python_version()} · 线程数 {args.threads}"
    md = [f"# Benchmark {stamp}", "",
          f"- 数据: `{d.name}`，{len(audio)} 条，共 {total_dur:.1f}s，{'有' if refs else '无'}参考文本",
          f"- 环境: {env}",
          f"- 切段方式: **{mode}**",
          f"- VAD: {'关闭' if args.no_vad else (f'{vad_total_ms:.0f} ms（{vad_segs} 段，对每段音频只算一次，所有模型共用）' if vad_ok else '已开启但未检出人声段（缺 silero-vad 模型），退回固定窗口')}",
          f"- CER: 字级编辑距离，去标点空格，{'不做' if args.no_num_norm else '做'}数字归一化",
          f"- RTF / 耗时只含 ASR 本身，不含 VAD", "",
          "| 模型 | 体积 MB | 流式 | CER | RTF | 总耗时(s) | 平均首字出现(s) | 输出阿拉伯数字 | 加载(s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(summary, key=lambda s: (s["cer"] is None, s["cer"] or 0)):
        md.append(f"| {s['model']} | {s['size']} | {'是' if s['streaming'] else '否'} | "
                  f"{pct(s['cer'])} | {s['rtf']:.4f} | {s['rtf'] * total_dur:.1f} | "
                  f"{'—' if s['first'] is None else round(s['first'], 2)} | {'是' if s['arabic'] else '否'} | {s['load']:.1f} |")
    md += ["", "## 逐条结果", "", "| 文件 | 参考 | " + " | ".join(m.key for m in models) + " |",
           "|---|---|" + "---|" * len(models)]
    for name, _, ref, _ in audio:
        cells = []
        for m in models:
            r = next(x for x in rows if x["model"] == m.key and x["file"] == name)
            cells.append(f"{r['hyp']}" + (f" ({float(r['cer']) * 100:.0f}%)" if r["cer"] else ""))
        md.append(f"| {name} | {ref} | " + " | ".join(c.replace("|", "/") for c in cells) + " |")
    (out_dir / f"{stamp}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n结果: benchmark/results/{stamp}.md 和 .csv")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
