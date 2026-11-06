"""主人识别评测：在声纹测试集上算 FAR / FRR / EER / 时长准确率，产出 benchmark/results/owner_eval.md。

数据（许可证不允许提交，音频不进仓库）:
    <data>/enroll/<人名>/*.wav   注册音频：每人 1–2 条，各 5–10s
    <data>/test/<人名>/*.wav     测试音频：必须来自和注册不同的视频
    <data>/test/<陌生人>/        目录名不在 enroll 里 → 陌生人（负例）

划分：dev 集（整条，用于调阈值）与 validation 集（1/2/3/5s 切片 + EER + FAR/FRR 曲线）分开，
报告 owner_eval.md 里数字分开列。反向验证：--shuffle 把注册向量打乱后 FAR 应升到 50% 上下。

用法:
    python tools/eval_owner.py --data D:/projects/lospoloes-proj/test_audio/speakers
    python tools/eval_owner.py --data ... --model titanet          # 对比英文 TitaNet
    python tools/eval_owner.py --data ... --shuffle                # 反向验证：打乱注册向量
    python tools/eval_owner.py --data ... --out benchmark/results/owner_eval.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asr_engine  # noqa: E402

SR = 16000
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4")
MODELS = {
    "eres2net": ROOT / "models" / "speaker-embedding" / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
    "titanet": ROOT / "models" / "speaker-embedding-en" / "nemo_en_titanet_small.onnx",
}
# dev/validation 划分：test/ 目录按文件名哈希固定划分，保证 --model/--shuffle 各次跑同一划分
DEV_FRACTION = 0.5


def load_audio(path: Path) -> "np.ndarray":
    """读取任意格式音频为 16kHz 单声道 float32；非 PCM wav 用 ffmpeg 转码。"""
    if path.suffix.lower() == ".wav":
        try:
            x, sr = asr_engine.read_wav(path.read_bytes())
            return asr_engine.resample(x, sr, SR)
        except Exception:  # noqa: BLE001 — 非 PCM wav 交给 ffmpeg
            pass
    if not __import__("shutil").which("ffmpeg"):
        raise RuntimeError(f"{path.name}: 需要 ffmpeg 转码（Windows: winget install ffmpeg）")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-sample_fmt", "s16", str(out)], check=True)
        x, sr = asr_engine.read_wav(out.read_bytes())
    return asr_engine.resample(x, sr, SR)


def speech_only(x, min_silence: float = 0.25):
    """VAD 去静音，只留人声；VAD 不可用或无段时原样返回。"""
    segs = asr_engine.vad_segments(x, min_silence=min_silence)
    if not segs:
        return x
    return np.concatenate([x[int(a * SR):int(b * SR)] for a, b in segs])


class Embedder:
    def __init__(self, model_path: Path, num_threads: int = 2):
        import sherpa_onnx as so

        cfg = so.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=num_threads)
        if not cfg.validate():
            raise RuntimeError(f"声纹模型配置无效: {model_path}")
        self.ex = so.SpeakerEmbeddingExtractor(cfg)
        self.dim = self.ex.dim

    def __call__(self, x) -> "np.ndarray":
        import numpy as np
        s = self.ex.create_stream()
        s.accept_waveform(SR, x)
        s.input_finished()
        e = np.asarray(self.ex.compute(s), dtype="float32")
        return e / (np.linalg.norm(e) + 1e-9)


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def chunks(x, sec: float, max_n: int = 20):
    n = int(sec * SR)
    return [x[i:i + n] for i in range(0, len(x) - n + 1, n)][:max_n]


def eer(target, impostor) -> tuple[float, float]:
    """等错误率：扫描阈值找 FAR≈FRR 的点，返回 (EER, 阈值)。"""
    if not target or not impostor:
        return float("nan"), float("nan")
    tg, im = np.asarray(target), np.asarray(impostor)
    best_gap, best = float("inf"), (float("nan"), float("nan"))
    for t in np.linspace(-0.2, 1.0, 241):
        frr, far = float(np.mean(tg < t)), float(np.mean(im >= t))
        if abs(frr - far) < best_gap:
            best_gap, best = abs(frr - far), ((frr + far) / 2, float(t))
    return best


def split_tests(tests: list[tuple[str, Path]]) -> tuple[list, list]:
    """按文件名哈希把测试集固定切成 dev / validation 两半（两层目录都参与哈希）。"""
    dev, val = [], []
    for truth, p in tests:
        h = hash((truth, p.name)) % 100
        (dev if h < DEV_FRACTION * 100 else val).append((truth, p))
    return dev, val


def enroll(emb, data_dir: Path) -> dict:
    """注册：每人所有注册音频的人声部分各算一个向量取平均，L2 归一化。"""
    enrolled = {}
    for d in sorted((data_dir / "enroll").iterdir()):
        if not d.is_dir():
            continue
        vs = [emb(speech_only(load_audio(p))) for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_EXT]
        if not vs:
            continue
        v = np.mean(vs, axis=0)
        enrolled[d.name] = v / np.linalg.norm(v)
    return enrolled


def collect_tests(data_dir: Path) -> list[tuple[str, Path]]:
    tests = []
    for d in sorted((data_dir / "test").iterdir()):
        if d.is_dir():
            tests += [(d.name, p) for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_EXT]
    return tests


def pred_of(scores: dict, threshold: float) -> tuple[str, str]:
    """阈值判定：最高相似度 ≥ 阈值 → 该注册名，否则陌生人。返回 (判定, 最高名)。"""
    best = max(scores, key=scores.get)
    return (best if scores[best] >= threshold else "陌生人"), best


def scan_thresholds(target, impostor, step: float = 0.05, lo: float = 0.30, hi: float = 0.60) -> list[dict]:
    """阈值扫描：FAR / FRR 曲线，0.30–0.60 步长 0.05。"""
    tg, im = np.asarray(target), np.asarray(impostor)
    rows = []
    for t in np.arange(lo, hi + 1e-9, step):
        rows.append({"thr": round(float(t), 2), "far": float(np.mean(im >= t)), "frr": float(np.mean(tg < t))})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="声纹数据目录（enroll/ + test/）")
    ap.add_argument("--model", choices=list(MODELS), default="eres2net")
    ap.add_argument("--threshold", type=float, default=0.41, help="整条识别用的阈值（默认 EER 起点 0.41）")
    ap.add_argument("--shuffle", action="store_true", help="反向验证：打乱注册向量（FAR 应升到 50%% 上下）")
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "results" / "owner_eval.md"))
    args = ap.parse_args()

    data = Path(args.data)
    if not (data / "enroll").is_dir() or not (data / "test").is_dir():
        print(f"没有数据：{data} 下要有 enroll/ 和 test/")
        return 1

    emb = Embedder(MODELS[args.model])
    enrolled = enroll(emb, data)
    if args.shuffle:  # 反向验证：把注册向量整体打乱（重排维度），相似度应掉到随机水平
        rng = np.random.default_rng(0)
        perm = rng.permutation(enrolled[next(iter(enrolled))].shape[0])
        enrolled = {k: v[perm] for k, v in enrolled.items()}
    names = sorted(enrolled)
    tests = collect_tests(data)
    if not enrolled or not tests:
        print(f"数据不足：enroll {len(enrolled)} 人，test {len(tests)} 条")
        return 1

    dev, val = split_tests(tests)
    lines = [f"# 主人识别评测（{args.model}）", "",
             f"- 注册 {len(enrolled)} 人：{'、'.join(names)}",
             f"- 测试 {len(tests)} 条：dev {len(dev)} 条（调阈值）/ validation {len(val)} 条（切片 + EER + FAR/FRR 曲线）",
             f"- 反向验证（--shuffle 打乱注册向量）：{'开' if args.shuffle else '关'}", ""]

    # dev 集：整条识别（调阈值用）
    lines += ["## dev 集（整条，调阈值用）", "", "| 文件 | 真实 | 最高相似度 | 判定（阈值 %.2f） |" % args.threshold,
              "|---|---|---|---|"]
    for truth, p in dev:
        emb_x = emb(speech_only(load_audio(p)))
        scores = {n: cosine(emb_x, e) for n, e in enrolled.items()}
        pred, best = pred_of(scores, args.threshold)
        want = truth if truth in enrolled else "陌生人"
        lines.append(f"| {p.name} | {truth} | {scores[best]:.3f} | {'✅' if pred == want else '❌'} {pred} |")
    lines.append("")

    # validation 集：整条 + 时长切片 + EER + 阈值曲线
    lines += ["## validation 集", "", "| 文件 | 时长 | 真实 | 最高相似度 |", "|---|---|---|---|"]
    target, impostor = [], []
    dur_rows: dict[int, dict] = {}
    for truth, p in val:
        x = speech_only(load_audio(p))
        emb_x = emb(x)
        scores = {n: cosine(emb_x, e) for n, e in enrolled.items()}
        best = max(scores, key=scores.get)
        lines.append(f"| {p.name} | 整条 | {truth} | {scores[best]:.3f} |")
        for sec in (1, 2, 3, 5):
            for c in chunks(x, sec):
                scores_c = {n: cosine(emb(c), e) for n, e in enrolled.items()}
                best_c = max(scores_c, key=scores_c.get)
                want_c = truth if truth in enrolled else "陌生人"
                d = dur_rows.setdefault(sec, {"n": 0, "ok": 0, "unk_n": 0, "unk_wrong": 0})
                d["n"] += 1
                d["ok"] += best_c == want_c
                if want_c == "陌生人":
                    d["unk_n"] += 1
                    d["unk_wrong"] += best_c != want_c
                if sec == 3:
                    for nm, sc in scores_c.items():
                        (target if nm == truth else impostor).append(sc)
    lines.append("")

    lines += ["### 整条识别（validation 集，阈值 %.2f）" % args.threshold, "",
              "| 文件 | 真实 | 最高相似度 | 判定 |", "|---|---|---|---|"]
    val_ok = 0
    for truth, p in val:
        x = speech_only(load_audio(p))
        scores = {n: cosine(emb(x), e) for n, e in enrolled.items()}
        pred, best = pred_of(scores, args.threshold)
        want = truth if truth in enrolled else "陌生人"
        val_ok += pred == want
        lines.append(f"| {p.name} | {truth} | {scores[best]:.3f} | {'✅' if pred == want else '❌'} {pred} |")
    lines += ["", f"整条准确率（阈值 {args.threshold:.2f}）：**{val_ok}/{len(val)}**", ""]

    lines += ["### 时长鲁棒性（validation 集，切片逐段识别）", "",
              "| 片段时长 | 片段数 | 准确率 | 陌生人误认率 |", "|---|---|---|---|"]
    for sec in (1, 2, 3, 5):
        d = dur_rows.get(sec, {"n": 0, "ok": 0, "unk_n": 0, "unk_wrong": 0})
        acc = f"{d['ok'] / d['n'] * 100:.0f}%" if d["n"] else "—"
        unk = f"{d['unk_wrong'] / d['unk_n'] * 100:.0f}%（{d['unk_n']} 段）" if d["unk_n"] else "—"
        lines.append(f"| {sec}s | {d['n']} | {acc} | {unk} |")
    lines.append("")

    e, t = eer(target, impostor)
    lines += ["### 阈值（validation 集 3s 片段）", ""]
    if target and impostor:
        lines.append(f"同一人相似度均值 {np.mean(target):.3f}，不同人均值 {np.mean(impostor):.3f}；"
                     f"**EER {e * 100:.1f}%，对应阈值 {t:.2f}**")
        lines += ["", "| 阈值 | FAR | FRR |", "|---|---|---|"]
        lines += [f"| {r['thr']:.2f} | {r['far'] * 100:.1f}% | {r['frr'] * 100:.1f}% |"
                  for r in scan_thresholds(target, impostor)]
    else:
        lines.append("样本不足，无法计算 EER / FAR-FRR 曲线")
    lines.append("")

    if args.shuffle:  # 反向验证：打乱后 FAR 应升到 50% 上下
        if target and impostor:
            lines.append(f"**反向验证（--shuffle）**：FAR {np.mean(impostor >= t) * 100:.1f}%（期望 ~50%）")
        else:
            lines.append("反向验证：样本不足，无法计算")
    lines.append("")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写入 {args.out}")
    print(json.dumps({"model": args.model, "n_enroll": len(enrolled), "n_test": len(tests),
                      "dev": len(dev), "val": len(val), "shuffle": args.shuffle}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
