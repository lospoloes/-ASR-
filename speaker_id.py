"""声纹注册 + 说话人识别实验（例如：先注册 Elon Musk、黄仁勋的声音，再让系统认人）。

目录约定（test_audio/ 已在 .gitignore 里，名人音频不要提交）:
    test_audio/speakers/
        enroll/<人名>/*.wav|mp3|m4a   注册音频：每人约 10s，可放多条（会取平均）
        test/<人名>/*.wav|mp3|m4a     测试音频：必须和注册音频来自【不同的视频】，否则测不出真实效果
        test/unknown/*.wav            没注册过的人（测“陌生人会不会被认错”），目录名不在 enroll 里就算陌生人
        sources.tsv                   来源记录：文件<TAB>链接<TAB>起止时间（方便复现）

非 wav 格式需要系统装了 ffmpeg（Mac: brew install ffmpeg），脚本会自动转成 16kHz 单声道。

实验内容:
    1. 整条识别：每条测试音频 → 最像谁、相似度多少；陌生人应被拒绝
    2. 时长鲁棒性：测试音频切成 1/2/3/5s 小段，看多短还能认准
    3. 阈值扫描：认错率（FAR）/ 拒认率（FRR）随阈值变化，给出等错误率 EER 和推荐阈值
    4. 拼接场景（mentor 的设想）：各人约 10s 依次拼接（间隔 0.5s，混入一个陌生人），
       先用分角色模型找“换人”的边界再逐段认人，按时长统计标对的比例
    5. 模型对比：中文 3D-Speaker（默认）vs 英文 NeMo TitaNet（若已下载）

用法:
    python download_models.py                                   # 默认已包含中文声纹模型 + VAD
    python download_models.py --models speaker-embedding-en --no-diarization   # 可选：英文声纹模型
    python speaker_id.py                                        # 读 test_audio/speakers/
    python speaker_id.py --dir 其他目录 --threshold 0.5
结果写到 benchmark/results/speaker_id.md
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import asr_engine
import registry

ROOT = Path(__file__).resolve().parent
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4")
SR = 16000


# ---------------------------------------------------------------- 音频读取
def load_audio(path: Path) -> np.ndarray:
    """读取任意格式音频为 16kHz 单声道 float32。非 wav 用 ffmpeg 转码。"""
    if path.suffix.lower() == ".wav":
        try:
            x, sr = asr_engine.read_wav(path.read_bytes())
            return asr_engine.resample(x, sr, SR)
        except Exception:  # noqa: BLE001 —— 非 PCM wav，交给 ffmpeg
            pass
    if not shutil.which("ffmpeg"):
        raise RuntimeError(f"{path.name}: 需要 ffmpeg 转码（Mac: brew install ffmpeg；Windows: winget install ffmpeg）")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR),
                        "-sample_fmt", "s16", str(out)], check=True)
        x, sr = asr_engine.read_wav(out.read_bytes())
    return x


def speech_only(x: np.ndarray) -> np.ndarray:
    """只保留 VAD 判定的人声部分（去掉静音、音乐间隙），没有 VAD 时原样返回。"""
    segs, _ = asr_engine.compute_vad(x)
    if not segs:
        return x
    return np.concatenate([x[int(a * SR):int(b * SR)] for a, b in segs])


# ---------------------------------------------------------------- 声纹
class Embedder:
    def __init__(self, model_path: Path, num_threads: int = 2):
        import sherpa_onnx as so

        cfg = so.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=num_threads)
        if not cfg.validate():
            raise RuntimeError(f"声纹模型配置无效: {model_path}")
        self.ex = so.SpeakerEmbeddingExtractor(cfg)
        self.dim = self.ex.dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        s = self.ex.create_stream()
        s.accept_waveform(SR, x)
        s.input_finished()
        e = np.asarray(self.ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


def cosine(a, b) -> float:
    return float(np.dot(a, b))


def identify(emb, enrolled: dict, threshold: float) -> tuple[str, float, dict]:
    scores = {n: cosine(emb, e) for n, e in enrolled.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= threshold else "陌生人"), scores[best], scores


def chunks(x: np.ndarray, sec: float, max_n: int = 20) -> list[np.ndarray]:
    n = int(sec * SR)
    out = [x[i:i + n] for i in range(0, len(x) - n + 1, n)]
    return out[:max_n]


def eer(target: list[float], impostor: list[float]) -> tuple[float, float]:
    """等错误率：扫描阈值，找认错率（FAR）≈ 拒认率（FRR）的点。返回 (EER, 阈值)。"""
    if not target or not impostor:
        return float("nan"), float("nan")
    tg, im = np.asarray(target), np.asarray(impostor)
    best_gap, best = float("inf"), (float("nan"), float("nan"))
    for t in np.linspace(-0.2, 1.0, 241):
        frr, far = float(np.mean(tg < t)), float(np.mean(im >= t))
        if abs(frr - far) < best_gap:
            best_gap, best = abs(frr - far), ((frr + far) / 2, float(t))
    return best


# ---------------------------------------------------------------- 实验
def run(model_name: str, emb: Embedder, enroll: dict, tests: list, threshold: float, lines: list):
    # 注册：每人所有注册音频的人声部分各算一个向量，取平均
    enrolled = {}
    for name, paths in enroll.items():
        vs = [emb(speech_only(load_audio(p))) for p in paths]
        v = np.mean(vs, axis=0)
        enrolled[name] = v / np.linalg.norm(v)
    names = sorted(enrolled)
    lines += [f"## 模型：{model_name}", "", f"已注册：{'、'.join(names)}（阈值 {threshold}）", ""]

    # 1. 整条识别
    lines += ["### 1. 整条识别", "", "| 测试音频 | 真实身份 | 识别结果 | 最高相似度 | " + " | ".join(names) + " |",
              "|---|---|---|---|" + "---|" * len(names)]
    correct = 0
    for truth, p, x in tests:
        pred, best, scores = identify(emb(x), enrolled, threshold)
        want = truth if truth in enrolled else "陌生人"
        ok = pred == want
        correct += ok
        lines.append(f"| {p.name} | {want} | {'✅' if ok else '❌'} {pred} | {best:.3f} | "
                     + " | ".join(f"{scores[n]:.3f}" for n in names) + " |")
    lines += ["", f"整条准确率：**{correct}/{len(tests)}**", ""]

    # 2. 时长鲁棒性 + 3. 阈值扫描（用切片的分数）
    lines += ["### 2. 时长鲁棒性（测试音频切成小段逐段识别）", "", "| 片段时长 | 片段数 | 准确率 | 陌生人误认率 |", "|---|---|---|---|"]
    target, impostor = [], []
    for sec in (1, 2, 3, 5):
        n = ok = unk_n = unk_wrong = 0
        for truth, _, x in tests:
            for c in chunks(x, sec):
                pred, _, scores = identify(emb(c), enrolled, threshold)
                want = truth if truth in enrolled else "陌生人"
                n += 1
                ok += pred == want
                if want == "陌生人":
                    unk_n += 1
                    unk_wrong += pred != "陌生人"
                if sec == 3:  # 用 3s 片段做阈值扫描
                    for nm, sc in scores.items():
                        (target if nm == truth else impostor).append(sc)
        if n:
            lines.append(f"| {sec}s | {n} | {ok / n * 100:.0f}% | "
                         f"{(f'{unk_wrong / unk_n * 100:.0f}%（{unk_n} 段）') if unk_n else '—（没有陌生人音频）'} |")
    e, t = eer(target, impostor)
    lines += ["", "### 3. 阈值", "",
              f"3s 片段：同一人相似度均值 {np.mean(target):.3f}，不同人均值 {np.mean(impostor):.3f}；"
              f"**等错误率 EER {e * 100:.1f}%，对应阈值 {t:.2f}**（认错率=拒认率的点，可作为推荐阈值起点）"
              if target and impostor else "样本不足，无法计算 EER", ""]
    if target and impostor and abs(t - threshold) > 1e-6:
        ok2 = sum(identify(emb(x), enrolled, t)[0] == (tr if tr in enrolled else "陌生人") for tr, _, x in tests)
        lines += [f"按推荐阈值 {t:.2f} 重算整条准确率：**{ok2}/{len(tests)}**（当前阈值 {threshold} 时为 {correct}/{len(tests)}）", ""]

    # 4. 拼接场景（mentor 的设想）：各人 10s 依次拼接，中间只隔 0.5s，再混入一个陌生人
    #    VAD 切段会把相邻两人粘在一起（0.5s 停顿不够），所以要先找“换人”的边界：
    #    A) 分角色模型（pyannote 分段 + 聚类）切出说话人段 → 每段认人（实际产品的做法）
    #    B) 对照：1.5s 滑动窗口逐窗认人
    known = [(tr, x) for tr, _, x in tests if tr in enrolled]
    strangers = [x for tr, _, x in tests if tr not in enrolled]
    if len({tr for tr, _ in known}) >= 2:
        pieces, timeline, t0 = [], [], 0.0
        by = {}
        for tr, x in known:
            by.setdefault(tr, []).append(x)
        order = sorted(by)
        for r in range(2):
            plan = [(tr, by[tr][r % len(by[tr])]) for tr in order]
            if strangers:
                plan.append(("陌生人", strangers[r % len(strangers)]))
            for tr, src in plan:
                seg = src[:10 * SR] if r == 0 or len(src) < 20 * SR else src[10 * SR:20 * SR]
                pieces += [seg, np.zeros(SR // 2, np.float32)]
                timeline.append((t0, t0 + len(seg) / SR, tr))
                t0 += len(seg) / SR + 0.5
        mix = np.concatenate(pieces)

        def truth_at(t):
            for a, b, tr in timeline:
                if a <= t < b:
                    return tr
            return None

        def time_acc(labels):  # labels: [(a, b, pred)]，按 0.1s 帧统计说话时间里标对的比例
            ok = tot = 0
            for k in range(int(t0 * 10)):
                t = k / 10 + 0.05
                tr = truth_at(t)
                if tr is None:
                    continue
                tot += 1
                pred = next((p for a, b, p in labels if a <= t < b), "未检出")
                ok += pred == tr
            return ok / max(tot, 1)

        lines += ["### 4. 拼接场景", "",
                  "顺序：" + " → ".join(tr for _, _, tr in timeline) + f"（每人约 10s，间隔 0.5s，共 {t0:.0f}s）", ""]
        seg_f = registry.resolve_files(registry.get_spec("pyannote-seg"))
        emb_f = registry.resolve_files(registry.get_spec("speaker-embedding"))
        if seg_f and emb_f:
            dz = asr_engine.Diarizer(seg_f["model"], emb_f["model"])
            dsegs, _ = dz.run(mix)
            labels = []
            rows = []
            for d in dsegs:
                a, b = d["start"], d["end"]
                if b - a < 0.5:
                    continue
                pred, best, _ = identify(emb(mix[int(a * SR):int(b * SR)]), enrolled, threshold)
                labels.append((a, b, pred))
                mid_truth = truth_at((a + b) / 2) or "（静音）"
                rows.append(f"| {a:.1f}–{b:.1f}s | {mid_truth} | {'✅' if pred == mid_truth else '❌'} {pred} | {best:.3f} |")
            lines += [f"**A. 分角色切段 + 认人：按时长计准确率 {time_acc(labels) * 100:.1f}%**（{len(labels)} 段）", "",
                      "| 分角色段 | 真实身份（段中点） | 识别结果 | 相似度 |", "|---|---|---|---|"] + rows + [""]
        win, hop = 1.5, 0.5
        labels = []
        k = 0.0
        while k + win <= t0:
            x = mix[int(k * SR):int((k + win) * SR)]
            if np.abs(x).max() > 1e-3:
                pred, _, _ = identify(emb(x), enrolled, threshold)
                labels.append((k + (win - hop) / 2, k + (win + hop) / 2, pred))
            k += hop
        lines += [f"**B. 对照：{win}s 滑动窗口逐窗认人：按时长计准确率 {time_acc(labels) * 100:.1f}%**", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "test_audio" / "speakers"))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "benchmark" / "results" / "speaker_id.md"))
    args = ap.parse_args()

    d = Path(args.dir)
    enroll = {p.name: sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXT)
              for p in sorted((d / "enroll").glob("*")) if p.is_dir()}
    enroll = {k: v for k, v in enroll.items() if v}
    test_files = [(p.name, f) for p in sorted((d / "test").glob("*")) if p.is_dir()
                  for f in sorted(p.iterdir()) if f.suffix.lower() in AUDIO_EXT]
    if len(enroll) < 2 or not test_files:
        print(f"目录不完整：需要 {d}/enroll/<人名>/ 至少 2 人，以及 {d}/test/<人名>/ 测试音频。见文件开头说明。")
        return 1
    tests = [(truth, f, speech_only(load_audio(f))) for truth, f in test_files]

    models = [(registry.get_spec(k), registry.resolve_files(registry.get_spec(k)))
              for k in ("speaker-embedding", "speaker-embedding-en")]
    models = [(s, f) for s, f in models if f]
    if not models:
        print("缺少声纹模型：python download_models.py")
        return 1
    lines = ["# 声纹注册 + 说话人识别实验", "",
             f"- 数据：`{d}`，注册 {len(enroll)} 人（{sum(len(v) for v in enroll.values())} 条），测试 {len(tests)} 条",
             "- 所有音频先用 VAD 去掉静音/音乐，只保留人声再提取声纹", ""]
    for spec, files in models:
        size = registry.disk_size_mb(files)
        print(f"== {spec['name']}（{size} MB）", flush=True)
        run(f"{spec['name']}（{size} MB）", Embedder(files["model"]), enroll, tests, args.threshold, lines)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n结果: {out}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
