"""声纹注册：录入或传入 4–6 句主人音频（每句 5–8 秒），提取声纹向量存 data/owner_profile.json。

向量是生物信息：data/owner_profile.json 与注册音频一律不提交（已在 .gitignore）。

用法:
    python tools/enroll_owner.py --mic                     # 麦克风录 4–6 句，每句 5–8 秒
    python tools/enroll_owner.py --files a.wav b.wav ...   # 传入已录好的音频（任意格式，需 ffmpeg）
质量检查：句间相似度均值 <0.5 警告重录（同人不同句相似度应明显偏高）。
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import subprocess
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import asr_engine  # noqa: E402
from speaker_owner import DEFAULT_PROFILE, OwnerProfile  # noqa: E402

SR = 16000
MIN_SENT = 5.0    # 每句最短 5 秒
MAX_SENT = 8.0    # 每句最长 8 秒
N_SENTS = 5       # 默认录 5 句（4–6 句都行）
WARN_SIM = 0.5    # 句间相似度均值低于这个值警告重录


def load_audio(path: Path) -> "np.ndarray":
    """读任意格式音频为 16kHz 单声道 float32；非 PCM wav 用 ffmpeg 转码。"""
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


def speech_only(x):
    """VAD 去静音，只留人声；VAD 不可用或无段时原样返回。"""
    segs = asr_engine.vad_segments(x, min_silence=0.25)
    if not segs:
        return x
    return np.concatenate([x[int(a * SR):int(b * SR)] for a, b in segs])


def embed(emb, x):
    """提取一条 L2 归一化声纹向量。"""
    s = emb.create_stream()
    s.accept_waveform(16000, x)
    s.input_finished()
    e = np.asarray(emb.compute(s), dtype=np.float32)
    return e / (np.linalg.norm(e) + 1e-9)


def record_mic(n_sents: int) -> list:
    """麦克风录 n_sents 句，每句 5–8 秒。"""
    import sounddevice as sd

    clips = []
    for i in range(n_sents):
        print(f"第 {i + 1}/{n_sents} 句：请说 5–8 秒（按回车开始）...")
        input()
        print("录音中...")
        audio = sd.rec(int(MAX_SENT * SR), samplerate=SR, channels=1, dtype="float32")
        sd.wait()
        x = audio.flatten()
        # 截到至少 MIN_SENT 秒
        clips.append(x[:int(MAX_SENT * SR)])
        print(f"录了 {len(x) / SR:.1f} 秒")
    return clips


def check_quality(vecs: list) -> tuple[float, bool]:
    """句间相似度均值；<WARN_SIM 返回 False（警告重录）。"""
    if len(vecs) < 2:
        return float("nan"), True
    sims = [float(np.dot(vecs[i], vecs[j]))
            for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
    return float(np.mean(sims)), float(np.mean(sims)) >= WARN_SIM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", action="store_true", help="麦克风录 4–6 句")
    ap.add_argument("--files", nargs="*", help="传入已录好的音频文件（任意格式，需 ffmpeg）")
    ap.add_argument("--n", type=int, default=N_SENTS, help="麦克风录几句（4–6）")
    ap.add_argument("--out", default=str(ROOT / "data" / "owner_profile.json"))
    args = ap.parse_args()

    if not args.mic and not args.files:
        ap.error("要 --mic 或 --files")
    if args.mic:
        try:
            import sounddevice  # noqa: F401
        except ImportError:
            print("麦克风录音需要 sounddevice（pip install sounddevice），或用 --files 传音频")
            return 1
        clips = record_mic(args.n)
    else:
        clips = [load_audio(Path(f)) for f in args.files]

    import sherpa_onnx as so
    from tools.eval_owner import MODELS

    model_path = MODELS["eres2net"]
    cfg = so.SpeakerEmbeddingExtractorConfig(model=str(model_path), num_threads=2)
    emb = so.SpeakerEmbeddingExtractor(cfg)

    vecs = []
    for i, x in enumerate(clips):
        dur = len(x) / SR
        if dur < MIN_SENT:
            print(f"第 {i + 1} 句只有 {dur:.1f} 秒（要求 ≥{MIN_SENT} 秒），跳过")
            continue
        v = embed(emb, speech_only(x))
        vecs.append(v)
        print(f"第 {i + 1} 句 {dur:.1f} 秒 → 向量 dim={len(v)}")
    if not vecs:
        print("没有可用句子（每句至少 5 秒）")
        return 1

    mean_sim, ok = check_quality(vecs)
    if not ok:
        print(f"警告：句间相似度均值 {mean_sim:.3f} < {WARN_SIM}，建议重录（环境噪声或说话方式差异太大）")
    else:
        print(f"句间相似度均值 {mean_sim:.3f}（≥{WARN_SIM}，质量合格）")

    profile = OwnerProfile(vecs)
    profile.save(args.out)
    print(f"已存 {args.out}（{len(profile)} 条向量，不提交）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
