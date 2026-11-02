"""用 sherpa-onnx 的本地 TTS 生成带标准答案的测试音频（对应会议待办“生成系列音频测试文件”）。

输出:
    benchmark/testset/<id>.wav    16kHz 单声道
    benchmark/testset/refs.tsv    文件名<TAB>参考文本
    benchmark/testset/dialog_2spk.wav + dialog_2spk.tsv   两个音色交替说话，用来测分角色

注意：TTS 合成语音比真人录音“干净”，CER 会偏乐观，只适合做横向对比，最终结论要用真人录音复核。
用法:
    python download_models.py --tts      # 先下载 TTS 模型（aishell3 29MB + melo 163MB）
    python make_test_audio.py
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

import asr_engine
import registry

ROOT = Path(__file__).resolve().parent
SENTENCES = ROOT / "benchmark" / "sentences.tsv"
OUT = ROOT / "benchmark" / "testset"


def load_tts(key: str):
    import sherpa_onnx as so

    spec = registry.get_spec(key)
    files = registry.resolve_files(spec)
    if not files:
        return None
    d = files["model"].parent
    fsts = ",".join(str(d / f) for f in spec.get("rule_fsts", []) if (d / f).exists())
    vits = so.OfflineTtsVitsModelConfig(
        model=str(files["model"]), lexicon=str(files["lexicon"]), tokens=str(files["tokens"]))
    cfg = so.OfflineTtsConfig(model=so.OfflineTtsModelConfig(vits=vits, num_threads=2), rule_fsts=fsts)
    if not cfg.validate():
        raise RuntimeError(f"TTS 配置校验失败: {key}")
    return so.OfflineTts(cfg)


def write_wav(path: Path, x: np.ndarray, sr: int = 16000):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--voice", choices=["melo", "mixed"], default="melo",
                    help="melo=全部用 MeloTTS（默认，音质最好）；mixed=按 sentences.tsv 指定音色（aishell3 多音色，音质较差，测鲁棒性）")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    tts = {"aishell3": load_tts("tts-aishell3"), "melo": load_tts("tts-melo-zh-en")}
    if not any(tts.values()):
        print("没有 TTS 模型，请先运行: python download_models.py --tts")
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = []
    for line in SENTENCES.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sid_, kind, sid, text = line.split("\t")
        if args.voice == "melo":
            kind, sid = "melo", "0"
        engine = tts.get(kind)
        if engine is None:
            print(f"跳过 {sid_}: 未下载 {kind} TTS")
            continue
        a = engine.generate(text, sid=int(sid), speed=args.speed)
        x = asr_engine.resample(np.asarray(a.samples, dtype=np.float32), a.sample_rate, 16000)
        x = np.concatenate([np.zeros(4000, np.float32), x, np.zeros(4000, np.float32)])  # 前后各 0.25s 静音
        write_wav(out / f"{sid_}.wav", x)
        refs.append(f"{sid_}.wav\t{text}")
        print(f"{sid_}.wav  {len(x) / 16000:.1f}s  {text}")
    (out / "refs.tsv").write_text("\n".join(refs) + "\n", encoding="utf-8")

    # 两人对话（分角色测试）
    if tts["aishell3"]:
        turns = [(10, "你今天下班早吗？"), (99, "应该六点左右就能到家。"),
                 (10, "那我们晚上一起做饭吧。"), (99, "好啊，我顺路去超市买点菜。")]
        parts, lines, t = [], [], 0.0
        for sid, text in turns:
            a = tts["aishell3"].generate(text, sid=sid, speed=args.speed)
            x = asr_engine.resample(np.asarray(a.samples, dtype=np.float32), a.sample_rate, 16000)
            parts += [x, np.zeros(9600, np.float32)]  # 0.6s 间隔
            lines.append(f"{t:.2f}\t{t + len(x) / 16000:.2f}\t说话人{'A' if sid == 10 else 'B'}\t{text}")
            t += len(x) / 16000 + 0.6
        write_wav(out / "dialog_2spk.wav", np.concatenate(parts))
        (out / "dialog_2spk.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"dialog_2spk.wav  {t:.1f}s（两个音色交替，测分角色）")
    print(f"\n完成：{out}。下一步: python benchmark.py")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
