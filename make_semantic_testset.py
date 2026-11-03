"""生成语义切段测试集：句内有“思考停顿”、句间停顿长短不一，句子边界已知。

每条测试句用 “|” 标出句内停顿位置，例如 “我想|明天带孩子去公园玩”：
各部分分别合成，中间插入 0.4~0.9s 静音（模拟“想一下”）；
句与句之间插入 0.35~1.6s 静音（部分句间停顿比句内停顿还短，纯声学 VAD 会切错）。

输出:
    benchmark/semantic/session{1,2,3}.wav
    benchmark/semantic/gt.json   每个文件的句子边界 [start, end] 和文本
用法:
    python download_models.py --models tts-melo-zh-en --no-diarization
    python make_semantic_testset.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

import asr_engine
from make_test_audio import load_tts, write_wav

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark" / "semantic"

SESSIONS = [
    ["我想|明天带孩子去公园玩", "帮我查一下明天的天气", "如果明天下雨的话|就提醒我带伞", "那个|上次你讲的故事|再讲一遍好不好",
     "今天好累啊", "因为孩子发烧了|所以今天不去上班了", "我今天去超市|然后买了一些水果和牛奶", "播放一首周杰伦的歌"],
    ["你知道为什么天空是蓝色的吗", "就是|我想问一下|明天几点出发", "晚安", "把客厅的灯|关掉",
     "我觉得|这个电影还挺好看的", "现在几点了", "等一下|先别说了", "给我讲个笑话吧"],
    ["嗯|让我想想|还是明天再说吧", "提醒我下午三点开会", "虽然有点贵|但是质量很好", "好的谢谢你",
     "先把音量调小一点|然后放点轻音乐", "明天早上七点叫我起床", "关于周末的安排|你有什么建议吗", "我回来了"],
]


def trim(x: np.ndarray, sr: int = 16000, thr: float = 0.02) -> np.ndarray:
    """去掉 TTS 输出首尾的静音，保证插入的停顿时长准确。"""
    frame = sr // 100
    e = np.array([np.abs(x[i:i + frame]).max() for i in range(0, len(x) - frame, frame)])
    idx = np.where(e > thr * e.max())[0]
    if len(idx) == 0:
        return x
    return x[idx[0] * frame:(idx[-1] + 1) * frame]


def main():
    tts = load_tts("tts-melo-zh-en")
    if tts is None:
        print("缺少 MeloTTS：python download_models.py --models tts-melo-zh-en --no-diarization")
        return 1
    rng = random.Random(20261103)
    OUT.mkdir(parents=True, exist_ok=True)
    gt = {}
    for si, sents in enumerate(SESSIONS, 1):
        parts_audio, t = [np.zeros(8000, np.float32)], 0.5
        items = []
        for k, s in enumerate(sents):
            start = t
            pieces = s.split("|")
            for j, piece in enumerate(pieces):
                a = tts.generate(piece, sid=0, speed=1.0)
                x = trim(asr_engine.resample(np.asarray(a.samples, np.float32), a.sample_rate))
                parts_audio.append(x)
                t += len(x) / 16000
                if j < len(pieces) - 1:
                    p = rng.uniform(0.4, 0.9)          # 句内思考停顿
                    parts_audio.append(np.zeros(int(p * 16000), np.float32))
                    t += p
            items.append({"start": round(start, 3), "end": round(t, 3), "text": s.replace("|", ""),
                          "n_pauses": len(pieces) - 1})
            if k < len(sents) - 1:
                p = rng.uniform(0.35, 1.6)             # 句间停顿（部分比句内停顿还短）
                parts_audio.append(np.zeros(int(p * 16000), np.float32))
                t += p
        parts_audio.append(np.zeros(8000, np.float32))
        name = f"session{si}.wav"
        write_wav(OUT / name, np.concatenate(parts_audio))
        gt[name] = items
        print(f"{name}: {t + 0.5:.1f}s, {len(items)} 句, 句内停顿 {sum(i['n_pauses'] for i in items)} 处")
    (OUT / "gt.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
