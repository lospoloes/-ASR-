"""下载一段真人双人对话（MagicData-RAMC），截取一段，生成带时间戳的标准文本，给持续监听做真人实测。

MagicData-RAMC：两个人面对面自由聊天，手机录制，16kHz；人工转写到句，带起止时间和说话人。
这里用 HuggingFace 上的镜像 EaseZh/magicdata_ramc，只流式读到第一段（或指定的）对话为止，不下载整个包。
许可证 CC BY-NC-ND 4.0（署名、非商用、不可改编）：音频和文本只放在 test_audio/（不提交），仓库里只记评测数字。

用法:
    python tools/fetch_ramc.py                        # dev 集第一段对话，从第 10 秒起截约 180 秒
    python tools/fetch_ramc.py --start 600 --dur 180  # 同一段对话的其他位置（已下载过就不再联网）
    python tools/fetch_ramc.py --hf-mirror            # 国内网络走 hf-mirror.com
    python tools/fetch_ramc.py --count 3              # 包里前 3 段对话都下（每段约 30 分钟、60MB），各截一段
                                                      # （tools/eval_vad.py 用 full/ 里的整段对话评测 VAD）
输出（截取点都落在两句话之间的空隙里，不会切断一句话）:
    test_audio/ramc/<对话ID>_<起>-<止>.wav            16kHz 单声道
    test_audio/ramc/<对话ID>_<起>-<止>.tsv            起<TAB>止<TAB>说话人<TAB>文本（秒，相对截取起点）
"""
from __future__ import annotations

import argparse
import re
import sys
import tarfile
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "test_audio" / "ramc"
FULL = OUT / "full"
REPO = "datasets/EaseZh/magicdata_ramc/resolve/main"
LINE = re.compile(r"^\[([\d.]+),([\d.]+)\]\t(\S+)\t([^\t]*)\t(.*)$")
TAG = re.compile(r"\[[A-Z*+]*\]")  # [SONANT] [MUSIC] [LAUGHTER] [*] 等非语音标记


def fetch(split: str, conv: str | None, base: str, count: int = 1) -> list[str]:
    """流式读 <split>.tar.gz：收集 txt，存下目标对话（或包里前 count 段对话）的 wav，凑齐就停止下载。返回对话 ID。"""
    url = f"{base}/{REPO}/{split}.tar.gz"
    print(f"下载 {url}（读到目标对话就停）", flush=True)
    txts: dict[str, bytes] = {}
    got: list[str] = []
    with urllib.request.urlopen(url, timeout=60) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = Path(m.name).name
            if name.endswith(".txt"):
                txts[name[:-4]] = tf.extractfile(m).read()
            elif name.endswith(".wav") and len(got) < count and (conv is None or name[:-4] == conv):
                cid = name[:-4]
                FULL.mkdir(parents=True, exist_ok=True)
                dst = FULL / f"{cid}.wav"
                if not dst.exists():   # 已经有了就不再写（流还得读过去）
                    src, done = tf.extractfile(m), 0
                    with open(dst.with_suffix(".part"), "wb") as f:
                        while chunk := src.read(1 << 20):
                            f.write(chunk)
                            done += len(chunk)
                            print(f"\r  {cid}.wav {done / 1e6:.0f} / {m.size / 1e6:.0f} MB", end="", flush=True)
                    print(flush=True)
                    dst.with_suffix(".part").rename(dst)
                got.append(cid)
            if got and (conv is not None or len(got) >= count) and all(c in txts for c in got):
                break
    missing = [c for c in got if c not in txts]
    if not got or missing:
        raise SystemExit(f"{split}.tar.gz 里没找到对话 {conv or ''} 的 wav/txt {missing}")
    for c in got:
        (FULL / f"{c}.txt").write_bytes(txts[c])
    return got


def parse(txt: Path, nonspeech: bool = False) -> list[tuple[float, float, str, str]]:
    """标注 → [(起, 终, 说话人, 文本)]，按起点排序，文本里的 [LAUGHTER] [*] 等标记去掉。
    默认丢掉无人声段（G00000000）和只有标记的句子；nonspeech=True 时都留着（文本可能为空），
    给 VAD 评测算"有声音"的时间（笑声、听不清的话判成说话不算误检）。"""
    rows = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        m = LINE.match(ln.strip("﻿"))
        if not m:
            continue
        s, e, spk, _attr, text = m.groups()
        text = TAG.sub("", text).strip()
        if not nonspeech and (spk == "G00000000" or not text):  # 无人声段 / 只有非语音标记
            continue
        rows.append((float(s), float(e), spk, text))
    return sorted(rows)


def free(t: float, rows) -> bool:
    """t 时刻没有人在说话（不在任何一句的起止之间）。"""
    return all(not (s < t < e) for s, e, _, _ in rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "train"])
    ap.add_argument("--conv", default=None, help="对话 ID，如 CTS-CN-F2F-2019-11-15-1434；默认取包里第一段")
    ap.add_argument("--start", type=float, default=10.0, help="从第几秒附近开始截（跳过开头的采集口播）")
    ap.add_argument("--dur", type=float, default=180.0, help="截取时长（秒，按句间空隙取整）")
    ap.add_argument("--count", type=int, default=1, help="下包里前几段对话（给了 --conv 时忽略）")
    ap.add_argument("--hf-mirror", action="store_true", help="走 hf-mirror.com")
    args = ap.parse_args()

    base = "https://hf-mirror.com" if args.hf_mirror else "https://huggingface.co"
    have = sorted(FULL.glob("*.wav")) if FULL.exists() else []
    if args.conv:
        convs = [args.conv] if (FULL / f"{args.conv}.wav").exists() else fetch(args.split, args.conv, base)
    elif len(have) >= args.count:
        convs = [h.stem for h in have[:args.count]]
    else:
        convs = fetch(args.split, None, base, args.count)
    for conv in convs:
        cut(conv, args.start, args.dur)
    return 0


def cut(conv: str, start: float, dur: float) -> None:
    """从整段对话里截 start 起约 dur 秒（起止都落在句间空隙），连同标准文本存到 test_audio/ramc/。"""
    wav_path, rows = FULL / f"{conv}.wav", parse(FULL / f"{conv}.txt")

    # 起点：start 之后第一句话开头前的空隙；终点：不超过 start+dur 的最后一个句间空隙
    first = next(r for r in rows if r[0] >= start and free(r[0] - 0.05, rows))
    prev_end = max((e for s, e, _, _ in rows if e <= first[0]), default=0.0)
    t0 = max(prev_end, first[0] - 0.5)
    ends = [e for s, e, _, _ in rows if t0 < e <= t0 + dur and free(e + 0.05, rows)]
    if not ends:
        raise SystemExit("这段时间里找不到句间空隙，换个 --start")
    last_end = max(ends)
    next_start = min((s for s, e, _, _ in rows if s >= last_end), default=last_end + 0.5)
    t1 = min(next_start, last_end + 0.5)

    with wave.open(str(wav_path), "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        if (sr, sw) != (16000, 2):
            raise SystemExit(f"期望 16kHz/16bit，实际 {sr}Hz/{sw * 8}bit")
        w.setpos(int(t0 * sr))
        pcm = w.readframes(int((t1 - t0) * sr))
    if ch != 1:  # 多声道 → 混成单声道（模拟机器人一个麦克风听两个人）
        import numpy as np
        x = np.frombuffer(pcm, "<i2").reshape(-1, ch).astype(np.int32).mean(axis=1)
        pcm = np.clip(x, -32768, 32767).astype("<i2").tobytes()

    stem = OUT / f"{conv}_{t0:.0f}-{t1:.0f}"
    with wave.open(str(stem.with_suffix(".wav")), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)
    seg = [(s - t0, e - t0, spk, text) for s, e, spk, text in rows if s >= t0 - 0.01 and e <= t1 + 0.01]
    spk_ids = {spk: "AB"[i] if i < 2 else str(i) for i, spk in enumerate(dict.fromkeys(r[2] for r in seg))}
    stem.with_suffix(".tsv").write_text(
        "".join(f"{s:.3f}\t{e:.3f}\t{spk_ids[spk]}\t{text}\n" for s, e, spk, text in seg), encoding="utf-8")
    turns = sum(1 for a, b in zip(seg, seg[1:]) if a[2] != b[2])
    print(f"已截取 {stem.with_suffix('.wav').relative_to(ROOT)}：{t1 - t0:.1f} 秒，{len(seg)} 句，"
          f"说话人 {len(spk_ids)} 个，换人 {turns} 次（原始 {ch} 声道）")


if __name__ == "__main__":
    sys.exit(main())
