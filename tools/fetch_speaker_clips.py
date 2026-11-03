"""按清单下载并截取说话人音频，放进 test_audio/speakers/ 供 speaker_id.py 使用。

清单（默认 benchmark/speakers/clips.tsv，可复制后修改）：每行一段音频，Tab 分隔
    人名    用途    链接或本地文件    开始    结束    备注
    musk    enroll  https://www.youtube.com/watch?v=xxxx    1:05    1:15    某次访谈
    musk    test    https://www.youtube.com/watch?v=yyyy    3:20    3:30    换一个视频！
    unknown test    https://...                              0:10    0:20    没注册的人（陌生人）
- 用途：enroll（注册，每人约 10s）/ test（测试，必须和注册来自不同的视频）
- 人名为 unknown，或人名从没出现在 enroll 里，都按“陌生人”处理
- 开始/结束：秒数或 分:秒 / 时:分:秒；结束留空表示到结尾
- 链接：YouTube / B 站等视频页（需要 yt-dlp），直接的音频文件链接（.wav/.mp3…），或本地文件路径

依赖：ffmpeg（Windows: winget install ffmpeg；Mac: brew install ffmpeg）；视频网站链接还需要 yt-dlp（pip install yt-dlp）。
用法:
    python tools/fetch_speaker_clips.py                         # 用默认清单
    python tools/fetch_speaker_clips.py --list 我的清单.tsv
    python tools/fetch_speaker_clips.py --cookies-from-browser chrome   # YouTube 要求登录验证时
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST = ROOT / "benchmark" / "speakers" / "clips.tsv"
OUT = ROOT / "test_audio" / "speakers"
AUDIO_EXT = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus")


def to_sec(t: str) -> float | None:
    t = t.strip()
    if not t:
        return None
    sec = 0.0
    for part in t.split(":"):
        sec = sec * 60 + float(part)
    return sec


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout) else "失败")


def cut(src: Path, dst: Path, start: float | None, end: float | None) -> None:
    """截取 [start, end] 并转成 16kHz 单声道 16bit wav。"""
    cmd = ["ffmpeg", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)]
    run(cmd)


def fetch(link: str, tmp: Path, start, end, cookies: str) -> tuple[Path, bool]:
    """返回 (源文件, 是否已按时间段截好)。"""
    if not link.startswith(("http://", "https://")):
        p = Path(link).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise RuntimeError(f"本地文件不存在: {p}")
        return p, False
    if link.lower().split("?")[0].endswith(AUDIO_EXT):  # 直接的音频文件
        dst = tmp / ("src" + Path(link.split("?")[0]).suffix)
        req = urllib.request.Request(link, headers={"User-Agent": "speaker-clip-fetcher"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        return dst, False
    # 视频网站：用 yt-dlp 只下载需要的时间段
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-playlist", "-f", "bestaudio/best", "-x",
           "--audio-format", "wav", "-o", str(tmp / "src.%(ext)s")]
    if start is not None or end is not None:
        cmd += ["--download-sections", f"*{start or 0}-{end if end is not None else 'inf'}",
                "--force-keyframes-at-cuts"]
    if cookies:
        cmd += ["--cookies-from-browser", cookies]
    try:
        run(cmd + [link])
    except FileNotFoundError as e:
        raise RuntimeError("没有 yt-dlp：pip install yt-dlp") from e
    files = sorted(tmp.glob("src.*"))
    if not files:
        raise RuntimeError("yt-dlp 没有下载到文件")
    return files[0], True


def main():
    ap = argparse.ArgumentParser(description="按清单下载/截取说话人音频")
    ap.add_argument("--list", default=str(DEFAULT_LIST))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--cookies-from-browser", default="", help="如 chrome / edge / firefox，YouTube 要求验证时用")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("没有 ffmpeg。Windows: winget install ffmpeg（装完重开终端）；Mac: brew install ffmpeg")
        return 1
    rows = []
    for ln, line in enumerate(Path(args.list).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        cols = (line.split("\t") + [""] * 6)[:6]
        name, role, link, s, e, note = (c.strip() for c in cols)
        if role not in ("enroll", "test") or not name or not link or link.startswith("<"):
            print(f"跳过第 {ln} 行（未填写完整）: {line[:60]}")
            continue
        rows.append((ln, name, role, link, to_sec(s), to_sec(e), note))

    # 同一个人的注册和测试不能来自同一个视频
    for name in {r[1] for r in rows}:
        enroll_links = {r[3] for r in rows if r[1] == name and r[2] == "enroll"}
        same = [r for r in rows if r[1] == name and r[2] == "test" and r[3] in enroll_links]
        if same:
            print(f"⚠ {name}：第 {', '.join(str(r[0]) for r in same)} 行的测试音频和注册音频来自同一个链接，"
                  f"结果会偏乐观，建议换一个视频")

    out = Path(args.out)
    sources = ["文件\t链接\t开始\t结束\t备注"]
    ok = fail = 0
    for ln, name, role, link, s, e, note in rows:
        tag = hashlib.md5(f"{link}|{s}|{e}".encode()).hexdigest()[:8]
        dst = out / role / name / f"{name}_{role}_{tag}.wav"
        sources.append(f"{dst.relative_to(out)}\t{link}\t{s if s is not None else ''}\t{e if e is not None else ''}\t{note}")
        if dst.exists() and not args.force:
            print(f"✔ 已存在 {dst.relative_to(out)}")
            ok += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"… 第 {ln} 行 {name}/{role}: {link}", flush=True)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                src, already_cut = fetch(link, Path(tmp), s, e, args.cookies_from_browser)
                cut(src, dst, None if already_cut else s, None if already_cut else e)
            dur = dst.stat().st_size / 32000
            print(f"✔ {dst.relative_to(out)}（{dur:.1f}s）")
            ok += 1
        except Exception as ex:  # noqa: BLE001 —— 一行失败不影响其他行
            print(f"✘ 第 {ln} 行失败：{ex}")
            fail += 1
    (out / "sources.tsv").write_text("\n".join(sources) + "\n", encoding="utf-8")
    print(f"\n完成 {ok} 条，失败 {fail} 条。来源已记录到 {out / 'sources.tsv'}")
    print("下一步: python speaker_id.py")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
