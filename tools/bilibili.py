"""B 站取音频的兜底通道（yt-dlp 拿不到网页时用）。

背景：有些公司网络/风控环境下，B 站视频页（www.bilibili.com/video/...）会返回
HTTP 412，yt-dlp 连页面都打不开；但 api.bilibili.com 通常还能用。这里直接走
官方 API 取 DASH 音频直链，再用 ffmpeg 下载指定时间段。

用法：
    from bilibili import is_bilibili, resolve_audio_url, BVID_RE
    url, title, dur = resolve_audio_url("https://www.bilibili.com/video/BV1AbwSzeEKD")

注意：直链有时效（约 2 小时），每次请求重新取。
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
BVID_RE = r"BV[0-9A-Za-z]{10}"


def is_bilibili(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return any(h in host for h in ("bilibili.com", "b23.tv", "acg.tv"))


def _get(url: str, cookie: str = "", referer: str = "https://www.bilibili.com/", tries: int = 3):
    last = None
    for i in range(tries):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": UA,
                "Referer": referer,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Cookie": cookie,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code in (412, 429):
                time.sleep(2 + 3 * i)  # 风控：退避后重试
                continue
            raise
    raise RuntimeError(f"B 站接口请求失败（{last}），可稍后重试或改用 yt-dlp")


def make_cookie() -> str:
    """先取 buvid3/buvid4，再补齐 _uuid / b_nut / b_lsid —— 缺这些会被判爬虫（412）。"""
    spi = _get("https://api.bilibili.com/x/frontend/finger/spi", cookie="")
    b3 = spi["data"]["b_3"]
    b4 = spi["data"]["b_4"]

    def hx(n: int) -> str:
        return "".join(random.choice("0123456789abcdef") for _ in range(n))

    uuid = f"{random.randint(0, 10**7)}-{hx(4)}-4{hx(3)}-a{hx(3)}-{hx(12)}infoc"
    lsid = "".join(random.choice("0123456789ABCDEF") for _ in range(16))
    return (
        f"buvid3={b3}; buvid4={b4}; _uuid={uuid}; "
        f"b_nut={int(time.time())}; b_lsid={lsid}; CURRENT_FNVAL=4048; theme=light"
    )


def bvid_of(url: str) -> str:
    """接受完整链接 / BV 号 / b23.tv 短链。"""
    import re

    m = re.search(BVID_RE, url)
    if m:
        return m.group(0)
    if "b23.tv" in url:  # 短链：跟随跳转拿 BV 号
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            m = re.search(BVID_RE, r.geturl())
            if m:
                return m.group(0)
    raise RuntimeError(f"链接里找不到 BV 号: {url}")


def resolve_audio_url(url: str) -> tuple[str, str, int]:
    """返回 (音频直链, 标题, 时长秒)。"""
    bvid = bvid_of(url)
    ck = make_cookie()
    v = _get(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}", ck)
    if v.get("code") != 0:
        raise RuntimeError(f"取视频信息失败: {v.get('code')} {v.get('message')}")
    d = v["data"]
    cid = d["cid"]
    q = urllib.parse.urlencode(
        {"bvid": bvid, "cid": cid, "fnval": "16", "fnver": "0", "fourk": "1", "qn": "80"}
    )
    pu = _get(f"https://api.bilibili.com/x/player/playurl?{q}", ck)
    if pu.get("code") != 0:
        raise RuntimeError(f"取播放地址失败: {pu.get('code')} {pu.get('message')}")
    audios = (pu.get("data", {}).get("dash") or {}).get("audio") or []
    if not audios:
        raise RuntimeError("该视频没有 DASH 音频轨（可能是付费/大会员内容）")
    audios.sort(key=lambda a: a.get("bandwidth", 0))
    return audios[-1]["baseUrl"], d.get("title", bvid), int(d.get("duration") or 0)


def download_section(url: str, dst: Path, start: float | None, end: float | None) -> Path:
    """用 ffmpeg 从直链截取 [start, end) 并转成 16kHz 单声道 16bit wav。"""
    import subprocess

    audio, _title, _dur = resolve_audio_url(url)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-headers", f"Referer: https://www.bilibili.com/\r\nUser-Agent: {UA}\r\n"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", audio]
    if end is not None:
        cmd += ["-t", str(end - (start or 0))]
    cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError("ffmpeg 截取失败: " + (r.stderr or "").strip().splitlines()[-1])
    return dst
