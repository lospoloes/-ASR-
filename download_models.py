"""下载模型到 models/<key>/。优先 HuggingFace，失败自动回退 GitHub Release。

示例:
    python download_models.py                     # 默认模型 + 说话人分离模型
    python download_models.py --list              # 列出所有可选模型
    python download_models.py --models sensevoice,dolphin-base
    python download_models.py --tts               # 额外下载 TTS（生成测试音频用）
    python download_models.py --hf-mirror         # 国内网络：走 hf-mirror.com
    python download_models.py --source github --gh-proxy https://<你的github加速前缀>/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import registry


def log(msg: str):
    print(msg, flush=True)


# ---------------------------------------------------------------- HuggingFace
def download_hf(spec: dict, dest: Path) -> None:
    from huggingface_hub import snapshot_download  # 延迟导入，保证 HF_ENDPOINT 已设置

    log(f"    [HF] {os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}/{spec['hf_repo']}")
    snapshot_download(repo_id=spec["hf_repo"], allow_patterns=spec["hf_allow"], local_dir=str(dest))


# ---------------------------------------------------------------- GitHub
def _fetch(url: str, out: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "asr-demo-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(out, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, last = 0, 0.0
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if time.time() - last > 1:
                pct = f"{done * 100 / total:5.1f}%" if total else ""
                print(f"\r      {done / 1048576:8.1f} MB {pct}", end="", flush=True)
                last = time.time()
    print(f"\r      {done / 1048576:8.1f} MB 完成      ", flush=True)


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for m in tar.getmembers():
        target = (path / m.name).resolve()
        if base != target and base not in target.parents:
            raise RuntimeError(f"压缩包包含越界路径: {m.name}")
    if sys.version_info >= (3, 12):
        tar.extractall(path, filter="data")
    else:
        tar.extractall(path)


def download_github(spec: dict, dest: Path, gh_proxy: str = "") -> None:
    url = spec.get("gh_url")
    if not url:
        raise RuntimeError("该模型没有 GitHub 下载地址")
    url = gh_proxy + url if gh_proxy else url
    log(f"    [GitHub] {url}")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tmp = Path(tmp)
        fname = tmp / url.rsplit("/", 1)[-1]
        _fetch(url, fname)
        if fname.suffix == ".onnx":
            shutil.move(str(fname), str(dest / fname.name))
            return
        log("      解压中 ...")
        ex = tmp / "x"
        ex.mkdir()
        with tarfile.open(fname, "r:*") as t:
            _safe_extract(t, ex)
        tops = list(ex.iterdir())
        src = tops[0] if len(tops) == 1 and tops[0].is_dir() else ex
        for p in src.iterdir():
            if p.name == "test_wavs":  # 保留官方测试音频，方便对比
                target = dest / "test_wavs"
                if target.exists():
                    shutil.rmtree(target)
            elif (dest / p.name).exists():
                if (dest / p.name).is_dir():
                    shutil.rmtree(dest / p.name)
                else:
                    (dest / p.name).unlink()
            shutil.move(str(p), str(dest / p.name))


# ---------------------------------------------------------------- 量化
def maybe_quantize(spec: dict, keep_fp32: bool) -> None:
    """官方压缩包只有 fp32 时，用 onnxruntime 动态量化成 int8（和 icefall 导出 int8 的方式一致）。"""
    q = spec.get("quantize")
    if not q:
        return
    dest = registry.MODELS_DIR / spec["key"]
    src, dst = dest / q["src"], dest / q["dst"]
    if dst.exists() or not src.exists():
        return
    log(f"      官方包没有 int8 文件，正在量化 {src.name} -> {dst.name}（约 10~60 秒）...")
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    log(f"      量化完成: {dst.stat().st_size / 1048576:.1f} MB")
    if not keep_fp32:
        src.unlink()


# ---------------------------------------------------------------- 主流程
def fetch_one(spec: dict, source: str, gh_proxy: str, force: bool, keep_fp32: bool = False) -> tuple[bool, str]:
    dest = registry.MODELS_DIR / spec["key"]
    if not force:
        maybe_quantize(spec, keep_fp32)
        if registry.resolve_files(spec):
            return True, "已存在，跳过"
    order = {"auto": ["hf", "github"], "hf": ["hf"], "github": ["github"]}[source]
    errors = []
    for src in order:
        try:
            if src == "hf":
                download_hf(spec, dest)
            else:
                download_github(spec, dest, gh_proxy)
            maybe_quantize(spec, keep_fp32)
            if registry.resolve_files(spec):
                return True, f"OK（来源: {src}）"
            errors.append(f"{src}: 下载完成但缺少必要文件")
        except Exception as e:  # noqa: BLE001
            msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
            errors.append(f"{src}: {type(e).__name__}: {msg}")
            log(f"      失败 -> {errors[-1]}")
    return False, " | ".join(errors)


def main():
    ap = argparse.ArgumentParser(description="下载端侧 ASR 对比 demo 所需模型")
    ap.add_argument("--models", default="", help="逗号分隔的 key；默认下载 default=True 的模型")
    ap.add_argument("--all", action="store_true", help="下载注册表里全部模型（含实验性）")
    ap.add_argument("--list", action="store_true", help="只列出模型")
    ap.add_argument("--no-diarization", action="store_true", help="不下载分角色（说话人分离）模型")
    ap.add_argument("--tts", action="store_true", help="额外下载 TTS 模型（make_test_audio.py 生成测试音频用）")
    ap.add_argument("--source", choices=["auto", "hf", "github"], default="auto")
    ap.add_argument("--hf-mirror", action="store_true", help="使用 https://hf-mirror.com（国内网络推荐）")
    ap.add_argument("--gh-proxy", default="", help="GitHub 下载加速前缀，拼在原始 URL 前面")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--keep-fp32", action="store_true", help="量化后保留 fp32 原文件（默认删除以省空间）")
    args = ap.parse_args()

    if args.hf_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    if args.list:
        log(f"{'key':30} {'默认':4} {'流式':4} {'文档体积MB':>10}  名称")
        for s in registry.ASR_MODELS:
            log(f"{s['key']:30} {'是' if s['default'] else '否':4} {'是' if s['streaming'] else '否':4} "
                f"{s['doc_size_mb']:>10}  {s['name']}")
        for s in registry.DIARIZATION_MODELS:
            log(f"{s['key']:30} {'是':4} {'-':4} {str(s['doc_size_mb']):>10}  {s['name']}（分角色）")
        for s in registry.TTS_MODELS:
            log(f"{s['key']:30} {'否':4} {'-':4} {str(s['doc_size_mb']):>10}  {s['name']}（--tts）")
        for s in registry.VAD_MODELS:
            log(f"{s['key']:30} {'是':4} {'-':4} {str(s['doc_size_mb']):>10}  {s['name']}（长音频切段）")
        return 0

    if args.all:
        specs = list(registry.ASR_MODELS)
    elif args.models:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        specs = [registry.get_spec(k) for k in keys]
    else:
        specs = [s for s in registry.ASR_MODELS if s["default"]]
    if not args.no_diarization:
        specs += [s for s in registry.DIARIZATION_MODELS if s not in specs]
    specs += [s for s in registry.VAD_MODELS if s not in specs]
    if args.tts:
        specs += [s for s in registry.TTS_MODELS if s not in specs]

    registry.MODELS_DIR.mkdir(exist_ok=True)
    summary = []
    for i, spec in enumerate(specs, 1):
        log(f"\n[{i}/{len(specs)}] {spec['name']}  ({spec['key']})")
        ok, msg = fetch_one(spec, args.source, args.gh_proxy, args.force, args.keep_fp32)
        size = registry.disk_size_mb(registry.resolve_files(spec))
        log(f"    => {'成功' if ok else '失败'}: {msg}" + (f"，磁盘 {size} MB" if size else ""))
        summary.append((spec["key"], ok, msg, size))

    log("\n================ 汇总 ================")
    for key, ok, msg, size in summary:
        log(f"{'✔' if ok else '✘'} {key:30} {(str(size) + ' MB') if size else '':>10}  {'' if ok else msg}")
    failed = [k for k, ok, *_ in summary if not ok]
    if failed:
        log("\n部分模型下载失败。可尝试: --hf-mirror（国内）、--source github、--gh-proxy <前缀>，"
            "或手动下载后解压到 models/<key>/ 。demo 会自动跳过缺失的模型。")
        return 1
    log("\n全部完成。下一步: python server.py")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    sys.exit(main())
