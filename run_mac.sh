#!/usr/bin/env bash
# 一键启动（macOS / Linux）：建 venv、装依赖、下载模型、启动服务
# 用法: ./run_mac.sh            （HuggingFace 慢可加 --hf-mirror）
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "[1/4] 创建虚拟环境 .venv ..."
  "$PY" -m venv .venv
fi
echo "[2/4] 安装依赖 ..."
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt
echo "[3/4] 下载模型（已下载的会跳过）..."
.venv/bin/python download_models.py "$@" || echo "部分模型下载失败，demo 会跳过缺失模型"
echo "[4/4] 启动服务: http://127.0.0.1:8000"
.venv/bin/python server.py
