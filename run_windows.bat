@echo off
REM One-click start for Windows: create venv, install deps, download models, start server.
REM Usage: double-click, or run in cmd:  run_windows.bat   (add  --hf-mirror  if HuggingFace is slow)
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

if not exist .venv\Scripts\python.exe (
    echo [1/4] Creating virtual env .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 ( echo Python 3.10+ not found. Install from https://www.python.org/downloads/ and tick "Add to PATH". & pause & exit /b 1 )
)
echo [2/4] Installing dependencies ...
.venv\Scripts\python.exe -m pip install -q --upgrade pip
.venv\Scripts\python.exe -m pip install -q -r requirements.txt
if errorlevel 1 ( echo pip install failed. & pause & exit /b 1 )

echo [3/4] Downloading models (skips ones already downloaded) ...
.venv\Scripts\python.exe download_models.py %*

echo [4/4] Starting server: http://127.0.0.1:8000
.venv\Scripts\python.exe server.py
pause
