@echo off
REM vision-server 起動スクリプト
cd /d "%~dp0"
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    .venv\Scripts\pip install open-clip-torch mediapipe opencv-python fastapi uvicorn pydantic numpy
)
echo Starting vision-server on port 8100...
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8100 --log-level info
