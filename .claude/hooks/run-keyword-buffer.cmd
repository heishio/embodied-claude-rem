@echo off
chcp 65001 >nul 2>nul
set PYTHONUTF8=1
python "%~dp0keyword-buffer.py" 2>nul
exit /b 0
