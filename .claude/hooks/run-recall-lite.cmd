@echo off
chcp 65001 >nul 2>nul
set PYTHONUTF8=1
python "%~dp0recall-lite.py" 2>nul
exit /b 0
