@echo off
REM start-all.cmd - Embodied Claude 一括起動
REM   1. SBV2 TTS サーバー
REM   2. Heartbeat daemon
REM   3. Claude Code (このウィンドウで起動)

echo === Embodied Claude - Starting All ===


REM --- SBV2 TTS サーバー (別ウィンドウ) ---
echo Starting SBV2 TTS Server...
start "SBV2 TTS Server" "D:\Tools\Style-Bert-VITS2\Server.bat"

REM --- Heartbeat daemon (バックグラウンド) ---
echo Starting Heartbeat daemon...
start /min "Heartbeat Daemon" powershell -ExecutionPolicy Bypass -WindowStyle Hidden -File "%~dp0.claude\hooks\heartbeat-daemon.ps1"

REM --- Claude Code (このウィンドウ) ---
echo Starting Claude Code...
cd /d "%~dp0"
claude --dangerously-skip-permissions