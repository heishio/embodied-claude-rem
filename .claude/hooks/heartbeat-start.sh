#!/bin/bash
# heartbeat-start.sh - 心拍デーモン起動スクリプト (WSL2/Linux版)
# launchd の代わりに、バックグラウンドループで heartbeat-daemon.sh を5秒ごとに実行する
# 使い方: bash heartbeat-start.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAEMON="${SCRIPT_DIR}/heartbeat-daemon.sh"
PID_FILE="/tmp/heartbeat-daemon.pid"

# 既に起動中なら終了
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "heartbeat daemon already running (PID: $OLD_PID)"
        exit 0
    fi
fi

echo $$ > "$PID_FILE"
echo "heartbeat daemon started (PID: $$)"

trap 'rm -f "$PID_FILE"; exit 0' INT TERM

while true; do
    bash "$DAEMON" 2>/dev/null
    sleep 5
done
