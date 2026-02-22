#!/bin/bash
# interoception.sh - AIの内受容感覚（interoception）
# UserPromptSubmitフックで毎ターン実行される
# heartbeat-daemon.sh が書き出した state file を読んでコンテキストに注入する
# + ユーザー入力のトーン分析（軽量キーワードベース）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Git Bash の /tmp は Windows Python から見えないため、実パスに変換
if command -v cygpath &>/dev/null; then
    STATE_FILE="$(cygpath -m /tmp)/interoception_state.json"
else
    STATE_FILE="/tmp/interoception_state.json"
fi

# セッション検出（前回プロンプトから10分以上空いたら新セッション）
if command -v cygpath &>/dev/null; then
    SESSION_MARKER="$(cygpath -m /tmp)/interoception_session.ts"
else
    SESSION_MARKER="/tmp/interoception_session.ts"
fi

SESSION_GAP=600  # 10分
SESSION_STATUS="continuing"
NOW_EPOCH=$(date +%s)

if [ ! -f "$SESSION_MARKER" ]; then
    SESSION_STATUS="new"
else
    LAST_PROMPT=$(cat "$SESSION_MARKER" 2>/dev/null || echo 0)
    ELAPSED=$((NOW_EPOCH - LAST_PROMPT))
    if [ "$ELAPSED" -gt "$SESSION_GAP" ]; then
        SESSION_STATUS="new"
    fi
fi
echo "$NOW_EPOCH" > "$SESSION_MARKER"

# stdin から JSON を読み取り、prompt を取得してトーン分析
STDIN_DATA=$(cat)
TONE=$(echo "$STDIN_DATA" | PYTHONUTF8=1 python -c "
import sys, json
data = json.load(sys.stdin)
prompt = data.get('prompt', '')
print(prompt)
" 2>/dev/null | PYTHONUTF8=1 python "$SCRIPT_DIR/tone-analyzer.py" 2>/dev/null)
TONE="${TONE:-neutral}"

# heartbeat daemon が動いていなければ自動起動
PID_FILE="/tmp/heartbeat-daemon.pid"
DAEMON_RUNNING=false
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        DAEMON_RUNNING=true
    fi
fi

if [ "$DAEMON_RUNNING" = false ]; then
    nohup bash "$SCRIPT_DIR/heartbeat-start.sh" >/dev/null 2>&1 &
    # state file がまだないので初回はフォールバック
    if [ ! -f "$STATE_FILE" ]; then
        CURRENT_TIME=$(date '+%H:%M:%S')
        echo "[interoception] time=${CURRENT_TIME} tone=${TONE} session=${SESSION_STATUS} (heartbeat daemon started)"
        exit 0
    fi
fi

# state file から読み取って1行に整形
PYTHONUTF8=1 python -c "
import json, sys

tone = '${TONE}'
session = '${SESSION_STATUS}'

try:
    with open('${STATE_FILE}') as f:
        data = json.load(f)
    now = data.get('now', {})
    trend = data.get('trend', {})
    window = data.get('window', [])

    # トレンド矢印
    arrows = {'rising': '↑', 'falling': '↓', 'stable': '→'}
    ar_arrow = arrows.get(trend.get('arousal', 'stable'), '→')
    mem_arrow = arrows.get(trend.get('mem_free', 'stable'), '→')

    # タイムスタンプから時刻部分だけ
    ts = now.get('ts', '?')
    if 'T' in ts:
        time_part = ts.split('T')[1][:8]
    else:
        time_part = ts

    parts = [
        f\"time={time_part}\",
        f\"phase={now.get('phase', '?')}\",
        f\"arousal={now.get('arousal', '?')}%({ar_arrow})\",
        f\"thermal={now.get('thermal', '?')}\",
        f\"mem_free={now.get('mem_free', '?')}%({mem_arrow})\",
        f\"uptime={now.get('uptime_min', '?')}min\",
        f\"heartbeats={len(window)}\",
        f\"tone={tone}\",
        f\"session={session}\",
    ]
    print('[interoception] ' + ' '.join(parts))
except Exception as e:
    print(f'[interoception] error reading state: {e}', file=sys.stderr)
    print('[interoception] state_file_error')
" 2>/dev/null

exit 0
