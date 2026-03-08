#!/bin/bash
# hearing-stop-hook.sh — Stop hook で聴覚バッファをチェックし、
# 新しい発話があればターンを延長する。
#
# バッファ管理: 行番号ベース
#   - バッファは消さずに読む（offset以降の新しい行だけ処理）
#   - 有効 → offset更新 & 処理済み行を削除 & block
#   - 無効 → offset据え置き → 短いsleep後に即リトライ（新データが溜まるのを待つ）

BUFFER_FILE="/tmp/hearing_buffer.jsonl"
PID_FILE="/tmp/hearing-daemon.pid"
OFFSET_FILE="/tmp/hearing_stop_offset"
TIMING_LOG="/tmp/hearing_timing.log"

# タイミング記録
NOW=$(python3 -c "import time; print(f'{time.time():.3f}')")
PREV=$(cat /tmp/hearing_stop_last_ts 2>/dev/null || echo "$NOW")
DELTA=$(python3 -c "print(f'{$NOW - $PREV:.1f}')")
echo "$NOW" > /tmp/hearing_stop_last_ts
echo "[$(date +%H:%M:%S)] stop-hook-start delta=${DELTA}s count=${COUNT:-?}" >> "$TIMING_LOG"
MAX_HEARING_CONTINUES=${MAX_HEARING_CONTINUES:-5}
COUNTER_FILE="/tmp/hearing-stop-counter"
WAIT_SECONDS=${HEARING_WAIT_SECONDS:-5}
RETRY_WAIT=${HEARING_RETRY_WAIT:-3}
NO_SPEECH_THRESHOLD=${HEARING_NO_SPEECH_THRESHOLD:-0.6}

# ── デーモン稼働確認 ──────────────────────────────────────────
DAEMON_RUNNING=false
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        DAEMON_RUNNING=true
    fi
fi

[ "$DAEMON_RUNNING" = "false" ] && exit 0

# ── カウンタ読み込み & 上限チェック ────────────────────────────
COUNT=$(cat "$COUNTER_FILE" 2>/dev/null || echo 0)

if [ "$COUNT" -ge "$MAX_HEARING_CONTINUES" ]; then
    rm -f "$COUNTER_FILE"
    exit 0
fi

# ── 応答を待つ ────────────────────────────────────────────────
sleep "$WAIT_SECONDS"

# ── バッファを行番号ベースで読み取り・判定 ─────────────────────
RESULT=$(python3 - "$NO_SPEECH_THRESHOLD" "$OFFSET_FILE" "$BUFFER_FILE" "$RETRY_WAIT" "$COUNT" <<'PYEOF' 2>>/tmp/hearing_timing.log
import json
import os
import sys
import time
from pathlib import Path

threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
offset_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/hearing_stop_offset")
buffer_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("/tmp/hearing_buffer.jsonl")
retry_wait = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

def read_offset():
    try:
        return int(offset_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0

def write_offset(n):
    offset_file.write_text(str(n))

def read_buffer_from(start_line):
    """バッファのstart_line行目以降を読む（0-indexed）"""
    if not buffer_file.exists() or buffer_file.stat().st_size == 0:
        return [], 0
    lines = []
    total = 0
    with open(buffer_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            total = i + 1
            if i >= start_line:
                lines.append((i, line))
    return lines, total

def filter_entries(lines):
    entries = []
    for line_no, line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        try:
            e = json.loads(line_s)
            if e.get("no_speech_prob", 1.0) <= threshold:
                entries.append(e)
        except json.JSONDecodeError:
            pass
    return entries

def truncate_buffer(up_to_line):
    """処理済み行をバッファから削除（up_to_line行目まで削除、それ以降を残す）"""
    if not buffer_file.exists():
        return
    with open(buffer_file, encoding="utf-8") as f:
        all_lines = f.readlines()
    remaining = all_lines[up_to_line:]
    with open(buffer_file, "w", encoding="utf-8") as f:
        f.writelines(remaining)
    # offset をリセット（バッファが切り詰められたので）
    write_offset(0)

def fmt_time(ts):
    if "T" in ts:
        return ts.split("T")[1][:8]
    return ts

def try_read():
    offset = read_offset()
    lines, total = read_buffer_from(offset)
    if not lines:
        return None, total
    entries = filter_entries(lines)
    if not entries:
        return None, total
    # 有効 → 出力 & バッファ切り詰め
    last_line_no = lines[-1][0]
    truncate_buffer(last_line_no + 1)
    n = len(entries)
    first_ts = fmt_time(entries[0]["ts"])
    last_ts = fmt_time(entries[-1]["ts"])
    texts = [e["text"] for e in entries]
    combined = " / ".join(texts)
    return f"[hearing] chunks={n} span={first_ts}~{last_ts} text={combined}", total

# チェーン保証回数: バッファが空でもこの回数まではリトライする
MIN_GUARANTEED = 2
count = int(sys.argv[5]) if len(sys.argv) > 5 else 0
t_start = time.time()

debug_lines = []
# 最大3回リトライ（retry_wait間隔）
for attempt in range(3):
    result, total = try_read()
    elapsed = time.time() - t_start
    debug_lines.append(f"  retry{attempt}: {'HIT' if result else 'empty'} buf_lines={total} +{elapsed:.1f}s")
    if result:
        # デバッグ情報を結果の後ろに付与
        debug = " | ".join(debug_lines)
        print(f"{result} [debug: count={count} {debug}]")
        sys.exit(0)
    # バッファ空でも保証回数以内ならリトライ
    if count >= MIN_GUARANTEED:
        debug_lines.append(f"  => skip retry (count={count} >= guaranteed={MIN_GUARANTEED})")
        break
    time.sleep(retry_wait)

# 全リトライ失敗時もデバッグ出力
debug = " | ".join(debug_lines)
print(f"[hearing-debug] no speech. count={count} {debug}", file=sys.stderr)
sys.exit(0)
PYEOF
)

# ── 判定 ──────────────────────────────────────────────────────
END_TS=$(python3 -c "import time; print(f'{time.time():.3f}')")
if [ -n "$RESULT" ]; then
    echo $((COUNT + 1)) > "$COUNTER_FILE"
    ELAPSED=$(python3 -c "print(f'{$END_TS - $NOW:.1f}')")
    echo "[$(date +%H:%M:%S)] stop-hook-block elapsed=${ELAPSED}s chain=$((COUNT+1))" >> "$TIMING_LOG"
    ESCAPED=$(echo "$RESULT" | sed 's/"/\\"/g')
    echo "{\"decision\": \"block\", \"reason\": \"Stop hook feedback:\n${ESCAPED}\nチェイン($((COUNT+1))/${MAX_HEARING_CONTINUES})\"}"
else
    ELAPSED=$(python3 -c "print(f'{$END_TS - $NOW:.1f}')")
    # チェーン保証: 保証回数以内なら空でもblockして待機
    MIN_GUARANTEED=2
    if [ "$COUNT" -lt "$MIN_GUARANTEED" ]; then
        echo $((COUNT + 1)) > "$COUNTER_FILE"
        echo "[$(date +%H:%M:%S)] stop-hook-wait elapsed=${ELAPSED}s chain=$((COUNT+1)) (guaranteed)" >> "$TIMING_LOG"
        echo "{\"decision\": \"block\", \"reason\": \"Stop hook feedback:\n[hearing] 聞き取り待機中... ($((COUNT+1))/${MAX_HEARING_CONTINUES})\nまだ聞き取れていないけど、静かに続きを待っています。\"}"
    else
        echo "[$(date +%H:%M:%S)] stop-hook-pass elapsed=${ELAPSED}s (no speech)" >> "$TIMING_LOG"
        rm -f "$COUNTER_FILE"
        exit 0
    fi
fi
