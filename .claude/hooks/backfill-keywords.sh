#!/bin/bash
# backfill-keywords.sh — 過去のトランスクリプトから keyword-buffer を一括生成（バッチ版）
# 使い方: ./backfill-keywords.sh --since=2026-02-15 --until=2026-03-05

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PROJECT_DIR}/memory-mcp/.venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3

SINCE="${1:---since=2026-02-15}"
UNTIL="${2:---until=$(date +%Y-%m-%d)}"

TMPFILE=$(mktemp /tmp/backfill-transcript.XXXXXX)
trap 'rm -f "$TMPFILE"' EXIT

echo "Fetching transcripts ($SINCE $UNTIL)..."
npx ccconv raws --format=talk "$SINCE" "$UNTIL" 2>/dev/null > "$TMPFILE"

if [ ! -s "$TMPFILE" ]; then
  echo "No transcripts found."
  exit 0
fi

LINES=$(wc -l < "$TMPFILE")
echo "Transcript: $LINES lines"

touch ~/.claude/sensory_buffer.jsonl
BEFORE=$(wc -l < ~/.claude/sensory_buffer.jsonl)

"$PYTHON" -u "$SCRIPT_DIR/backfill-keywords-batch.py" < "$TMPFILE"

AFTER=$(wc -l < ~/.claude/sensory_buffer.jsonl)
echo "Buffer: $BEFORE -> $AFTER entries (+$((AFTER - BEFORE)))"
echo "Run 'crystallize' in memory-mcp to convert to verb chains."
