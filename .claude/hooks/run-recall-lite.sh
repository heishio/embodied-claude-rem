#!/bin/bash
# recall-lite hook wrapper (uses memory-mcp venv for sudachipy)
export PYTHONUTF8=1
export MEMORY_DB_PATH="${MEMORY_DB_PATH:-$HOME/.claude/memories/memory.db}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PROJECT_DIR}/memory-mcp/.venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
"$PYTHON" "$SCRIPT_DIR/recall-lite.py" 2>/dev/null
exit 0
