#!/bin/bash
# keyword-buffer hook wrapper
export PYTHONUTF8=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/keyword-buffer.py" 2>/dev/null
exit 0
