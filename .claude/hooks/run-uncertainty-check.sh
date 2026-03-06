#!/bin/bash
# uncertainty-check hook wrapper
export PYTHONUTF8=1
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/uncertainty-check.sh" 2>/dev/null
exit 0
