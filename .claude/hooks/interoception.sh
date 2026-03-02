#!/bin/bash
# interoception.sh - 現在時刻をコンテキストに注入
# UserPromptSubmitフックで毎ターン実行される

CURRENT_TIME=$(date '+%H:%M:%S')

# stdin 消費（hookの仕様上必要）
cat > /dev/null

echo "[interoception] time=${CURRENT_TIME}"

exit 0
