#!/bin/bash
# uncertainty-check.sh - 不確実性駆動の能動知覚
# UserPromptSubmitフックで実行。ユーザー入力のテキストを分析し、
# 不確実性スコアを算出してコンテキストに注入する。
# Claude は AGENTS.md のルールに従い、スコアを見てカメラを使うか判断する。

# Git Bash の /tmp は Windows Python から見えないため、実パスに変換
if command -v cygpath &>/dev/null; then
    TMPDIR_REAL="$(cygpath -m /tmp)"
else
    TMPDIR_REAL="/tmp"
fi

COOLDOWN_FILE="${TMPDIR_REAL}/uncertainty_last_check.txt"
COOLDOWN_SEC=120  # 2分

# stdin を一旦保存（Python に2回渡すため）
STDIN_DATA=$(cat)

# 入力が空なら何も出力せず終了
if [ -z "$STDIN_DATA" ]; then
    exit 0
fi

# Python で不確実性スコアを算出（stdin 経由でJSON全体を渡す）
# PYTHONUTF8=1 で cp932 問題を回避
RESULT=$(echo "$STDIN_DATA" | PYTHONUTF8=1 python -c "
import re, sys, json

data = json.load(sys.stdin)
text = data.get('prompt', '').strip()

# 空なら対象外
if not text:
    sys.exit(0)

score = 0.0
sig_type = 'normal'

# 日本語判定（IMEアーティファクトを除く）
has_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
has_fullwidth_latin = bool(re.search(r'[\uFF01-\uFF5E]', text))  # 全角英数記号
is_short = len(text) <= 6

# 全角英数 + かな/漢字混在 = IMEアーティファクトの可能性
# 半角英数も混在していればさらに強いシグナル
has_halfwidth_latin = bool(re.search(r'[a-zA-Z0-9]', text))
mixed_scripts = has_japanese and (has_fullwidth_latin or (has_halfwidth_latin and is_short))

if mixed_scripts:
    score += 0.6
    sig_type = 'ime_artifact'
# 全角英数なし・半角英数なしの純粋な日本語 = 意図的な入力
elif has_japanese:
    sys.exit(0)

# よく使う短い英単語・略語・コマンド（辞書）
known_words = {
    'ok','yes','no','hi','hey','bye','thx','ty','np','lol','omg','wtf',
    'brb','afk','gg','gl','hf','pls','plz','idk','imo','fyi','asap',
    'help','quit','exit','stop','go','run','test','fix','add','del',
    'git','ls','cd','rm','cp','mv','cat','pip','npm','uv',
    'y','n','q','x',
    'a','i',  # 英語の冠詞・代名詞
    'commit','push','pull','merge','diff','log','status',
    'done','good','nice','cool','great','sure','fine','nope','yep','yup',
    'what','why','how','who','where','when',
}

lower = text.lower().strip()

# 既知の単語なら対象外
if lower in known_words:
    sys.exit(0)

# IME artifact は既にスコア付き、以下は英数入力向けの追加判定
# 同じ文字の繰り返し（寝落ちでキーを押し続ける: 'mmmmmm', 'aaaa'）
unique_chars = set(lower)
if len(unique_chars) <= 2 and len(text) >= 2:
    score += 0.5
    if sig_type == 'normal':
        sig_type = 'repeated'

# 母音が少ない（英語基準）
vowel_ratio = len(re.findall(r'[aiueoAIUEO]', text)) / max(len(text), 1)
if vowel_ratio < 0.15 and not has_japanese:
    score += 0.4
    if sig_type == 'normal':
        sig_type = 'gibberish'

# 辞書にない英数字のみの入力（メインシグナル）
if re.match(r'^[a-zA-Z0-9]+$', text) and lower not in known_words:
    score += 0.4
    if sig_type == 'normal':
        sig_type = 'unknown_word'

# 数字と文字が混在（コード片やタイポ: '091a', '3fg'）
if re.search(r'[0-9]', text) and re.search(r'[a-zA-Z]', text):
    score += 0.2
    if sig_type == 'normal':
        sig_type = 'mixed_alnum'

# 極端に短い（1-3文字）ボーナス
if len(text) <= 3:
    score += 0.1

score = min(score, 1.0)

# 閾値以下なら出力しない
if score < 0.5:
    sys.exit(0)

print(f'{score:.2f} {sig_type}')
" 2>/dev/null)

# Python がスコアを出力しなかった場合（正常テキスト）
if [ -z "$RESULT" ]; then
    exit 0
fi

SCORE=$(echo "$RESULT" | awk '{print $1}')
TYPE=$(echo "$RESULT" | awk '{print $2}')

# クールダウン判定
COOLDOWN_STATUS="ok"
if [ -f "$COOLDOWN_FILE" ]; then
    LAST_CHECK=$(cat "$COOLDOWN_FILE" 2>/dev/null)
    NOW=$(date +%s)
    if [ -n "$LAST_CHECK" ]; then
        ELAPSED=$((NOW - LAST_CHECK))
        if [ "$ELAPSED" -lt "$COOLDOWN_SEC" ]; then
            COOLDOWN_STATUS="wait"
        fi
    fi
fi

# クールダウンタイムスタンプを更新（cooldown=ok の場合のみ）
if [ "$COOLDOWN_STATUS" = "ok" ]; then
    date +%s > "$COOLDOWN_FILE"
fi

echo "[uncertainty] score=${SCORE} type=${TYPE} cooldown=${COOLDOWN_STATUS}"
exit 0
