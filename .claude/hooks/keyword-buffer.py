#!/usr/bin/env python
"""keyword-buffer.py - 会話キーワードを雑に溜めるフックスクリプト"""
import json
import os
import sys

text = ""
try:
    data = json.load(sys.stdin)
    text = data.get("prompt", "")
except Exception:
    sys.exit(0)

if not text or len(text) < 2:
    sys.exit(0)

# autonomous-action のプロンプトはバッファに入れない
if os.environ.get("CLAUDE_AUTONOMOUS"):
    sys.exit(0)
if "自律行動タイム" in text:
    sys.exit(0)

# サロゲート文字を除去（Windowsのstdin経由で入ることがある）
text = text.encode("utf-8", errors="ignore").decode("utf-8")

try:
    from sudachipy import Dictionary

    tokenizer = Dictionary().create()
except ImportError:
    sys.exit(0)

try:
    tokens = tokenizer.tokenize(text)
except Exception:
    sys.exit(0)

# 名詞・固有名詞のみ、2文字以上
words = []
# 動詞（原形で保存、出現順を維持）
verbs = []
for t in tokens:
    pos = t.part_of_speech()
    if pos[0] == "名詞" and len(t.surface()) >= 2:
        words.append(t.surface())
    elif pos[0] == "動詞":
        # 辞書形（原形）で保存
        lemma = t.dictionary_form()
        if len(lemma) >= 2:
            verbs.append(lemma)

if not words and not verbs:
    sys.exit(0)

# 重複除去（順序保持）
def dedupe(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

unique_words = dedupe(words)
unique_verbs = dedupe(verbs)

entry = {}
if unique_words:
    entry["w"] = unique_words
if unique_verbs:
    entry["v"] = unique_verbs

if not entry:
    sys.exit(0)

buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
os.makedirs(os.path.dirname(buf_path), exist_ok=True)
with open(buf_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
