#!/usr/bin/env python
"""session-end-buffer.py - SessionEndフックでアシスタント発話をキーワードバッファに追記する"""
import json
import os
import re
import sys

# === stdin読み取り ===
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)

transcript_path = data.get("transcript_path", "")

# fallback: transcript_pathが無い場合、最新のJSONLを探す
if not transcript_path or not os.path.exists(transcript_path):
    projects_dir = os.path.join(os.path.expanduser("~"), ".claude", "projects")
    if os.path.isdir(projects_dir):
        newest = None
        newest_mtime = 0
        for root, dirs, files in os.walk(projects_dir):
            for f in files:
                if f.endswith(".jsonl"):
                    fpath = os.path.join(root, f)
                    mtime = os.path.getmtime(fpath)
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest = fpath
        transcript_path = newest or ""

if not transcript_path or not os.path.exists(transcript_path):
    sys.exit(0)

# === トランスクリプトからアシスタントテキスト抽出 ===
texts = []

with open(transcript_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        if obj.get("type") != "assistant":
            continue

        # サブエージェントのメッセージはスキップ
        if obj.get("isSidechain"):
            continue

        message = obj.get("message", {})
        content = message.get("content", [])

        for c in content:
            if not isinstance(c, dict):
                continue

            # sayツールの発話テキスト
            if c.get("type") == "tool_use" and c.get("name") == "mcp__tts__say":
                say_text = c.get("input", {}).get("text", "")
                if say_text and len(say_text) >= 2:
                    texts.append(say_text)

            # テキスト出力（thinkingは除外）
            elif c.get("type") == "text":
                txt = c.get("text", "").strip()
                if txt and len(txt) >= 2:
                    texts.append(txt)

if not texts:
    sys.exit(0)

# === Sudachiで名詞・動詞抽出（keyword-buffer.pyと同じロジック） ===
try:
    from sudachipy import Dictionary
    tokenizer = Dictionary().create()
except ImportError:
    sys.exit(0)

combined = "\n".join(texts)

# サロゲート文字を除去
combined = combined.encode("utf-8", errors="ignore").decode("utf-8")

# <system-reminder>タグを除去
combined = re.sub(r"<system-reminder>.*?</system-reminder>", "", combined, flags=re.DOTALL)

try:
    tokens = tokenizer.tokenize(combined)
except Exception:
    sys.exit(0)

# ノイズ除去フィルタ
_PATH_CHARS = re.compile(r"[/\\.]")
_NOISE_WORDS = frozenset({
    "task", "notification", "output", "file", "status", "summary",
    "Background", "command", "tests", "exit", "Read", "retrieve",
    "result", "completed", "Users", "AppData", "Local", "Temp",
    "tasks", "Run", "mcp", "memory", "embodied", "claude", "code",
    "ClaudeCode",
})

def _is_noise(surface: str) -> bool:
    if surface in _NOISE_WORDS:
        return True
    if _PATH_CHARS.search(surface):
        return True
    if re.fullmatch(r"[0-9a-fA-F\-]+", surface):
        return True
    return False

words = []
verbs = []
for t in tokens:
    pos = t.part_of_speech()
    if pos[0] == "名詞" and len(t.surface()) >= 2:
        if not _is_noise(t.surface()):
            words.append(t.surface())
    elif pos[0] == "動詞":
        lemma = t.normalized_form()
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

# sensory_buffer.jsonlに追記
buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
os.makedirs(os.path.dirname(buf_path), exist_ok=True)
with open(buf_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
