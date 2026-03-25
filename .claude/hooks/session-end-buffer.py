#!/usr/bin/env python
"""session-end-buffer.py - SessionEndフックでアシスタント発話をキーワードバッファに追記する

各テキスト（Say発話/テキスト出力）を個別エントリとしてバッファに書く。
Sudachi初期化は1回、tokenizeはテキストごと。
"""
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

# === Sudachi初期化（1回だけ） ===
try:
    from sudachipy import Dictionary
    tokenizer = Dictionary().create()
except ImportError:
    sys.exit(0)

# === 定数・ヘルパー ===
_PATH_CHARS = re.compile(r"[/\\.]")
_NOISE_WORDS = frozenset({
    "task", "notification", "output", "file", "status", "summary",
    "Background", "command", "tests", "exit", "Read", "retrieve",
    "result", "completed", "Users", "AppData", "Local", "Temp",
    "tasks", "Run", "mcp", "memory", "embodied", "claude", "code",
    "ClaudeCode",
})
_SYS_TAG = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

def _is_noise(surface: str) -> bool:
    if surface in _NOISE_WORDS:
        return True
    if _PATH_CHARS.search(surface):
        return True
    if re.fullmatch(r"[0-9a-fA-F\-]+", surface):
        return True
    return False

def _dedupe(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def extract_entry(text: str) -> dict | None:
    """テキストから名詞・動詞を抽出して1エントリを返す。"""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = _SYS_TAG.sub("", text)
    if len(text) < 2:
        return None
    try:
        tokens = tokenizer.tokenize(text)
    except Exception:
        return None
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
        return None
    entry = {}
    if words:
        entry["w"] = _dedupe(words)
    if verbs:
        entry["v"] = _dedupe(verbs)
    return entry if entry else None

# === トランスクリプトからアシスタントテキスト抽出 → 個別にバッファ追記 ===
buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
os.makedirs(os.path.dirname(buf_path), exist_ok=True)

written = 0
with open(buf_path, "a", encoding="utf-8") as buf_f:
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
            if obj.get("isSidechain"):
                continue

            message = obj.get("message", {})
            content = message.get("content", [])

            for c in content:
                if not isinstance(c, dict):
                    continue

                text = None
                # sayツールの発話テキスト
                if c.get("type") == "tool_use" and c.get("name") == "mcp__tts__say":
                    text = c.get("input", {}).get("text", "")
                # テキスト出力（thinkingは除外）
                elif c.get("type") == "text":
                    text = c.get("text", "").strip()

                if not text or len(text) < 2:
                    continue

                entry = extract_entry(text)
                if entry:
                    buf_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    written += 1
