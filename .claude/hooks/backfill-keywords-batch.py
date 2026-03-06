#!/usr/bin/env python3
"""backfill-keywords-batch.py — トランスクリプトからキーワードを一括抽出

stdin から ccconv の talk 形式（1行=1発話）を読み、
名詞・動詞を抽出して sensory_buffer.jsonl に追記する。
keyword-buffer.py のバッチ版。1プロセスで全行を処理する。
"""
import json
import os
import re
import sys

# ── sudachipy 初期化（1回だけ）──────────────────────
try:
    from sudachipy import Dictionary
    tokenizer = Dictionary().create()
except ImportError:
    print("sudachipy not found", file=sys.stderr)
    sys.exit(1)

# ── ノイズフィルタ（keyword-buffer.py と同じ）──────────
_PATH_CHARS = re.compile(r"[/\\.]")
_NOISE_WORDS = frozenset({
    "task", "notification", "output", "file", "status", "summary",
    "Background", "command", "tests", "exit", "Read", "retrieve",
    "result", "completed", "Users", "AppData", "Local", "Temp",
    "tasks", "Run", "mcp", "memory", "embodied", "claude", "code",
    "ClaudeCode",
})
_SYSTEM_TAG = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)
# talk 形式の話者プレフィックス (例: "Human: ", "Assistant: ")
_SPEAKER_PREFIX = re.compile(r"^(Human|Assistant|System):\s*")


def _is_noise(surface: str) -> bool:
    if surface in _NOISE_WORDS:
        return True
    if _PATH_CHARS.search(surface):
        return True
    if re.fullmatch(r"[0-9a-fA-F\-]+", surface):
        return True
    return False


def dedupe(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def extract_keywords(text: str) -> dict | None:
    """テキストから名詞・動詞を抽出して dict を返す。空なら None。"""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    text = _SYSTEM_TAG.sub("", text)
    text = _SPEAKER_PREFIX.sub("", text)

    if not text or len(text) < 2:
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

    unique_words = dedupe(words)
    unique_verbs = dedupe(verbs)

    entry = {}
    if unique_words:
        entry["w"] = unique_words
    if unique_verbs:
        entry["v"] = unique_verbs

    return entry if entry else None


# ── メイン処理 ──────────────────────────────────
buf_path = os.path.join(os.path.expanduser("~"), ".claude", "sensory_buffer.jsonl")
os.makedirs(os.path.dirname(buf_path), exist_ok=True)

count = 0
skipped = 0

with open(buf_path, "a", encoding="utf-8") as f:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        entry = extract_keywords(line)
        if entry:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1
        else:
            skipped += 1

        # 進捗表示（500行ごと）
        total = count + skipped
        if total % 500 == 0:
            print(f"  processed {total} lines ({count} entries)...", flush=True)

print(f"Done: {count} entries extracted, {skipped} lines skipped.")
