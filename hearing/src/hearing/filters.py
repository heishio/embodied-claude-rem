"""Post-filters for Whisper transcription output."""

from __future__ import annotations

import time

# Whisper hallucination blacklist (case-insensitive substring match)
HALLUCINATION_BLACKLIST = [
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "チャンネル登録お願いします",
    "チャンネル登録",
    "次回も",
    "次回の動画",
    "見てくれてありがとう",
    "おめでとうございます",
    "字幕",
    "字幕制作",
    "翻訳",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe",
    "subtitles by",
]

FILLER_WORDS = frozenset(
    "えー ええと えっと あの その うーん んー ま はい うん ん".split()
)


def _is_repetitive(text: str) -> bool:
    """Detect repetitive patterns like 'ん ん ん ん ん' or '4日 4日 4日'."""
    words = text.split()
    if len(words) < 3:
        return False
    unique = set(words)
    # 80%以上が同じ単語の繰り返し
    if len(unique) <= max(1, len(words) // 5):
        return True
    return False


def _is_only_punct_or_symbol(s: str) -> bool:
    return all(
        c in "。、！？…・「」『』（）()!?,. " or not c.isalnum()
        for c in s
    )


def should_skip(text: str) -> bool:
    """Return True if the transcription should be discarded."""
    text = text.strip()

    # Too short
    if len(text) < 2:
        return True

    # Punctuation / symbols only
    if _is_only_punct_or_symbol(text):
        return True

    # Filler words
    if text in FILLER_WORDS:
        return True

    # Hallucination blacklist (case-insensitive substring)
    text_lower = text.lower()
    for phrase in HALLUCINATION_BLACKLIST:
        if phrase.lower() in text_lower:
            return True

    # Repetitive patterns (e.g. "ん ん ん ん ん")
    if _is_repetitive(text):
        return True

    return False


class Debouncer:
    """Suppress duplicate transcriptions within a time window."""

    def __init__(self, window_sec: float = 1.5):
        self._last_text: str = ""
        self._last_time: float = 0.0
        self._window = window_sec
        self._repeat_count: int = 0
        self._repeat_threshold: int = 1  # 同一テキスト連続2回目から抑制

    def is_duplicate(self, text: str) -> bool:
        now = time.time()
        # 短い窓での完全一致（従来）
        if text == self._last_text and now - self._last_time < self._window:
            return True
        # 連続同一テキスト検出（窓を超えても同じテキストが繰り返される場合）
        if text == self._last_text:
            self._repeat_count += 1
            if self._repeat_count >= self._repeat_threshold:
                self._last_time = now
                return True
        else:
            self._repeat_count = 0
        self._last_text = text
        self._last_time = now
        return False
