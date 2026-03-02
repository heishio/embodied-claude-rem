"""Scoring functions for memory recall (time decay, emotion boost, importance)."""

import math
from datetime import datetime, timezone

EMOTION_BOOST_MAP: dict[str, float] = {
    "5": 0.4,
    "3": 0.35,
    "4": 0.3,
    "2": 0.25,
    "1": 0.2,
    "6": 0.15,
    "7": 0.1,
    "8": 0.0,
}


def calculate_time_decay(
    timestamp: str,
    now: datetime | None = None,
    half_life_days: float = 30.0,
) -> float:
    """
    時間減衰係数を計算。

    Args:
        timestamp: 記憶のタイムスタンプ（ISO 8601形式）
        now: 現在時刻（省略時は現在）
        half_life_days: 半減期（日数）

    Returns:
        0.0（完全に忘却）〜 1.0（新鮮な記憶）
    """
    if now is None:
        now = datetime.now(timezone.utc)

    try:
        memory_time = datetime.fromisoformat(timestamp)
        if memory_time.tzinfo is None:
            memory_time = memory_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return 1.0  # パースできない場合は減衰なし

    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    age_seconds = (now - memory_time).total_seconds()
    if age_seconds < 0:
        return 1.0  # 未来の記憶は減衰なし

    age_days = age_seconds / 86400
    # 指数減衰: decay = 2^(-age / half_life)
    decay = math.pow(2, -age_days / half_life_days)
    return max(0.0, min(1.0, decay))


def calculate_emotion_boost(emotion: str) -> float:
    """感情に基づくブースト値を返す。"""
    return EMOTION_BOOST_MAP.get(emotion, 0.0)


def calculate_importance_boost(importance: int) -> float:
    """
    重要度に基づくブースト。

    Args:
        importance: 1-5

    Returns:
        0.0 〜 0.4
    """
    clamped = max(1, min(5, importance))
    return (clamped - 1) / 10  # 1→0.0, 5→0.4


def calculate_final_score(
    semantic_distance: float,
    time_decay: float,
    emotion_boost: float,
    importance_boost: float,
    semantic_weight: float = 1.0,
    decay_weight: float = 0.3,
    emotion_weight: float = 0.0,
    importance_weight: float = 0.2,
) -> float:
    """
    最終スコアを計算。低いほど「良い」（想起されやすい）。

    Args:
        semantic_distance: ChromaDBからの距離（0〜2くらい）
        time_decay: 時間減衰係数（0.0〜1.0）
        emotion_boost: 感情ブースト
        importance_boost: 重要度ブースト

    Returns:
        最終スコア（低いほど良い）
    """
    # 時間減衰ペナルティ：新しい記憶ほど有利
    decay_penalty = (1.0 - time_decay) * decay_weight

    # ブーストは距離を減らす方向
    total_boost = emotion_boost * emotion_weight + importance_boost * importance_weight

    final = semantic_distance * semantic_weight + decay_penalty - total_boost
    return max(0.0, final)
