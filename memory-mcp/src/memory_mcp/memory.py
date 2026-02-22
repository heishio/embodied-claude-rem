"""Memory operations (Phase 11: SQLite+numpy backend via store.py).

This module re-exports the MemoryStore and scoring helpers from store.py
for backward compatibility. All new code should import from store directly.
"""

from .store import (  # noqa: F401
    EMOTION_BOOST_MAP,
    MemoryStore,
    calculate_emotion_boost,
    calculate_final_score,
    calculate_importance_boost,
    calculate_time_decay,
)
