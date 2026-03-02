"""Pytest fixtures for Memory MCP tests."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import pytest_asyncio

from memory_mcp.chive import ChiVeEmbedding
from memory_mcp.config import MemoryConfig
from memory_mcp.memory import MemoryStore


class MockChiVeEmbedding(ChiVeEmbedding):
    """Mock chiVe embedding for tests: returns random but deterministic vectors."""

    def __init__(self, dim: int = 300):
        self._model_path = ""
        self._wv = None
        self._dim = dim
        self._cache: dict[str, np.ndarray] = {}

    def _load(self) -> None:
        pass

    def _ensure_loaded(self):
        return self

    @property
    def vector_size(self) -> int:
        return self._dim

    def get_word_vector(self, word: str) -> np.ndarray | None:
        if word not in self._cache:
            # Deterministic pseudo-random vector from word hash
            rng = np.random.RandomState(hash(word) % (2**31))
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-10
            self._cache[word] = vec
        return self._cache[word]


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Create a temporary database path."""
    return str(tmp_path / "test_memory.db")


@pytest.fixture
def mock_chive() -> MockChiVeEmbedding:
    """Create a mock chiVe embedding for testing."""
    return MockChiVeEmbedding()


@pytest.fixture
def memory_config(temp_db_path: str) -> MemoryConfig:
    """Create test memory config."""
    return MemoryConfig(
        db_path=temp_db_path,
        collection_name="test_memories",
        chive_model_path="/dummy/path",
    )


@pytest_asyncio.fixture
async def memory_store(memory_config: MemoryConfig, mock_chive: MockChiVeEmbedding) -> MemoryStore:
    """Create and connect a memory store with mock chiVe."""
    store = MemoryStore(memory_config, chive=mock_chive)
    await store.connect()
    yield store
    await store.disconnect()
