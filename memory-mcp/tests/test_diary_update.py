"""Tests for diary update (strikethrough + amendment)."""

import pytest

from memory_mcp.memory import MemoryStore


class TestDiaryUpdate:
    """Tests for update_diary_content."""

    @pytest.mark.asyncio
    async def test_first_amendment_adds_strikethrough(self, memory_store: MemoryStore):
        """First amendment wraps original in ~~strikethrough~~."""
        memory = await memory_store.save(
            content="今日は晴れだった",
            emotion="1",
            importance=3,
        )

        updated = await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="実は曇りだった",
        )

        assert updated is not None
        assert updated.content.startswith("~~今日は晴れだった~~")
        assert "[追記" in updated.content
        assert "実は曇りだった" in updated.content

    @pytest.mark.asyncio
    async def test_embedding_recomputed(self, memory_store: MemoryStore):
        """Embedding should be recomputed after amendment."""
        memory = await memory_store.save(content="猫を見た")

        # Get original embedding
        db = memory_store._ensure_connected()
        orig_row = db.execute(
            "SELECT vector FROM embeddings WHERE memory_id = ?", (memory.id,)
        ).fetchone()
        orig_vector = orig_row[0]

        await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="犬も一緒にいた",
        )

        # Get updated embedding
        new_row = db.execute(
            "SELECT vector FROM embeddings WHERE memory_id = ?", (memory.id,)
        ).fetchone()
        new_vector = new_row[0]

        assert orig_vector != new_vector

    @pytest.mark.asyncio
    async def test_emotion_importance_update(self, memory_store: MemoryStore):
        """Emotion and importance can be updated."""
        memory = await memory_store.save(
            content="普通の出来事",
            emotion="8",
            importance=2,
        )

        updated = await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="実は大事だった",
            emotion="1",
            importance=5,
        )

        assert updated is not None
        assert updated.emotion == "1"
        assert updated.importance == 5

    @pytest.mark.asyncio
    async def test_emotion_importance_unchanged_when_not_specified(self, memory_store: MemoryStore):
        """Emotion and importance stay the same when not specified."""
        memory = await memory_store.save(
            content="テスト",
            emotion="3",
            importance=4,
        )

        updated = await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="追記テスト",
        )

        assert updated is not None
        assert updated.emotion == "3"
        assert updated.importance == 4

    @pytest.mark.asyncio
    async def test_nonexistent_id_returns_none(self, memory_store: MemoryStore):
        """Updating a nonexistent ID returns None."""
        result = await memory_store.update_diary_content(
            memory_id="nonexistent-id-12345",
            amendment="これは失敗するはず",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_amendments(self, memory_store: MemoryStore):
        """Multiple amendments append without re-wrapping strikethrough."""
        memory = await memory_store.save(content="最初の内容")

        # First amendment
        updated1 = await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="一回目の追記",
        )
        assert updated1 is not None
        assert updated1.content.startswith("~~最初の内容~~")
        assert "一回目の追記" in updated1.content

        # Second amendment
        updated2 = await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="二回目の追記",
        )
        assert updated2 is not None
        # Original strikethrough should appear exactly once
        assert updated2.content.count("~~最初の内容~~") == 1
        # Both amendments should be present
        assert "一回目の追記" in updated2.content
        assert "二回目の追記" in updated2.content
        # Should have two [追記] markers
        assert updated2.content.count("[追記") == 2

    @pytest.mark.asyncio
    async def test_get_by_id_returns_updated_content(self, memory_store: MemoryStore):
        """get_by_id should return the updated content after amendment."""
        memory = await memory_store.save(content="元の記憶")

        await memory_store.update_diary_content(
            memory_id=memory.id,
            amendment="修正された記憶",
        )

        fetched = await memory_store.get_by_id(memory.id)
        assert fetched is not None
        assert "~~元の記憶~~" in fetched.content
        assert "修正された記憶" in fetched.content
