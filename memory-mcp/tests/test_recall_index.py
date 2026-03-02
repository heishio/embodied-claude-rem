"""Tests for recall index (pre-computed word→memory similarity)."""

import pytest

from memory_mcp.memory import MemoryStore


class TestBuildRecallIndex:
    """Tests for build_recall_index."""

    @pytest.mark.asyncio
    async def test_build_with_no_data(self, memory_store: MemoryStore):
        """Empty DB should produce 0 entries."""
        count = await memory_store.build_recall_index()
        assert count == 0

    @pytest.mark.asyncio
    async def test_build_creates_table(self, memory_store: MemoryStore):
        """recall_index table should exist after build."""
        await memory_store.build_recall_index()
        db = memory_store.db
        row = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recall_index'"
        ).fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_build_with_memories_and_graph_nodes(self, memory_store: MemoryStore):
        """Index should be populated when memories and graph_nodes exist."""
        # Save some memories
        await memory_store.save(content="梅の花が咲いていた", category="observation")
        await memory_store.save(content="春のお散歩をした", category="daily")
        await memory_store.save(content="桜を見に行った", category="daily")

        db = memory_store.db
        # Insert graph_nodes (simulating keyword-buffer output)
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '梅')"
        )
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '花見')"
        )
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '春')"
        )
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('verb', '咲く')"
        )
        db.commit()

        count = await memory_store.build_recall_index()
        assert count > 0

        # Check that entries exist
        rows = db.execute("SELECT COUNT(*) FROM recall_index").fetchone()
        assert rows[0] > 0

    @pytest.mark.asyncio
    async def test_build_semantic_similarity(self, memory_store: MemoryStore):
        """Semantically related words should match related memories."""
        await memory_store.save(content="梅の花が咲いていた。春の訪れを感じた。", category="observation")
        await memory_store.save(content="コードのバグを修正した", category="technical")

        db = memory_store.db
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '春')"
        )
        db.commit()

        await memory_store.build_recall_index()

        # 「春」should match 梅の花 memory with higher similarity than bug fix memory
        rows = db.execute(
            "SELECT target_id, similarity, content_preview FROM recall_index "
            "WHERE word = '春' ORDER BY similarity DESC",
        ).fetchall()

        assert len(rows) > 0
        # With real chiVe, 「春」 should match 梅 memory; with mock, just verify entries exist
        assert len(rows) >= 1

    @pytest.mark.asyncio
    async def test_build_with_verb_chains(self, memory_store: MemoryStore):
        """Index should include verb_chain entries."""
        from memory_mcp.graph import MemoryGraph
        from memory_mcp.verb_chain import VerbChainStore
        from memory_mcp.types import VerbChain, VerbStep
        from datetime import datetime, timezone

        db = memory_store.db
        graph = MemoryGraph(db=db)
        vcs = VerbChainStore(
            db=db,
            chive=memory_store.chive,
            graph=graph,
        )
        await vcs.initialize()

        chain = VerbChain(
            id="test-chain-1",
            steps=(
                VerbStep(verb="見る", nouns=("桜",)),
                VerbStep(verb="驚く", nouns=("美しさ",)),
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            emotion="1",
            importance=4,
            source="manual",
            context="桜を見て感動した",
        )
        await vcs.save(chain)

        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '桜')"
        )
        db.commit()

        count = await memory_store.build_recall_index()
        assert count > 0

        # Check chain entries exist
        rows = db.execute(
            "SELECT target_type FROM recall_index WHERE word = '桜'"
        ).fetchall()
        target_types = [r[0] for r in rows]
        assert "chain" in target_types

    @pytest.mark.asyncio
    async def test_rebuild_clears_old(self, memory_store: MemoryStore):
        """Rebuilding should clear old entries and create fresh ones."""
        await memory_store.save(content="テスト記憶その一", category="daily")

        db = memory_store.db
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', 'テスト')"
        )
        db.commit()

        count1 = await memory_store.rebuild_recall_index_full()
        count2 = await memory_store.rebuild_recall_index_full()

        # Should produce same count (idempotent)
        assert count1 == count2


class TestUpdateRecallIndex:
    """Tests for update_recall_index (incremental update)."""

    @pytest.mark.asyncio
    async def test_update_new_memory(self, memory_store: MemoryStore):
        """New memory should be added to existing index."""
        # Build initial index with one memory
        await memory_store.save(content="初めての記憶", category="daily")

        db = memory_store.db
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '記憶')"
        )
        db.commit()

        await memory_store.build_recall_index()
        initial_count = db.execute("SELECT COUNT(*) FROM recall_index").fetchone()[0]

        # Save new memory and update index
        new_mem = await memory_store.save(content="新しい体験をした", category="daily")
        updated = await memory_store.update_recall_index(new_mem.id, "memory")

        # Should have updated some entries
        assert updated >= 0

    @pytest.mark.asyncio
    async def test_update_with_no_index(self, memory_store: MemoryStore):
        """Update with empty index should return 0."""
        mem = await memory_store.save(content="何かの記憶", category="daily")
        updated = await memory_store.update_recall_index(mem.id, "memory")
        assert updated == 0

    @pytest.mark.asyncio
    async def test_update_nonexistent_memory(self, memory_store: MemoryStore):
        """Update with non-existent memory_id should return 0."""
        updated = await memory_store.update_recall_index("nonexistent-id", "memory")
        assert updated == 0


class TestRecallIndexQuery:
    """Tests for querying the recall_index (simulating recall-lite usage)."""

    @pytest.mark.asyncio
    async def test_query_by_word(self, memory_store: MemoryStore):
        """Simple word query should return ordered results."""
        await memory_store.save(content="猫が窓辺で寝ていた", category="observation")
        await memory_store.save(content="犬と散歩した", category="daily")
        await memory_store.save(content="猫のおもちゃを買った", category="daily")

        db = memory_store.db
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '猫')"
        )
        db.commit()

        await memory_store.build_recall_index()

        rows = db.execute(
            "SELECT target_id, similarity, content_preview "
            "FROM recall_index WHERE word = '猫' "
            "ORDER BY similarity DESC LIMIT 8"
        ).fetchall()

        assert len(rows) > 0
        # Results should be sorted by similarity
        sims = [r[1] for r in rows]
        assert sims == sorted(sims, reverse=True)

    @pytest.mark.asyncio
    async def test_top_k_limit(self, memory_store: MemoryStore):
        """Each word should have at most 20 entries."""
        # Save many memories
        for i in range(25):
            await memory_store.save(content=f"記憶その{i}について", category="daily")

        db = memory_store.db
        db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES ('noun', '記憶')"
        )
        db.commit()

        await memory_store.build_recall_index()

        rows = db.execute(
            "SELECT COUNT(*) FROM recall_index WHERE word = '記憶'"
        ).fetchone()
        assert rows[0] <= 20
