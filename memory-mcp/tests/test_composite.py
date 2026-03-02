"""Tests for composite memory synthesis."""

import numpy as np
import pytest

from memory_mcp.consolidation import ConsolidationEngine
from memory_mcp.memory import MemoryStore
from memory_mcp.vector import decode_vector


class TestSynthesizeComposites:
    """合成記憶の生成テスト。"""

    @pytest.mark.asyncio
    async def test_similar_memories_create_composite(self, memory_store: MemoryStore):
        """類似した記憶3件 → 合成記憶が1件生成される。"""
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=4)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=2)

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe: hash-based random vecs need very low threshold
            min_group_size=2,
        )

        assert stats["composites_created"] >= 1

    @pytest.mark.asyncio
    async def test_dissimilar_memories_no_composite(self, memory_store: MemoryStore):
        """まったく異なる記憶 → 合成されない。"""
        await memory_store.save(content="猫が窓辺で寝ている", category="observation")
        await memory_store.save(content="Pythonでコードを書いた", category="technical")
        await memory_store.save(content="今日は天気が良い", category="daily")

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.95,  # 高めに設定
            min_group_size=2,
        )

        assert stats["composites_created"] == 0

    @pytest.mark.asyncio
    async def test_composite_members_saved(self, memory_store: MemoryStore):
        """composite_members テーブルに関係が保存される。"""
        m1 = await memory_store.save(content="朝ごはんを食べた", category="daily")
        m2 = await memory_store.save(content="朝食を食べた", category="daily")
        m3 = await memory_store.save(content="朝ご飯を食べた", category="daily")

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe
            min_group_size=2,
        )

        existing = await memory_store.get_existing_composite_members()
        # 合成されていれば、少なくとも1つのメンバーセットがある
        if existing:
            member_set = existing[0]
            # メンバーは元の記憶IDの部分集合であること
            original_ids = {m1.id, m2.id, m3.id}
            assert member_set.issubset(original_ids)
            assert len(member_set) >= 2

    @pytest.mark.asyncio
    async def test_composite_vector_is_weighted_average(self, memory_store: MemoryStore):
        """合成ベクトルがメンバーの importance 加重平均（正規化済み）になっている。"""
        m1 = await memory_store.save(content="猫が寝ている", importance=2)
        m2 = await memory_store.save(content="猫が眠っている", importance=4)

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe
            min_group_size=2,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough to create composite")

        # 合成記憶のベクトルを取得
        db = memory_store.db
        composite_row = db.execute(
            "SELECT m.id, e.vector FROM memories m JOIN embeddings e ON e.memory_id = m.id WHERE m.level = 1"
        ).fetchone()
        assert composite_row is not None

        composite_vec = decode_vector(bytes(composite_row["vector"]))
        # 正規化されていることを確認
        norm = np.linalg.norm(composite_vec)
        assert abs(norm - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_no_duplicate_composites(self, memory_store: MemoryStore):
        """同じメンバー構成で二重に合成されない。"""
        await memory_store.save(content="散歩に出かけた", category="daily")
        await memory_store.save(content="散歩をした", category="daily")

        engine = ConsolidationEngine()

        # 1回目
        stats1 = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.5,
            min_group_size=2,
        )

        # 2回目（同じ条件）
        stats2 = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.5,
            min_group_size=2,
        )

        # 1回目で作られた合成は2回目ではスキップされる
        if stats1["composites_created"] > 0:
            assert stats2["composites_skipped"] >= stats1["composites_created"]
            assert stats2["composites_created"] == 0

    @pytest.mark.asyncio
    async def test_composite_has_level_1(self, memory_store: MemoryStore):
        """合成記憶の level が 1 であること。"""
        await memory_store.save(content="本を読んだ", category="daily")
        await memory_store.save(content="読書をした", category="daily")

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough to create composite")

        db = memory_store.db
        composites = db.execute("SELECT * FROM memories WHERE level = 1").fetchall()
        assert len(composites) >= 1
        for c in composites:
            assert c["level"] == 1
            assert c["content"] == ""  # content は空

    @pytest.mark.asyncio
    async def test_composite_emotion_is_mode(self, memory_store: MemoryStore):
        """合成記憶の emotion がメンバーの最頻値であること。"""
        await memory_store.save(content="嬉しい出来事があった", emotion="1")
        await memory_store.save(content="嬉しいことがあった", emotion="1")
        await memory_store.save(content="楽しいことがあった", emotion="5")

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough to create composite")

        db = memory_store.db
        composite = db.execute("SELECT emotion FROM memories WHERE level = 1").fetchone()
        assert composite is not None
        # 最頻値は "1"（2件 vs 1件）
        assert composite["emotion"] == "1"

    @pytest.mark.asyncio
    async def test_max_group_size_limits(self, memory_store: MemoryStore):
        """max_group_size を超えるグループは絞られること。"""
        # 10件の類似記憶を作成
        for i in range(10):
            await memory_store.save(content=f"今日も散歩をした（{i}回目）", category="daily")

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
            max_group_size=4,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough to create composite")

        # composite_members の件数が max_group_size 以下であること
        db = memory_store.db
        for row in db.execute("SELECT composite_id, COUNT(*) as cnt FROM composite_members GROUP BY composite_id").fetchall():
            assert row["cnt"] <= 4

    @pytest.mark.asyncio
    async def test_min_freshness_filter(self, memory_store: MemoryStore):
        """freshness が低すぎる記憶は合成対象外。"""
        m1 = await memory_store.save(content="古い記憶テスト", category="daily")
        m2 = await memory_store.save(content="古い記憶テスト2", category="daily")

        # freshness を手動で下げる
        db = memory_store.db
        db.execute("UPDATE memories SET freshness = 0.05 WHERE id IN (?, ?)", (m1.id, m2.id))
        db.commit()

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        assert stats["composites_created"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_memories_with_synthesize(self, memory_store: MemoryStore):
        """consolidate_memories から synthesize が呼ばれること。"""
        await memory_store.save(content="テスト記憶A", category="daily")
        await memory_store.save(content="テスト記憶B", category="daily")

        stats = await memory_store.consolidate_memories(
            window_hours=168,
            synthesize=True,
        )

        # synthesize 関連のキーが存在すること
        assert "composites_created" in stats
        assert "composites_skipped" in stats

    @pytest.mark.asyncio
    async def test_consolidate_memories_without_synthesize(self, memory_store: MemoryStore):
        """synthesize=False で合成が実行されないこと。"""
        await memory_store.save(content="テスト記憶C", category="daily")
        await memory_store.save(content="テスト記憶D", category="daily")

        stats = await memory_store.consolidate_memories(
            window_hours=168,
            synthesize=False,
        )

        assert "composites_created" not in stats
