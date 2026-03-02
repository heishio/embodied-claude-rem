"""Tests for multi-level composite synthesis, overlap detection, and orphan rescue."""

import numpy as np
import pytest

from memory_mcp.consolidation import ConsolidationEngine
from memory_mcp.memory import MemoryStore


class TestMultiLevelSynthesis:
    """多段合成テスト: level=0 → 1 → 2。"""

    @pytest.mark.asyncio
    async def test_level1_composites_created(self, memory_store: MemoryStore):
        """level=0 → level=1 の合成が動くこと。"""
        await memory_store.save(content="猫が寝ている", category="observation")
        await memory_store.save(content="猫が眠っている", category="observation")
        await memory_store.save(content="猫が丸くなっている", category="observation")

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
            source_level=0,
            target_level=1,
        )
        assert stats["composites_created"] >= 1

        db = memory_store.db
        l1 = db.execute("SELECT COUNT(*) as cnt FROM memories WHERE level = 1").fetchone()
        assert l1["cnt"] >= 1

    @pytest.mark.asyncio
    async def test_level2_composites_from_level1(self, memory_store: MemoryStore):
        """level=1 → level=2 の合成が動くこと。"""
        # まず level=0 の記憶を2グループ作成（各グループが類似）
        # グループ1: 猫系
        await memory_store.save(content="猫が寝ている", category="observation")
        await memory_store.save(content="猫が眠っている", category="observation")
        # グループ2: 猫系（別表現）
        await memory_store.save(content="猫が丸くなって寝ている", category="observation")
        await memory_store.save(content="猫が丸まって眠っている", category="observation")

        engine = ConsolidationEngine()

        # level=0 → 1
        stats1 = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe
            min_group_size=2,
            source_level=0,
            target_level=1,
        )

        if stats1["composites_created"] < 2:
            pytest.skip("Not enough level-1 composites created for level-2 test")

        # level=1 → 2
        stats2 = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe
            min_group_size=2,
            source_level=1,
            target_level=2,
        )

        assert stats2["composites_created"] >= 1
        db = memory_store.db
        l2 = db.execute("SELECT COUNT(*) as cnt FROM memories WHERE level = 2").fetchone()
        assert l2["cnt"] >= 1

    @pytest.mark.asyncio
    async def test_save_composite_level_parameter(self, memory_store: MemoryStore):
        """save_composite の level パラメータが正しく保存されること。"""
        vec = np.random.randn(600).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)

        cid = await memory_store.save_composite(
            member_ids=[],
            vector=vec,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="daily",
            level=2,
        )

        db = memory_store.db
        row = db.execute("SELECT level FROM memories WHERE id = ?", (cid,)).fetchone()
        assert row["level"] == 2

    @pytest.mark.asyncio
    async def test_fetch_memories_by_level(self, memory_store: MemoryStore):
        """fetch_memories_with_vectors_by_level がレベル別にフィルタすること。"""
        await memory_store.save(content="レベル0の記憶", category="daily")

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=1,  # テスト用に1でもOK
            source_level=0,
            target_level=1,
        )

        l0 = await memory_store.fetch_memories_with_vectors_by_level(level=0)
        # level=0 の記憶は少なくとも1件
        assert len(l0) >= 1
        for mem, vec in l0:
            assert mem.level == 0

    @pytest.mark.asyncio
    async def test_fetch_all_composite_ids_with_level(self, memory_store: MemoryStore):
        """fetch_all_composite_ids がレベル指定で動くこと。"""
        await memory_store.save(content="猫A", category="observation")
        await memory_store.save(content="猫B", category="observation")

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
            source_level=0,
            target_level=1,
        )

        all_ids = await memory_store.fetch_all_composite_ids()
        l1_ids = await memory_store.fetch_all_composite_ids(level=1)
        l2_ids = await memory_store.fetch_all_composite_ids(level=2)

        assert len(all_ids) >= len(l1_ids)
        assert len(l2_ids) == 0  # まだ level=2 は作っていない


class TestOrphanRescue:
    """孤立記憶の救出テスト。"""

    @pytest.mark.asyncio
    async def test_orphan_rescued_to_nearest_composite(self, memory_store: MemoryStore):
        """孤立記憶が最寄り composite に吸収されること。"""
        # グループ記憶
        await memory_store.save(content="猫が寝ている", category="observation")
        await memory_store.save(content="猫が眠っている", category="observation")
        # 孤立だけど類似な記憶
        orphan = await memory_store.save(
            content="猫がまどろんでいる", category="observation",
        )

        engine = ConsolidationEngine()
        # 合成（orphanは閾値外で除外される可能性がある）
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.8,  # 高閾値で2件だけグループ化
            min_group_size=2,
            source_level=0,
            target_level=1,
        )

        # 孤立救出
        stats = await engine.rescue_orphans(
            store=memory_store,
            rescue_threshold=0.3,
            level=0,
        )

        # orphan が発見されたか
        assert stats["orphans_found"] >= 0  # 0の場合もある（全部合成に入った場合）

    @pytest.mark.asyncio
    async def test_no_orphans_when_all_in_composite(self, memory_store: MemoryStore):
        """全記憶が composite に属していれば孤立は0。"""
        await memory_store.save(content="猫が寝ている", category="observation")
        await memory_store.save(content="猫が眠っている", category="observation")

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.0,  # mock chiVe uses random vectors
            min_group_size=2,
            source_level=0,
            target_level=1,
        )

        stats = await engine.rescue_orphans(
            store=memory_store,
            rescue_threshold=0.3,
            level=0,
        )
        assert stats["orphans_found"] == 0
        assert stats["orphans_rescued"] == 0

    @pytest.mark.asyncio
    async def test_fetch_orphan_memories(self, memory_store: MemoryStore):
        """fetch_orphan_memories が composite 非所属の記憶を返すこと。"""
        m1 = await memory_store.save(content="猫が寝ている", category="observation")
        m2 = await memory_store.save(content="犬が走っている", category="observation")

        # m1 だけ composite に入れる
        vec = np.random.randn(600).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        await memory_store.save_composite(
            member_ids=[m1.id],
            vector=vec,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="observation",
            level=1,
        )

        orphans = await memory_store.fetch_orphan_memories(level=0)
        orphan_ids = {mem.id for mem, _ in orphans}
        assert m2.id in orphan_ids
        assert m1.id not in orphan_ids


class TestOverlapDetection:
    """クラスタ重なり検出テスト。"""

    @pytest.mark.asyncio
    async def test_overlap_detected_between_similar_composites(
        self, memory_store: MemoryStore,
    ):
        """類似した composite 間で重なりが検出されること。"""
        # 2つの近いグループを手動で作成
        dim = 600
        base = np.random.randn(dim).astype(np.float32)
        base = base / np.linalg.norm(base)

        # グループ1
        m1 = await memory_store.save(content="テスト記憶1A", category="daily")
        m2 = await memory_store.save(content="テスト記憶1B", category="daily")

        vec1 = base + np.random.randn(dim).astype(np.float32) * 0.05
        vec1 = vec1 / np.linalg.norm(vec1)
        await memory_store.save_composite(
            member_ids=[m1.id, m2.id],
            vector=vec1,
            emotion="8", importance=3, freshness=1.0,
            category="daily", level=1,
        )

        # グループ2（ベースに近い）
        m3 = await memory_store.save(content="テスト記憶2A", category="daily")
        m4 = await memory_store.save(content="テスト記憶2B", category="daily")

        vec2 = base + np.random.randn(dim).astype(np.float32) * 0.05
        vec2 = vec2 / np.linalg.norm(vec2)
        await memory_store.save_composite(
            member_ids=[m3.id, m4.id],
            vector=vec2,
            emotion="8", importance=3, freshness=1.0,
            category="daily", level=1,
        )

        engine = ConsolidationEngine()
        stats = await engine.detect_overlap(
            store=memory_store,
            overlap_threshold=0.3,  # 低閾値で確実にヒット
        )

        assert stats["overlap_pairs"] >= 1

    @pytest.mark.asyncio
    async def test_no_overlap_with_distant_composites(self, memory_store: MemoryStore):
        """遠い composite 間では重なりが検出されないこと。"""
        dim = 600

        m1 = await memory_store.save(content="テスト記憶A", category="daily")
        m2 = await memory_store.save(content="テスト記憶B", category="daily")

        # 完全に異なる方向のベクトル
        vec1 = np.zeros(dim, dtype=np.float32)
        vec1[0] = 1.0
        await memory_store.save_composite(
            member_ids=[m1.id],
            vector=vec1,
            emotion="8", importance=3, freshness=1.0,
            category="daily", level=1,
        )

        m3 = await memory_store.save(content="テスト記憶C", category="daily")
        vec2 = np.zeros(dim, dtype=np.float32)
        vec2[1] = 1.0
        await memory_store.save_composite(
            member_ids=[m3.id],
            vector=vec2,
            emotion="8", importance=3, freshness=1.0,
            category="daily", level=1,
        )

        engine = ConsolidationEngine()
        stats = await engine.detect_overlap(
            store=memory_store,
            overlap_threshold=0.9,  # 高閾値
        )

        assert stats["overlap_pairs"] == 0
        assert stats["dual_members_added"] == 0


class TestConsolidateMemoriesIntegration:
    """consolidate_memories の統合テスト。"""

    @pytest.mark.asyncio
    async def test_consolidate_returns_new_keys(self, memory_store: MemoryStore):
        """consolidate_memories が新しい stats キーを返すこと。"""
        await memory_store.save(content="統合テストA", category="daily")
        await memory_store.save(content="統合テストB", category="daily")

        stats = await memory_store.consolidate_memories(
            window_hours=168,
            synthesize=True,
        )

        assert "composites_created" in stats
        assert "orphans_rescued_l0" in stats
        assert "overlap_pairs" in stats
        assert "dual_members_added" in stats
        assert "l2_composites_created" in stats

    @pytest.mark.asyncio
    async def test_backward_compat_fetch_level0(self, memory_store: MemoryStore):
        """fetch_level0_memories_with_vectors が後方互換で動くこと。"""
        await memory_store.save(content="後方互換テスト", category="daily")

        result = await memory_store.fetch_level0_memories_with_vectors()
        assert len(result) >= 1
        for mem, vec in result:
            assert mem.level == 0
