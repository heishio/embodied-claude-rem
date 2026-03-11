"""Tests for visual graph edge strengthening (Phase 2 Step 2+3)."""

import numpy as np
import pytest

from memory_mcp.consolidation import ConsolidationEngine
from memory_mcp.graph import MemoryGraph
from memory_mcp.memory import MemoryStore
from memory_mcp.vector import encode_vector


class TestStrengthenVisualGraphEdges:
    """tag付きimage_compositesからグラフのvnエッジを強化するテスト。"""

    def _insert_image_composite(self, store: MemoryStore, tag: str | None, member_count: int) -> str:
        """テスト用にimage_compositeを直接DBに挿入。"""
        import uuid
        from datetime import datetime, timezone

        db = store._ensure_connected()
        cid = f"img-{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        dummy_vec = np.random.randn(512).astype(np.float32)
        db.execute(
            """INSERT INTO image_composites
               (id, delta_centroid, member_count, freshness, tag, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (cid, encode_vector(dummy_vec), member_count, 1.0, tag, now, now),
        )
        db.commit()
        return cid

    @pytest.mark.asyncio
    async def test_tagged_composites_create_graph_edges(self, memory_store: MemoryStore):
        """tag付きcompositeが「見る → {tag}」のvnエッジを生成する。"""
        self._insert_image_composite(memory_store, tag="シオ", member_count=5)
        self._insert_image_composite(memory_store, tag="猫", member_count=3)

        graph = MemoryGraph(memory_store.db)
        engine = ConsolidationEngine()
        stats = await engine.strengthen_visual_graph_edges(store=memory_store, graph=graph)

        assert stats["visual_edges_strengthened"] == 2

        # グラフにノードが存在するか
        db = memory_store._ensure_connected()
        verb_node = db.execute(
            "SELECT id FROM graph_nodes WHERE type='verb' AND surface_form='見る'"
        ).fetchone()
        assert verb_node is not None

        noun_shio = db.execute(
            "SELECT id FROM graph_nodes WHERE type='noun' AND surface_form='シオ'"
        ).fetchone()
        assert noun_shio is not None

        noun_neko = db.execute(
            "SELECT id FROM graph_nodes WHERE type='noun' AND surface_form='猫'"
        ).fetchone()
        assert noun_neko is not None

        # vnエッジが存在するか（graph_edgesのカラム名を確認）
        edge_shio = db.execute(
            "SELECT weight FROM graph_edges WHERE from_id=? AND to_id=? AND link_type='vn'",
            (verb_node["id"], noun_shio["id"]),
        ).fetchone()
        assert edge_shio is not None
        assert edge_shio["weight"] > 0

        edge_neko = db.execute(
            "SELECT weight FROM graph_edges WHERE from_id=? AND to_id=? AND link_type='vn'",
            (verb_node["id"], noun_neko["id"]),
        ).fetchone()
        assert edge_neko is not None
        assert edge_neko["weight"] > 0

    @pytest.mark.asyncio
    async def test_untagged_composites_skipped(self, memory_store: MemoryStore):
        """tagなしcompositeはスキップされる。"""
        self._insert_image_composite(memory_store, tag=None, member_count=5)

        graph = MemoryGraph(memory_store.db)
        engine = ConsolidationEngine()
        stats = await engine.strengthen_visual_graph_edges(store=memory_store, graph=graph)

        assert stats["visual_edges_strengthened"] == 0

    @pytest.mark.asyncio
    async def test_member_count_affects_weight(self, memory_store: MemoryStore):
        """member_countが多いほどvnエッジの重みが大きい。"""
        self._insert_image_composite(memory_store, tag="シオ", member_count=10)
        self._insert_image_composite(memory_store, tag="猫", member_count=2)

        graph = MemoryGraph(memory_store.db)
        engine = ConsolidationEngine()
        await engine.strengthen_visual_graph_edges(store=memory_store, graph=graph)

        db = memory_store._ensure_connected()
        verb_id = db.execute(
            "SELECT id FROM graph_nodes WHERE type='verb' AND surface_form='見る'"
        ).fetchone()["id"]
        shio_id = db.execute(
            "SELECT id FROM graph_nodes WHERE type='noun' AND surface_form='シオ'"
        ).fetchone()["id"]
        neko_id = db.execute(
            "SELECT id FROM graph_nodes WHERE type='noun' AND surface_form='猫'"
        ).fetchone()["id"]

        w_shio = db.execute(
            "SELECT weight FROM graph_edges WHERE from_id=? AND to_id=? AND link_type='vn'",
            (verb_id, shio_id),
        ).fetchone()["weight"]
        w_neko = db.execute(
            "SELECT weight FROM graph_edges WHERE from_id=? AND to_id=? AND link_type='vn'",
            (verb_id, neko_id),
        ).fetchone()["weight"]

        # シオ(10) > 猫(2) なのでシオの方が重みが大きい
        assert w_shio > w_neko

    @pytest.mark.asyncio
    async def test_low_freshness_composites_excluded(self, memory_store: MemoryStore):
        """freshnessが低いcompositeは除外される。"""
        import uuid
        from datetime import datetime, timezone

        db = memory_store._ensure_connected()
        cid = f"img-{uuid.uuid4()}"
        now = datetime.now(timezone.utc).isoformat()
        dummy_vec = np.random.randn(512).astype(np.float32)
        db.execute(
            """INSERT INTO image_composites
               (id, delta_centroid, member_count, freshness, tag, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (cid, encode_vector(dummy_vec), 5, 0.05, "シオ", now, now),
        )
        db.commit()

        graph = MemoryGraph(memory_store.db)
        engine = ConsolidationEngine()
        stats = await engine.strengthen_visual_graph_edges(store=memory_store, graph=graph)

        assert stats["visual_edges_strengthened"] == 0
