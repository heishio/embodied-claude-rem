"""Tests for principal axis (PCA), intersection detection, and anisotropic edge classification."""

import numpy as np
import pytest

from memory_mcp.consolidation import ConsolidationEngine
from memory_mcp.memory import MemoryStore
from memory_mcp.vector import encode_vector


class TestPrincipalAxis:
    """PCA計算のユニットテスト。"""

    def test_compute_principal_axis_basic(self):
        """2つ以上のベクトルで主成分が計算できる。"""
        engine = ConsolidationEngine()
        # 意図的に1方向に伸びたベクトル群を作成
        rng = np.random.default_rng(42)
        base = rng.normal(size=384).astype(np.float32)
        base /= np.linalg.norm(base)

        # base方向に大きく散らばるベクトル
        vecs = []
        for i in range(5):
            noise = rng.normal(size=384).astype(np.float32) * 0.01
            offset = base * (i - 2) * 0.1
            v = base + offset + noise
            v /= np.linalg.norm(v)
            vecs.append(v)

        member_vecs = np.array(vecs)
        result = engine._compute_principal_axis(member_vecs)

        assert result is not None
        axis, explained = result
        assert axis.shape == (384,)
        assert 0.0 < explained <= 1.0
        # 軸はほぼ正規化されている
        assert abs(np.linalg.norm(axis) - 1.0) < 1e-5

    def test_compute_principal_axis_single_member(self):
        """メンバーが1つの場合はNoneを返す。"""
        engine = ConsolidationEngine()
        vec = np.random.randn(384).astype(np.float32)
        result = engine._compute_principal_axis(vec.reshape(1, -1))
        assert result is None

    def test_compute_principal_axis_explained_variance(self):
        """1方向に強く伸びたデータでは寄与率が高い。"""
        engine = ConsolidationEngine()
        rng = np.random.default_rng(123)
        direction = rng.normal(size=384).astype(np.float32)
        direction /= np.linalg.norm(direction)

        vecs = []
        for i in range(10):
            # 主方向に大きなバリアンス、その他は微小
            v = direction * (i * 0.5) + rng.normal(size=384).astype(np.float32) * 0.001
            vecs.append(v)

        result = engine._compute_principal_axis(np.array(vecs))
        assert result is not None
        _, explained = result
        # ほぼ1方向なので寄与率が高いはず
        assert explained > 0.5


class TestAnisotropicEdgeClassification:
    """異方的エッジ判定のユニットテスト。"""

    def test_isotropic_fallback(self):
        """axis_vector=Noneの場合、等方的分類にフォールバック。"""
        engine = ConsolidationEngine()
        rng = np.random.default_rng(42)
        centroid = rng.normal(size=384).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        member_vecs = np.array([
            centroid + rng.normal(size=384).astype(np.float32) * 0.01,
            centroid + rng.normal(size=384).astype(np.float32) * 0.05,
            centroid + rng.normal(size=384).astype(np.float32) * 0.1,
        ])
        for i in range(len(member_vecs)):
            member_vecs[i] /= np.linalg.norm(member_vecs[i])

        result = engine._classify_edge_core(member_vecs, centroid, axis_vector=None)
        assert len(result) == 3
        assert all(r in (0, 1) for r in result)

    def test_anisotropic_axis_direction_tolerant(self):
        """軸方向に離れたメンバーは coreに留まりやすい。"""
        engine = ConsolidationEngine()
        rng = np.random.default_rng(42)
        dim = 384

        # 軸方向
        axis = np.zeros(dim, dtype=np.float32)
        axis[0] = 1.0

        centroid = rng.normal(size=dim).astype(np.float32)
        centroid /= np.linalg.norm(centroid)

        # メンバー: centroidから軸方向に大きくずれたもの（core寄りのはず）
        axial_member = centroid + axis * 0.15
        axial_member /= np.linalg.norm(axial_member)

        # メンバー: centroidから垂直方向に大きくずれたもの（edge寄りのはず）
        perp = np.zeros(dim, dtype=np.float32)
        perp[1] = 1.0
        perp_member = centroid + perp * 0.15
        perp_member /= np.linalg.norm(perp_member)

        # メンバー: centroidに近いもの（core）
        close_member = centroid + rng.normal(size=dim).astype(np.float32) * 0.001
        close_member /= np.linalg.norm(close_member)

        member_vecs = np.array([axial_member, perp_member, close_member])

        result_aniso = engine._classify_edge_core(member_vecs, centroid, axis_vector=axis)
        result_iso = engine._classify_edge_core(member_vecs, centroid, axis_vector=None)

        # 両方とも有効な分類
        assert len(result_aniso) == 3
        assert len(result_iso) == 3
        assert all(r in (0, 1) for r in result_aniso)


class TestIntersectionDetection:
    """交差検出の統合テスト。"""

    @pytest.mark.asyncio
    async def test_detect_intersections_no_composites(self, memory_store: MemoryStore):
        """compositeがない場合は空の結果。"""
        engine = ConsolidationEngine()
        stats = await engine.detect_intersections(store=memory_store)
        assert stats["parallel_found"] == 0
        assert stats["transversal_found"] == 0
        assert stats["intersection_nodes"] == 0

    @pytest.mark.asyncio
    async def test_detect_intersections_single_composite(self, memory_store: MemoryStore):
        """composite が1つだけの場合は交差なし。"""
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        stats = await engine.detect_intersections(store=memory_store)
        # 1つだけなのでペアなし
        assert stats["parallel_found"] == 0
        assert stats["transversal_found"] == 0

    @pytest.mark.asyncio
    async def test_schema_created(self, memory_store: MemoryStore):
        """composite_axes と composite_intersections テーブルが存在する。"""
        db = memory_store.db
        tables = {r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "composite_axes" in tables
        assert "composite_intersections" in tables

    @pytest.mark.asyncio
    async def test_save_composite_with_axis(self, memory_store: MemoryStore):
        """save_composite で axis_vector が保存される。"""
        rng = np.random.default_rng(42)
        vector = rng.normal(size=384).astype(np.float32)
        vector /= np.linalg.norm(vector)
        axis = rng.normal(size=384).astype(np.float32)
        axis /= np.linalg.norm(axis)

        # ダミーの level=0 記憶を作成
        m1 = await memory_store.save(content="テスト1", importance=3)
        m2 = await memory_store.save(content="テスト2", importance=3)

        cid = await memory_store.save_composite(
            member_ids=[m1.id, m2.id],
            vector=vector,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="daily",
            axis_vector=axis,
            explained_variance_ratio=0.85,
        )

        # composite_axes に保存されている
        db = memory_store.db
        row = db.execute(
            "SELECT * FROM composite_axes WHERE composite_id = ?", (cid,)
        ).fetchone()
        assert row is not None
        assert abs(row["explained_variance_ratio"] - 0.85) < 0.01

    @pytest.mark.asyncio
    async def test_save_composite_without_axis(self, memory_store: MemoryStore):
        """axis_vector=None の場合は composite_axes に保存されない。"""
        rng = np.random.default_rng(42)
        vector = rng.normal(size=384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        m1 = await memory_store.save(content="テスト3", importance=3)
        m2 = await memory_store.save(content="テスト4", importance=3)

        cid = await memory_store.save_composite(
            member_ids=[m1.id, m2.id],
            vector=vector,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="daily",
        )

        db = memory_store.db
        row = db.execute(
            "SELECT * FROM composite_axes WHERE composite_id = ?", (cid,)
        ).fetchone()
        assert row is None

    @pytest.mark.asyncio
    async def test_fetch_all_composite_axes(self, memory_store: MemoryStore):
        """fetch_all_composite_axes が正しく返す。"""
        rng = np.random.default_rng(42)
        vector = rng.normal(size=384).astype(np.float32)
        vector /= np.linalg.norm(vector)
        axis = rng.normal(size=384).astype(np.float32)
        axis /= np.linalg.norm(axis)

        m1 = await memory_store.save(content="テスト5", importance=3)
        m2 = await memory_store.save(content="テスト6", importance=3)

        cid = await memory_store.save_composite(
            member_ids=[m1.id, m2.id],
            vector=vector,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="daily",
            axis_vector=axis,
            explained_variance_ratio=0.7,
        )

        axes = await memory_store.fetch_all_composite_axes()
        assert cid in axes
        assert axes[cid].shape == (384,)

    @pytest.mark.asyncio
    async def test_fetch_all_composite_member_sets(self, memory_store: MemoryStore):
        """fetch_all_composite_member_sets が正しく返す。"""
        rng = np.random.default_rng(42)
        vector = rng.normal(size=384).astype(np.float32)
        vector /= np.linalg.norm(vector)

        m1 = await memory_store.save(content="テスト7", importance=3)
        m2 = await memory_store.save(content="テスト8", importance=3)

        cid = await memory_store.save_composite(
            member_ids=[m1.id, m2.id],
            vector=vector,
            emotion="8",
            importance=3,
            freshness=1.0,
            category="daily",
        )

        member_sets = await memory_store.fetch_all_composite_member_sets()
        assert cid in member_sets
        assert m1.id in member_sets[cid]
        assert m2.id in member_sets[cid]

    @pytest.mark.asyncio
    async def test_get_intersection_nodes_empty(self, memory_store: MemoryStore):
        """交差がない場合は空dict。"""
        scores = await memory_store.get_intersection_nodes(["some-id"])
        assert scores == {}

    @pytest.mark.asyncio
    async def test_save_and_get_intersections(self, memory_store: MemoryStore):
        """save_intersections → get_intersection_nodes が正しく動作する。"""
        m1 = await memory_store.save(content="テスト9", importance=3)

        # 手動で交差を保存
        await memory_store.save_intersections([
            ("comp-a", "comp-b", "transversal", 0.15, [m1.id]),
            ("comp-c", "comp-d", "parallel", 0.9, [m1.id]),
        ])

        scores = await memory_store.get_intersection_nodes([m1.id])
        assert m1.id in scores
        # transversal=0.8 > parallel=0.3 なので最大値0.8が返る
        assert abs(scores[m1.id] - 0.8) < 0.01


class TestConsolidateWithIntersections:
    """consolidate_memoriesでintersection統合テスト。"""

    @pytest.mark.asyncio
    async def test_consolidate_includes_intersection_stats(self, memory_store: MemoryStore):
        """consolidate_memoriesのstatsにintersectionフィールドが含まれる。"""
        await memory_store.save(content="テストA記憶", category="daily", importance=3)
        await memory_store.save(content="テストB記憶", category="daily", importance=3)

        result = await memory_store.consolidate_memories(
            window_hours=168,
            synthesize=True,
            n_layers=1,
        )

        # intersection stats が結果に含まれること
        assert "parallel_found" in result
        assert "transversal_found" in result
        assert "intersection_nodes" in result
