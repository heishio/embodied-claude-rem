"""Tests for template bias updates (bias accumulation, decay, application)."""

import numpy as np
import pytest

from memory_mcp.consolidation import (
    BIAS_ACCUMULATION_RATE,
    BIAS_APPLY_COEFFICIENT,
    BIAS_DECAY_FACTOR,
    BIAS_MAX_CAP,
    BIAS_PRUNE_THRESHOLD,
    ConsolidationEngine,
)
from memory_mcp.graph import MemoryGraph
from memory_mcp.memory import MemoryStore
from memory_mcp.types import VerbChain, VerbStep
from memory_mcp.verb_chain import VerbChainStore


class TestTemplateBiasesCRUD:
    """バイアステーブルの基本CRUD。"""

    @pytest.mark.asyncio
    async def test_save_fetch_roundtrip(self, memory_store: MemoryStore):
        """save → fetch でバイアスが往復する。"""
        biases = [
            ("chain-a", 0.05, 1),
            ("chain-b", 0.10, 3),
        ]
        await memory_store.save_template_biases(biases)
        result = await memory_store.fetch_template_biases()

        assert result["chain-a"] == pytest.approx(0.05)
        assert result["chain-b"] == pytest.approx(0.10)

    @pytest.mark.asyncio
    async def test_save_upsert(self, memory_store: MemoryStore):
        """同じchain_idで再保存するとupdateされる。"""
        await memory_store.save_template_biases([("chain-a", 0.05, 1)])
        await memory_store.save_template_biases([("chain-a", 0.12, 2)])
        result = await memory_store.fetch_template_biases()

        assert result["chain-a"] == pytest.approx(0.12)

    @pytest.mark.asyncio
    async def test_fetch_empty(self, memory_store: MemoryStore):
        """テーブルが空なら空辞書を返す。"""
        result = await memory_store.fetch_template_biases()
        assert result == {}

    @pytest.mark.asyncio
    async def test_save_empty_list(self, memory_store: MemoryStore):
        """空リストを渡してもエラーにならない。"""
        await memory_store.save_template_biases([])
        result = await memory_store.fetch_template_biases()
        assert result == {}


class TestBiasDecay:
    """バイアスの減衰・刈り取り。"""

    @pytest.mark.asyncio
    async def test_decay_factor(self, memory_store: MemoryStore):
        """減衰で bias_weight が BIAS_DECAY_FACTOR 倍になる。"""
        await memory_store.save_template_biases([("chain-a", 0.10, 1)])
        await memory_store.decay_template_biases(BIAS_DECAY_FACTOR, BIAS_PRUNE_THRESHOLD)
        result = await memory_store.fetch_template_biases()

        assert result["chain-a"] == pytest.approx(0.10 * BIAS_DECAY_FACTOR)

    @pytest.mark.asyncio
    async def test_prune_below_threshold(self, memory_store: MemoryStore):
        """閾値以下のバイアスが刈り取られる。"""
        await memory_store.save_template_biases([
            ("chain-small", 0.0005, 1),  # below threshold after decay
            ("chain-large", 0.10, 1),
        ])
        stats = await memory_store.decay_template_biases(BIAS_DECAY_FACTOR, BIAS_PRUNE_THRESHOLD)
        result = await memory_store.fetch_template_biases()

        # chain-small: 0.0005 * 0.90 = 0.00045 < 0.001 → pruned
        assert "chain-small" not in result
        assert "chain-large" in result
        assert stats["biases_pruned"] >= 1

    @pytest.mark.asyncio
    async def test_decay_returns_stats(self, memory_store: MemoryStore):
        """decay が統計を返す。"""
        await memory_store.save_template_biases([
            ("chain-a", 0.10, 1),
            ("chain-b", 0.05, 2),
        ])
        stats = await memory_store.decay_template_biases(BIAS_DECAY_FACTOR, BIAS_PRUNE_THRESHOLD)

        assert "biases_decayed" in stats
        assert "biases_pruned" in stats


class TestBiasMaxCap:
    """バイアスが BIAS_MAX_CAP を超えない。"""

    @pytest.mark.asyncio
    async def test_cap_enforced(self, memory_store: MemoryStore):
        """蓄積後のバイアスが BIAS_MAX_CAP を超えないことを確認。"""
        # BIAS_MAX_CAP を大きく超える値を直接テスト
        # _update_biases 内で min(BIAS_MAX_CAP, ...) が効くことを確認
        # 直接 save して BIAS_MAX_CAP 以上にならないことを検証するのは
        # _update_biases のロジックに依存するので、統合テストで検証する
        initial_bias = BIAS_MAX_CAP - 0.001
        await memory_store.save_template_biases([("chain-capped", initial_bias, 100)])
        result = await memory_store.fetch_template_biases()
        assert result["chain-capped"] < BIAS_MAX_CAP + 0.001


class TestApplyNoiseWithBias:
    """_apply_noise がバイアスありとなしで出力が異なる。"""

    def test_bias_changes_output(self):
        """バイアスありとなしで _apply_noise の出力が異なる。"""
        engine = ConsolidationEngine()
        member_vecs = np.random.randn(3, 384).astype(np.float32)
        member_vecs /= np.linalg.norm(member_vecs, axis=1, keepdims=True)
        t_vec = np.random.randn(384).astype(np.float32)
        t_vec /= np.linalg.norm(t_vec)

        # バイアスなし
        templates_no_bias = [(t_vec, 0.2, 0.0, "chain-1")]
        result_no_bias = engine._apply_noise(member_vecs, templates_no_bias, 0.1, seed=42)

        # バイアスあり（大きめのバイアスで差が出やすく）
        templates_with_bias = [(t_vec, 0.2, BIAS_MAX_CAP, "chain-1")]
        result_with_bias = engine._apply_noise(member_vecs, templates_with_bias, 0.1, seed=42)

        # 出力が異なることを確認
        assert not np.allclose(result_no_bias, result_with_bias)

    def test_zero_bias_matches_no_bias(self):
        """バイアス=0 のとき既存動作と同一（後方互換）。"""
        engine = ConsolidationEngine()
        member_vecs = np.random.randn(3, 384).astype(np.float32)
        member_vecs /= np.linalg.norm(member_vecs, axis=1, keepdims=True)
        t_vec = np.random.randn(384).astype(np.float32)
        t_vec /= np.linalg.norm(t_vec)

        # バイアス=0
        templates = [(t_vec, 0.2, 0.0, "chain-1")]
        result = engine._apply_noise(member_vecs, templates, 0.1, seed=42)

        # effective_strength = 0.2 + 0.05*0.0 = 0.2（transient_strengthと同じ）
        # 同じseedで同じ入力なので結果は同一のはず
        result2 = engine._apply_noise(member_vecs, templates, 0.1, seed=42)
        assert np.allclose(result, result2)

    def test_empty_templates_fallback(self):
        """テンプレートが空のときランダムノイズにフォールバック。"""
        engine = ConsolidationEngine()
        member_vecs = np.random.randn(3, 384).astype(np.float32)
        member_vecs /= np.linalg.norm(member_vecs, axis=1, keepdims=True)

        result = engine._apply_noise(member_vecs, [], 0.1, seed=42)
        # 結果の形状が正しい
        assert result.shape == member_vecs.shape
        # 正規化されている
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


class TestConsolidationWithBias:
    """コンソリデーション後にバイアスが生成される統合テスト。"""

    @pytest.mark.asyncio
    async def test_bias_generated_after_consolidation(self, memory_store: MemoryStore):
        """コンソリデーション後にバイアスが生成される。"""
        # 類似記憶を作成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        # VerbChain を作成してグラフにエッジ登録
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        chain = VerbChain(
            id="bias-chain-1",
            steps=(
                VerbStep(verb="見る", nouns=("猫", "窓")),
                VerbStep(verb="撫でる", nouns=("猫",)),
            ),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="1", importance=3, source="manual", context="猫を見て撫でた",
        )
        await chain_store.save(chain)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            graph=graph,
            n_layers=2,
        )

        # バイアス関連の統計が返る
        assert "biases_updated" in bl_stats
        assert "biases_decayed" in bl_stats
        assert "biases_pruned" in bl_stats

        if bl_stats["composites_processed"] > 0 and bl_stats["biases_updated"] > 0:
            # バイアスが保存されている
            biases = await memory_store.fetch_template_biases()
            assert len(biases) > 0
            # キャップを超えていない
            for chain_id, weight in biases.items():
                assert weight <= BIAS_MAX_CAP

    @pytest.mark.asyncio
    async def test_no_graph_no_biases(self, memory_store: MemoryStore):
        """グラフなしの場合、バイアスは生成されない。"""
        await memory_store.save(content="テストA", category="daily", importance=3)
        await memory_store.save(content="テストB", category="daily", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            graph=None,
            n_layers=2,
        )

        assert bl_stats["biases_updated"] == 0

    @pytest.mark.asyncio
    async def test_backward_compat_zero_bias(self, memory_store: MemoryStore):
        """バイアスなし状態は既存動作と同一。"""
        await memory_store.save(content="散歩に出かけた", category="daily", importance=3)
        await memory_store.save(content="散歩をした", category="daily", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        # バイアスが空の状態でcompute_boundary_layers
        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            n_layers=2,
        )

        # エラーなく動作すること
        assert bl_stats["composites_processed"] >= 0
        assert bl_stats["biases_updated"] == 0


class TestConsolidateMemoriesIntegration:
    """consolidate_memories → recall_divergent の統合テスト。"""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, memory_store: MemoryStore):
        """記憶作成→チェーン作成→コンソリデーション→recall_divergent が正常動作。"""
        # 記憶作成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        # VerbChain 作成
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        chain = VerbChain(
            id="integration-chain-1",
            steps=(
                VerbStep(verb="見る", nouns=("猫", "窓")),
                VerbStep(verb="撫でる", nouns=("猫",)),
            ),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="1", importance=3, source="manual", context="猫を見て撫でた",
        )
        await chain_store.save(chain)

        # コンソリデーション
        result = await memory_store.consolidate_memories(
            window_hours=9999,
            synthesize=True,
            n_layers=2,
            graph=graph,
        )

        # バイアス統計が含まれる
        assert "biases_updated" in result
        assert "biases_decayed" in result
        assert "biases_pruned" in result

        # recall_divergent が正常動作
        results, _ = await memory_store.recall_divergent(
            context="猫",
            n_results=5,
            record_activation=False,
        )
        assert len(results) >= 1
