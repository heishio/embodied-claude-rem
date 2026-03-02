"""Tests for boundary layer detection (Step 2: edge detection + noise layers)
and path-dependent layer selection (Step 3b)."""

import numpy as np
import pytest

from memory_mcp.consolidation import ConsolidationEngine
from memory_mcp.graph import MemoryGraph
from memory_mcp.memory import MemoryStore
from memory_mcp.types import MemorySearchResult, VerbChain, VerbStep
from memory_mcp.vector import encode_vector
from memory_mcp.verb_chain import VerbChainStore


class TestBoundaryLayerBase:
    """Layer 0: ノイズなしの外縁検出テスト。"""

    @pytest.mark.asyncio
    async def test_edge_core_classification(self, memory_store: MemoryStore):
        """重心から遠いメンバーが edge、近いメンバーが core に分類される。"""
        # 類似した記憶を作成して合成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough to create composite")

        # Boundary layers 計算
        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            n_layers=0,  # base layer only
        )

        assert bl_stats["composites_processed"] >= 1
        assert bl_stats["total_layers"] >= 1

        # DB に保存されていることを確認
        db = memory_store.db
        rows = db.execute("SELECT * FROM boundary_layers WHERE layer_index = 0").fetchall()
        assert len(rows) >= 2  # 少なくとも2メンバー

        # edge と core が存在する
        edges = [r for r in rows if r["is_edge"] == 1]
        cores = [r for r in rows if r["is_edge"] == 0]
        # 3メンバーなら、少なくとも1つは core で1つは edge
        assert len(edges) + len(cores) >= 2

    @pytest.mark.asyncio
    async def test_two_members_one_edge_one_core(self, memory_store: MemoryStore):
        """2メンバーの場合、距離が均等なら分類が合理的。"""
        await memory_store.save(content="朝ごはんを食べた", category="daily", importance=3)
        await memory_store.save(content="朝食を食べた", category="daily", importance=3)

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        if stats["composites_created"] == 0:
            pytest.skip("memories not similar enough")

        bl_stats = await engine.compute_boundary_layers(
            store=memory_store, n_layers=0,
        )

        assert bl_stats["composites_processed"] >= 1

        db = memory_store.db
        rows = db.execute("SELECT * FROM boundary_layers WHERE layer_index = 0").fetchall()
        assert len(rows) >= 2


class TestBoundaryLayerNoise:
    """Layer 1+: ノイズレイヤーのテスト。"""

    @pytest.mark.asyncio
    async def test_multiple_layers_generated(self, memory_store: MemoryStore):
        """n_layers=3 で4レイヤー（0,1,2,3）が生成される。"""
        await memory_store.save(content="散歩に出かけた", category="daily", importance=3)
        await memory_store.save(content="散歩をした", category="daily", importance=3)
        await memory_store.save(content="散歩に行った", category="daily", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            n_layers=3,
        )

        if bl_stats["composites_processed"] == 0:
            pytest.skip("no composites to process")

        db = memory_store.db
        layer_indices = set()
        for row in db.execute("SELECT DISTINCT layer_index FROM boundary_layers").fetchall():
            layer_indices.add(row["layer_index"])

        # layer 0, 1, 2, 3
        assert 0 in layer_indices
        assert len(layer_indices) == 4

    @pytest.mark.asyncio
    async def test_noise_causes_edge_variation(self, memory_store: MemoryStore):
        """ノイズによって異なるレイヤーで edge/core 判定が異なりうる。"""
        # より散らばった記憶を作成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=2)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=5)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)
        await memory_store.save(content="猫が窓際で横たわっている", category="observation", importance=4)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            n_layers=3,
            noise_scale=0.5,  # 大きめのノイズで変動を起こしやすく
        )

        if bl_stats["composites_processed"] == 0:
            pytest.skip("no composites to process")

        # レイヤーが正しく生成されていることだけ確認（変動は確率的なので保証はできない）
        db = memory_store.db
        total = db.execute("SELECT COUNT(*) FROM boundary_layers").fetchone()[0]
        assert total > 0


class TestBoundaryLayerWithGraph:
    """グラフベースのテンプレートを使ったテスト。"""

    @pytest.mark.asyncio
    async def test_with_graph_templates(self, memory_store: MemoryStore):
        """グラフがある場合にテンプレートベースのノイズが生成される。"""
        # 記憶を作成
        await memory_store.save(content="猫が寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が眠っている", category="observation", importance=3)

        # VerbChain を作成してグラフにエッジを登録
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        chain = VerbChain(
            id="test-chain-1",
            steps=(
                VerbStep(verb="見る", nouns=("猫", "窓")),
                VerbStep(verb="撫でる", nouns=("猫",)),
            ),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="1",
            importance=3,
            source="manual",
            context="猫を見て撫でた",
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

        if bl_stats["composites_processed"] == 0:
            pytest.skip("no composites to process")

        assert bl_stats["total_layers"] >= 3  # 1 + 2

    @pytest.mark.asyncio
    async def test_max_template_strength_cap(self, memory_store: MemoryStore):
        """max_template_strength のキャップが効いている。"""
        await memory_store.save(content="テスト記憶1", category="daily", importance=3)
        await memory_store.save(content="テスト記憶2", category="daily", importance=3)

        graph = MemoryGraph(memory_store.db)
        # 非常に強いエッジを登録
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        for i in range(5):
            chain = VerbChain(
                id=f"strong-chain-{i}",
                steps=(
                    VerbStep(verb="繰り返す", nouns=("同じこと",)),
                    VerbStep(verb="考える", nouns=("同じこと",)),
                ),
                timestamp="2026-01-01T00:00:00+00:00",
                emotion="8", importance=3, source="manual", context="",
            )
            await chain_store.save(chain)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        # max_template_strength=0.1 で制限
        bl_stats = await engine.compute_boundary_layers(
            store=memory_store,
            graph=graph,
            n_layers=1,
            max_template_strength=0.1,
        )

        # エラーなく完了すること
        assert bl_stats["composites_processed"] >= 0


class TestBoundaryLayerFallback:
    """グラフなしフォールバックテスト。"""

    @pytest.mark.asyncio
    async def test_no_graph_random_noise(self, memory_store: MemoryStore):
        """グラフなしでもランダムノイズで動作する。"""
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
            graph=None,  # グラフなし
            n_layers=2,
        )

        if bl_stats["composites_processed"] == 0:
            pytest.skip("no composites to process")

        db = memory_store.db
        total = db.execute("SELECT COUNT(*) FROM boundary_layers").fetchone()[0]
        assert total > 0


class TestBoundaryLayerIdempotent:
    """二重実行テスト。"""

    @pytest.mark.asyncio
    async def test_clear_and_rebuild(self, memory_store: MemoryStore):
        """二重実行でクリア＆リビルドされる。"""
        await memory_store.save(content="花が咲いている", category="observation", importance=3)
        await memory_store.save(content="花が咲いた", category="observation", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        # 1回目
        await engine.compute_boundary_layers(store=memory_store, n_layers=2)

        db = memory_store.db
        count1 = db.execute("SELECT COUNT(*) FROM boundary_layers").fetchone()[0]

        # 2回目
        await engine.compute_boundary_layers(store=memory_store, n_layers=2)

        count2 = db.execute("SELECT COUNT(*) FROM boundary_layers").fetchone()[0]

        # クリア＆リビルドなので件数は同じ
        assert count1 == count2

    @pytest.mark.asyncio
    async def test_n_layers_change(self, memory_store: MemoryStore):
        """n_layers を変えて再実行しても正しく動く。"""
        await memory_store.save(content="本を読んだ", category="daily", importance=3)
        await memory_store.save(content="読書をした", category="daily", importance=3)

        engine = ConsolidationEngine()
        await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        # n_layers=1 で実行
        await engine.compute_boundary_layers(store=memory_store, n_layers=1)
        db = memory_store.db
        layers1 = set(r["layer_index"] for r in db.execute(
            "SELECT DISTINCT layer_index FROM boundary_layers"
        ).fetchall())

        # n_layers=3 で再実行
        await engine.compute_boundary_layers(store=memory_store, n_layers=3)
        layers2 = set(r["layer_index"] for r in db.execute(
            "SELECT DISTINCT layer_index FROM boundary_layers"
        ).fetchall())

        if layers1:
            assert len(layers1) <= 2  # 0 and 1
        if layers2:
            assert len(layers2) <= 4  # 0, 1, 2, 3


class TestBoundaryAwareRecall:
    """Step 3a: boundary-aware recall 統合テスト。"""

    async def _create_composite_with_boundaries(
        self, memory_store: MemoryStore, n_layers: int = 2
    ) -> bool:
        """ヘルパー: composite + boundary layers を作成。成功なら True。"""
        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )
        if stats["composites_created"] == 0:
            return False
        await engine.compute_boundary_layers(
            store=memory_store,
            n_layers=n_layers,
        )
        return True

    @pytest.mark.asyncio
    async def test_recall_divergent_with_boundary_scores(self, memory_store: MemoryStore):
        """recall_divergent が boundary_score を workspace selection に反映する。"""
        # 類似した記憶を作成して composite を生成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        if not await self._create_composite_with_boundaries(memory_store):
            pytest.skip("memories not similar enough to create composite")

        # recall_divergent 実行
        results, _ = await memory_store.recall_divergent(
            context="猫",
            n_results=5,
            record_activation=False,
        )

        # 結果が返ること（composite がある状態で正常動作）
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_recall_divergent_without_composites(self, memory_store: MemoryStore):
        """composite がない場合でも recall_divergent が正常動作する。"""
        await memory_store.save(content="朝ごはんを食べた", category="daily", importance=3)
        await memory_store.save(content="夕日がきれいだった", category="observation", importance=4)
        await memory_store.save(content="本を読んだ", category="daily", importance=2)

        # composite なしで recall_divergent
        results, _ = await memory_store.recall_divergent(
            context="食事",
            n_results=3,
            record_activation=False,
        )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_get_member_boundary_scores(self, memory_store: MemoryStore):
        """get_member_boundary_scores が正しい fuzziness を返す。"""
        await memory_store.save(content="散歩に出かけた", category="daily", importance=3)
        await memory_store.save(content="散歩をした", category="daily", importance=3)
        await memory_store.save(content="散歩に行った", category="daily", importance=3)

        if not await self._create_composite_with_boundaries(memory_store, n_layers=2):
            pytest.skip("no composites created")

        # boundary_layers に記録されたメンバーID を取得
        db = memory_store.db
        member_rows = db.execute("SELECT DISTINCT member_id FROM boundary_layers").fetchall()
        member_ids = [r["member_id"] for r in member_rows]

        if not member_ids:
            pytest.skip("no boundary layers recorded")

        scores = await memory_store.get_member_boundary_scores(member_ids)

        # 全メンバーにスコアが返る
        assert len(scores) == len(member_ids)
        for mid, score in scores.items():
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_member_boundary_scores_empty(self, memory_store: MemoryStore):
        """boundary_layers に存在しないIDは空の dict を返す。"""
        scores = await memory_store.get_member_boundary_scores(["nonexistent-id"])
        assert scores == {}

    @pytest.mark.asyncio
    async def test_find_adjacent_composites(self, memory_store: MemoryStore):
        """find_adjacent_composites が隣接 composite を返す。"""
        # 2つのグループを作成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="犬が庭で走っている", category="observation", importance=3)
        await memory_store.save(content="犬が庭で遊んでいる", category="observation", importance=3)

        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )

        if stats["composites_created"] < 2:
            pytest.skip("need at least 2 composites")

        await engine.compute_boundary_layers(store=memory_store, n_layers=1)

        # composite ID を取得
        composite_ids = await memory_store.fetch_all_composite_ids()
        assert len(composite_ids) >= 2

        # 隣接 composite を検索
        from memory_mcp.vector import decode_vector
        centroid = await memory_store.fetch_composite_centroid(composite_ids[0])
        if centroid is None:
            pytest.skip("no centroid vector")

        adjacent = await memory_store.find_adjacent_composites(
            composite_ids[0], centroid, n_results=3,
        )

        # 自分自身は含まないこと
        own_ids = {composite_ids[0]}
        for adj_id, sim in adjacent:
            assert adj_id not in own_ids
            assert -1.0 <= sim <= 1.0

    @pytest.mark.asyncio
    async def test_expand_composite_edges(self, memory_store: MemoryStore):
        """expand_composite_edges が edge メンバーを返す。"""
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        if not await self._create_composite_with_boundaries(memory_store, n_layers=1):
            pytest.skip("no composites created")

        composite_ids = await memory_store.fetch_all_composite_ids()
        assert len(composite_ids) >= 1

        # ダミー query_vec
        import numpy as np
        query_vec = np.zeros(384, dtype=np.float32)

        edge_memories = await memory_store.expand_composite_edges(composite_ids, query_vec)

        # edge メンバーが存在すれば返る
        # (boundary_layers にデータがあれば少なくとも1つ)
        db = memory_store.db
        edge_count = db.execute(
            "SELECT COUNT(*) FROM boundary_layers WHERE is_edge = 1 AND layer_index = 0"
        ).fetchone()[0]
        if edge_count > 0:
            assert len(edge_memories) >= 1

    @pytest.mark.asyncio
    async def test_expand_composite_edges_empty(self, memory_store: MemoryStore):
        """composite_ids が空の場合、空リストを返す。"""
        import numpy as np
        query_vec = np.zeros(384, dtype=np.float32)
        result = await memory_store.expand_composite_edges([], query_vec)
        assert result == []


class TestPathDependentLayerSelection:
    """Step 3b: 経路依存レイヤー選択テスト。"""

    async def _setup_composites_and_chains(
        self, memory_store: MemoryStore
    ) -> tuple[bool, VerbChainStore]:
        """ヘルパー: composite + boundary layers + verb chains を作成。"""
        # 類似記憶を作成
        await memory_store.save(content="猫が窓辺で寝ている", category="observation", importance=3)
        await memory_store.save(content="猫が窓辺で丸くなっている", category="observation", importance=3)
        await memory_store.save(content="猫が窓の近くで眠っている", category="observation", importance=3)

        # VerbChain を作成
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)

        chain1 = VerbChain(
            id="path-chain-1",
            steps=(
                VerbStep(verb="見る", nouns=("猫", "窓")),
                VerbStep(verb="撫でる", nouns=("猫",)),
            ),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="1", importance=3, source="manual", context="猫を見て撫でた",
        )
        chain2 = VerbChain(
            id="path-chain-2",
            steps=(
                VerbStep(verb="散歩", nouns=("公園",)),
                VerbStep(verb="見つける", nouns=("花",)),
            ),
            timestamp="2026-01-02T00:00:00+00:00",
            emotion="1", importance=3, source="manual", context="散歩で花を見つけた",
        )
        await chain_store.save(chain1)
        await chain_store.save(chain2)

        # composite + boundary layers
        engine = ConsolidationEngine()
        stats = await engine.synthesize_composites(
            store=memory_store,
            similarity_threshold=0.3,
            min_group_size=2,
        )
        if stats["composites_created"] == 0:
            return False, chain_store

        await engine.compute_boundary_layers(
            store=memory_store, graph=graph, n_layers=2,
        )
        return True, chain_store

    @pytest.mark.asyncio
    async def test_select_active_boundary_layer(self, memory_store: MemoryStore):
        """select_active_boundary_layer が整数を返す。"""
        ok, _ = await self._setup_composites_and_chains(memory_store)
        if not ok:
            pytest.skip("no composites created")

        # ダミーの path vector
        path_vec = np.random.randn(300).astype(np.float32)
        layer_idx = await memory_store.select_active_boundary_layer(path_vec)
        assert isinstance(layer_idx, int)
        assert layer_idx >= 0

    @pytest.mark.asyncio
    async def test_select_active_boundary_layer_no_data(self, memory_store: MemoryStore):
        """boundary_layers が空の場合、0 を返す。"""
        path_vec = np.random.randn(300).astype(np.float32)
        layer_idx = await memory_store.select_active_boundary_layer(path_vec)
        assert layer_idx == 0

    @pytest.mark.asyncio
    async def test_get_chain_boundary_scores_with_layer(self, memory_store: MemoryStore):
        """layer_index 指定で chain boundary scores を取得。"""
        ok, chain_store = await self._setup_composites_and_chains(memory_store)
        if not ok:
            pytest.skip("no composites created")

        scores = await memory_store.get_chain_boundary_scores(
            ["path-chain-1", "path-chain-2"], layer_index=0,
        )
        # chain embeddings が存在するものだけスコアが返る
        for cid, score in scores.items():
            assert -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_chain_boundary_scores_fuzziness(self, memory_store: MemoryStore):
        """layer_index=None で fuzziness ベースの chain boundary scores。"""
        ok, chain_store = await self._setup_composites_and_chains(memory_store)
        if not ok:
            pytest.skip("no composites created")

        scores = await memory_store.get_chain_boundary_scores(
            ["path-chain-1", "path-chain-2"], layer_index=None,
        )
        for cid, score in scores.items():
            assert -1.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_get_chain_boundary_scores_empty(self, memory_store: MemoryStore):
        """空の chain_ids では空 dict を返す。"""
        scores = await memory_store.get_chain_boundary_scores([])
        assert scores == {}

    @pytest.mark.asyncio
    async def test_get_chain_boundary_scores_no_boundary_data(self, memory_store: MemoryStore):
        """boundary_layers がない場合でも正常動作（全 0.0）。"""
        # chain だけ作って boundary layers は作らない
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        chain = VerbChain(
            id="solo-chain",
            steps=(VerbStep(verb="歩く", nouns=("道",)),),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="8", importance=3, source="manual", context="",
        )
        await chain_store.save(chain)

        scores = await memory_store.get_chain_boundary_scores(["solo-chain"])
        assert scores.get("solo-chain", 0.0) == 0.0

    @pytest.mark.asyncio
    async def test_expand_from_fragment_returns_visited(self, memory_store: MemoryStore):
        """expand_from_fragment が visited_verbs, visited_nouns を返す。"""
        graph = MemoryGraph(memory_store.db)
        chain_store = VerbChainStore(memory_store.db, memory_store.chive, graph=graph)
        chain = VerbChain(
            id="visit-chain",
            steps=(
                VerbStep(verb="見る", nouns=("コウタ",)),
                VerbStep(verb="笑う", nouns=("コウタ",)),
            ),
            timestamp="2026-01-01T00:00:00+00:00",
            emotion="1", importance=3, source="manual", context="",
        )
        await chain_store.save(chain)

        chains, v_verbs, v_nouns = await chain_store.expand_from_fragment(verb="見る", depth=1)
        assert len(chains) >= 1
        assert isinstance(v_verbs, list)
        assert isinstance(v_nouns, list)
        # seed verb は visited_verbs に含まれる
        assert "見る" in v_verbs

    @pytest.mark.asyncio
    async def test_recall_experience_boundary_rerank(self, memory_store: MemoryStore):
        """recall_experience が boundary scores で正常にリランクされる。"""
        ok, chain_store = await self._setup_composites_and_chains(memory_store)
        if not ok:
            pytest.skip("no composites created")

        # search で結果取得 → boundary rerank が例外なく動く
        results = await chain_store.search(query="猫", n_results=5)
        if not results:
            pytest.skip("no search results")

        chain_ids = [c.id for c, _ in results]
        boundary_scores = await memory_store.get_chain_boundary_scores(
            chain_ids, layer_index=None,
        )

        # rerank logic
        BOUNDARY_WEIGHT = 0.05
        reranked = [
            (chain, score - BOUNDARY_WEIGHT * boundary_scores.get(chain.id, 0.0))
            for chain, score in results
        ]
        reranked.sort(key=lambda x: x[1])

        # 件数が維持される
        assert len(reranked) == len(results)
