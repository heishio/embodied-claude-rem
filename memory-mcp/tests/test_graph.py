"""Tests for MemoryGraph (weighted verb/noun graph)."""

import pytest
import pytest_asyncio

from memory_mcp.graph import (
    WEIGHTS_INITIAL,
    WEIGHTS_RECALL,
    MemoryGraph,
)
from memory_mcp.memory import MemoryStore


@pytest_asyncio.fixture
async def graph(memory_store: MemoryStore) -> MemoryGraph:
    """Create a MemoryGraph using the test DB."""
    return MemoryGraph(db=memory_store.db)


@pytest.mark.asyncio
async def test_register_chain_creates_vv_edges(graph: MemoryGraph):
    """register_chain creates vv edges for consecutive verbs."""
    await graph.register_chain(
        verbs=["見る", "驚く", "話す"],
        nouns_per_step=[["空"], ["光"], ["シオ"]],
    )
    neighbors = await graph.query_neighbors("verb", "見る")
    # 見る -> 驚く should exist as vv
    vv_neighbors = [(t, sf) for t, sf, w in neighbors if t == "verb"]
    assert ("verb", "驚く") in vv_neighbors


@pytest.mark.asyncio
async def test_register_chain_creates_vn_edges(graph: MemoryGraph):
    """register_chain creates vn edges for verb -> noun in same step."""
    await graph.register_chain(
        verbs=["見る"],
        nouns_per_step=[["空", "雲"]],
    )
    neighbors = await graph.query_neighbors("verb", "見る")
    noun_neighbors = {sf for t, sf, w in neighbors if t == "noun"}
    assert "空" in noun_neighbors
    assert "雲" in noun_neighbors


@pytest.mark.asyncio
async def test_register_chain_creates_nn_edges(graph: MemoryGraph):
    """register_chain creates bidirectional nn edges for nouns in same step."""
    await graph.register_chain(
        verbs=["見る"],
        nouns_per_step=[["空", "雲"]],
    )
    # 空 -> 雲
    neighbors_sky = await graph.query_neighbors("noun", "空")
    assert any(sf == "雲" for _, sf, _ in neighbors_sky)
    # 雲 -> 空
    neighbors_cloud = await graph.query_neighbors("noun", "雲")
    assert any(sf == "空" for _, sf, _ in neighbors_cloud)


@pytest.mark.asyncio
async def test_initial_weights_match_constants(graph: MemoryGraph):
    """Passive (initial) weights match WEIGHTS_INITIAL."""
    await graph.register_chain(
        verbs=["見る", "驚く"],
        nouns_per_step=[["空"], ["光"]],
    )
    # vv: 見る -> 驚く
    neighbors = await graph.query_neighbors("verb", "見る")
    vv_weight = next(w for t, sf, w in neighbors if sf == "驚く")
    assert abs(vv_weight - WEIGHTS_INITIAL["vv"]) < 1e-6

    # vn: 見る -> 空
    vn_weight = next(w for t, sf, w in neighbors if sf == "空")
    assert abs(vn_weight - WEIGHTS_INITIAL["vn"]) < 1e-6

    # nn: 空 and 光 are in different steps, so no nn edge between them
    # Let's test with same-step nouns
    await graph.register_chain(
        verbs=["歩く"],
        nouns_per_step=[["海", "砂"]],
    )
    nn_neighbors = await graph.query_neighbors("noun", "海")
    nn_weight = next(w for t, sf, w in nn_neighbors if sf == "砂")
    assert abs(nn_weight - WEIGHTS_INITIAL["nn"]) < 1e-6


@pytest.mark.asyncio
async def test_recall_bump_increases_weight(graph: MemoryGraph):
    """Recall bump adds WEIGHTS_RECALL to existing edges."""
    # Initial save
    await graph.register_chain(
        verbs=["見る", "驚く"],
        nouns_per_step=[["空"], ["光"]],
    )
    # Recall bump
    await graph.register_chain(
        verbs=["見る", "驚く"],
        nouns_per_step=[["空"], ["光"]],
        delta_override=WEIGHTS_RECALL,
    )
    neighbors = await graph.query_neighbors("verb", "見る")
    vv_weight = next(w for t, sf, w in neighbors if sf == "驚く")
    expected = WEIGHTS_INITIAL["vv"] + WEIGHTS_RECALL["vv"]
    assert abs(vv_weight - expected) < 1e-6


@pytest.mark.asyncio
async def test_query_neighbors_sorted_by_weight_desc(graph: MemoryGraph):
    """query_neighbors returns results sorted by weight descending."""
    # Create edges with different weights
    await graph.register_chain(
        verbs=["見る"],
        nouns_per_step=[["空", "雲", "光"]],
    )
    # Bump "空" extra
    await graph.register_chain(
        verbs=["見る"],
        nouns_per_step=[["空"]],
        delta_override=WEIGHTS_RECALL,
    )
    neighbors = await graph.query_neighbors("verb", "見る")
    weights = [w for _, _, w in neighbors]
    assert weights == sorted(weights, reverse=True)


@pytest.mark.asyncio
async def test_consolidate_decays_weights(graph: MemoryGraph):
    """consolidate reduces edge weights by decay factor."""
    await graph.register_chain(
        verbs=["見る", "驚く"],
        nouns_per_step=[["空"], ["光"]],
    )
    original_neighbors = await graph.query_neighbors("verb", "見る")
    original_vv = next(w for t, sf, w in original_neighbors if sf == "驚く")

    await graph.consolidate()

    new_neighbors = await graph.query_neighbors("verb", "見る")
    new_vv = next(w for t, sf, w in new_neighbors if sf == "驚く")
    # After decay + normalization, weight should be reduced
    assert new_vv < original_vv


@pytest.mark.asyncio
async def test_consolidate_prunes_weak_edges(graph: MemoryGraph):
    """After enough decay cycles, weak edges are pruned."""
    await graph.register_chain(
        verbs=["歩く"],
        nouns_per_step=[["海", "砂"]],
    )
    # nn edges start at 0.05, decay * 0.8 each cycle
    # After enough cycles, should be pruned below 0.01
    for _ in range(20):
        await graph.consolidate()

    neighbors = await graph.query_neighbors("noun", "海")
    # Should be pruned (or very few left)
    nn_neighbors = [sf for t, sf, w in neighbors if t == "noun"]
    assert "砂" not in nn_neighbors


@pytest.mark.asyncio
async def test_query_nonexistent_node_returns_empty(graph: MemoryGraph):
    """query_neighbors for a nonexistent node returns empty list."""
    neighbors = await graph.query_neighbors("verb", "存在しない動詞")
    assert neighbors == []


@pytest.mark.asyncio
async def test_weight_capped_at_one(graph: MemoryGraph):
    """Edge weights should never exceed 1.0."""
    # Bump the same edge many times
    for _ in range(20):
        await graph.register_chain(
            verbs=["見る", "驚く"],
            nouns_per_step=[["空"], ["光"]],
            delta_override=WEIGHTS_RECALL,
        )
    neighbors = await graph.query_neighbors("verb", "見る")
    for _, _, w in neighbors:
        assert w <= 1.0


@pytest.mark.asyncio
async def test_consolidate_returns_stats(graph: MemoryGraph):
    """consolidate returns meaningful statistics."""
    await graph.register_chain(
        verbs=["見る", "驚く"],
        nouns_per_step=[["空"], ["光"]],
    )
    stats = await graph.consolidate()
    assert "graph_decayed" in stats
    assert "graph_pruned" in stats
    assert "graph_remaining" in stats
    assert stats["graph_decayed"] > 0
