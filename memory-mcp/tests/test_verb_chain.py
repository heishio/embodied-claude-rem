"""Tests for VerbChainStore and crystallize_buffer."""

import pytest
import pytest_asyncio

from memory_mcp.config import MemoryConfig
from memory_mcp.memory import MemoryStore
from memory_mcp.types import VerbChain, VerbStep
from memory_mcp.verb_chain import VerbChainStore, crystallize_buffer


@pytest.fixture
def sample_steps():
    return (
        VerbStep(verb="見る", nouns=("シオ", "画面")),
        VerbStep(verb="驚く", nouns=("音",)),
        VerbStep(verb="話しかける", nouns=("シオ",)),
    )


@pytest.fixture
def sample_chain(sample_steps):
    return VerbChain(
        id="test-chain-1",
        steps=sample_steps,
        timestamp="2026-02-21T12:00:00+00:00",
        emotion="happy",
        importance=4,
        source="manual",
        context="テスト用体験",
    )


# --- VerbStep tests ---


class TestVerbStep:
    def test_to_text_with_nouns(self):
        step = VerbStep(verb="見る", nouns=("シオ", "画面"))
        assert step.to_text() == "見る(シオ, 画面)"

    def test_to_text_without_nouns(self):
        step = VerbStep(verb="驚く", nouns=())
        assert step.to_text() == "驚く"

    def test_to_dict_roundtrip(self):
        step = VerbStep(verb="話す", nouns=("シオ",))
        restored = VerbStep.from_dict(step.to_dict())
        assert restored == step


# --- VerbChain tests ---


class TestVerbChain:
    def test_to_document(self, sample_chain):
        doc = sample_chain.to_document()
        assert "見る(シオ, 画面)" in doc
        assert "→" in doc
        assert "テスト用体験" in doc

    def test_to_metadata_roundtrip(self, sample_chain):
        metadata = sample_chain.to_metadata()
        restored = VerbChain.from_metadata(sample_chain.id, metadata)
        assert restored.id == sample_chain.id
        assert restored.steps == sample_chain.steps
        assert restored.emotion == sample_chain.emotion
        assert restored.importance == sample_chain.importance

    def test_all_verbs_in_metadata(self, sample_chain):
        metadata = sample_chain.to_metadata()
        verbs = set(metadata["all_verbs"].split(","))
        assert "見る" in verbs
        assert "驚く" in verbs
        assert "話しかける" in verbs

    def test_all_nouns_in_metadata(self, sample_chain):
        metadata = sample_chain.to_metadata()
        nouns = set(metadata["all_nouns"].split(","))
        assert "シオ" in nouns
        assert "画面" in nouns
        assert "音" in nouns


# --- crystallize_buffer tests ---


class TestCrystallizeBuffer:
    def test_empty_entries(self):
        assert crystallize_buffer([]) == []

    def test_single_entry_below_min(self):
        entries = [{"v": ["見る"], "w": ["シオ"]}]
        chains = crystallize_buffer(entries, min_verbs=2)
        assert chains == []

    def test_single_entry_meets_min(self):
        entries = [{"v": ["見る", "思う"], "w": ["シオ"]}]
        chains = crystallize_buffer(entries, min_verbs=2)
        assert len(chains) == 1
        assert len(chains[0].steps) == 2

    def test_grouping_by_shared_nouns(self):
        entries = [
            {"v": ["見る"], "w": ["シオ", "画面"]},
            {"v": ["驚く"], "w": ["シオ", "音"]},  # shares "シオ" → same group
            {"v": ["食べる"], "w": ["ケーキ"]},  # no shared noun → new group
        ]
        chains = crystallize_buffer(entries, min_verbs=1)
        assert len(chains) == 2
        # First chain: 見る + 驚く
        assert len(chains[0].steps) == 2
        # Second chain: 食べる
        assert len(chains[1].steps) == 1

    def test_no_verbs_skipped(self):
        entries = [
            {"v": [], "w": ["シオ"]},
            {"v": ["見る", "思う"], "w": ["画面"]},
        ]
        chains = crystallize_buffer(entries, min_verbs=2)
        assert len(chains) == 1

    def test_source_is_buffer(self):
        entries = [{"v": ["見る", "思う"], "w": ["シオ"]}]
        chains = crystallize_buffer(entries, min_verbs=1)
        assert chains[0].source == "buffer"

    def test_emotion_and_importance(self):
        entries = [{"v": ["見る", "思う"], "w": ["シオ"]}]
        chains = crystallize_buffer(entries, emotion="happy", importance=5, min_verbs=1)
        assert chains[0].emotion == "happy"
        assert chains[0].importance == 5


# --- VerbChainStore integration tests ---


@pytest_asyncio.fixture
async def memory_store(tmp_path):
    config = MemoryConfig(
        db_path=str(tmp_path / "test_memory.db"),
        collection_name="test_memories",
    )
    store = MemoryStore(config)
    await store.connect()
    yield store
    await store.disconnect()


@pytest_asyncio.fixture
async def verb_chain_store(memory_store):
    store = VerbChainStore(
        db=memory_store.db,
        embedding_fn=memory_store.embedding_fn,
    )
    await store.initialize()
    return store


@pytest.mark.asyncio
class TestVerbChainStore:
    async def test_save_and_search(self, verb_chain_store, sample_chain):
        await verb_chain_store.save(sample_chain)
        results = await verb_chain_store.search("シオに話しかけた")
        assert len(results) >= 1
        found_chain, score = results[0]
        assert found_chain.id == sample_chain.id

    async def test_find_by_verb(self, verb_chain_store, sample_chain):
        await verb_chain_store.save(sample_chain)
        chains = await verb_chain_store.find_by_verb("見る")
        assert len(chains) == 1
        assert chains[0].id == sample_chain.id

    async def test_find_by_noun(self, verb_chain_store, sample_chain):
        await verb_chain_store.save(sample_chain)
        chains = await verb_chain_store.find_by_noun("シオ")
        assert len(chains) == 1
        assert chains[0].id == sample_chain.id

    async def test_find_by_verb_not_found(self, verb_chain_store, sample_chain):
        await verb_chain_store.save(sample_chain)
        chains = await verb_chain_store.find_by_verb("飛ぶ")
        assert len(chains) == 0

    async def test_expand_from_verb(self, verb_chain_store):
        chain1 = VerbChain(
            id="c1",
            steps=(
                VerbStep(verb="見る", nouns=("シオ",)),
                VerbStep(verb="笑う", nouns=("シオ",)),
            ),
            timestamp="2026-02-21T12:00:00+00:00",
            emotion="happy",
            importance=3,
            source="manual",
            context="",
        )
        chain2 = VerbChain(
            id="c2",
            steps=(
                VerbStep(verb="笑う", nouns=("画面",)),
                VerbStep(verb="話す", nouns=("画面",)),
            ),
            timestamp="2026-02-21T13:00:00+00:00",
            emotion="happy",
            importance=3,
            source="manual",
            context="",
        )
        await verb_chain_store.save(chain1)
        await verb_chain_store.save(chain2)

        # "見る" → chain1 → chain1 has "笑う" → chain2 also has "笑う"
        results = await verb_chain_store.expand_from_fragment(verb="見る", depth=2)
        result_ids = {c.id for c in results}
        assert "c1" in result_ids
        assert "c2" in result_ids

    async def test_expand_from_noun(self, verb_chain_store):
        chain1 = VerbChain(
            id="c1",
            steps=(VerbStep(verb="見る", nouns=("シオ",)),),
            timestamp="2026-02-21T12:00:00+00:00",
            emotion="neutral",
            importance=3,
            source="manual",
            context="",
        )
        await verb_chain_store.save(chain1)

        results = await verb_chain_store.expand_from_fragment(noun="シオ", depth=1)
        assert len(results) == 1
        assert results[0].id == "c1"

    async def test_get_all(self, verb_chain_store, sample_chain):
        await verb_chain_store.save(sample_chain)
        all_chains = await verb_chain_store.get_all()
        assert len(all_chains) == 1

    async def test_initialize_rebuilds_index(self, memory_store, sample_chain):
        store = VerbChainStore(
            db=memory_store.db,
            embedding_fn=memory_store.embedding_fn,
        )
        await store.initialize()
        await store.save(sample_chain)

        # Create a new store from the same DB to test initialize
        new_store = VerbChainStore(
            db=memory_store.db,
            embedding_fn=memory_store.embedding_fn,
        )
        await new_store.initialize()

        chains = await new_store.find_by_verb("見る")
        assert len(chains) == 1
