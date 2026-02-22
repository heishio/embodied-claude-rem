"""VerbChainStore: 動詞チェーン記憶の保存・検索・連想展開 (SQLite backend)."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .embedding import E5EmbeddingFunction
from .graph import WEIGHTS_RECALL, MemoryGraph
from .scoring import (
    calculate_emotion_boost,
    calculate_final_score,
    calculate_importance_boost,
    calculate_time_decay,
)
from .types import VerbChain, VerbStep
from .vector import cosine_similarity, decode_vector, encode_vector


class VerbChainStore:
    """SQLite-backed verb chain storage with inverted indexes."""

    def __init__(
        self,
        db: sqlite3.Connection,
        embedding_fn: E5EmbeddingFunction,
        graph: MemoryGraph | None = None,
    ):
        self._db = db
        self._embedding_fn = embedding_fn
        self._graph = graph
        # 転置インデックス（インメモリ）
        self._noun_to_chain_ids: dict[str, set[str]] = defaultdict(set)
        self._verb_to_chain_ids: dict[str, set[str]] = defaultdict(set)

    async def initialize(self) -> None:
        """起動時に全チェーンから転置インデックスを構築."""
        rows = await asyncio.to_thread(
            lambda: self._db.execute(
                "SELECT id, all_verbs, all_nouns FROM verb_chains"
            ).fetchall()
        )

        for row in rows:
            chain_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
            verbs_str = row["all_verbs"] if isinstance(row, sqlite3.Row) else row[1]
            nouns_str = row["all_nouns"] if isinstance(row, sqlite3.Row) else row[2]

            for v in verbs_str.split(","):
                v = v.strip()
                if v:
                    self._verb_to_chain_ids[v].add(chain_id)
            for n in nouns_str.split(","):
                n = n.strip()
                if n:
                    self._noun_to_chain_ids[n].add(chain_id)

    def _index_chain(self, chain: VerbChain) -> None:
        """転置インデックスにチェーンを追加."""
        for step in chain.steps:
            self._verb_to_chain_ids[step.verb].add(chain.id)
            for noun in step.nouns:
                self._noun_to_chain_ids[noun].add(chain.id)

    async def save(self, chain: VerbChain) -> VerbChain:
        """チェーンをSQLiteに保存."""
        document = chain.to_document()
        meta = chain.to_metadata()

        def _insert() -> None:
            self._db.execute(
                """INSERT OR IGNORE INTO verb_chains
                   (id, document, steps_json, all_verbs, all_nouns,
                    timestamp, emotion, importance, source, context)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    chain.id,
                    document,
                    meta["steps_json"],
                    meta["all_verbs"],
                    meta["all_nouns"],
                    meta["timestamp"],
                    meta["emotion"],
                    meta["importance"],
                    meta["source"],
                    meta["context"],
                ),
            )
            # 埋め込みを保存
            vec = self._embedding_fn([document])[0]
            vec_bytes = encode_vector(vec)
            self._db.execute(
                "INSERT OR IGNORE INTO verb_chain_embeddings (chain_id, vector) VALUES (?,?)",
                (chain.id, vec_bytes),
            )
            self._db.commit()

        await asyncio.to_thread(_insert)
        self._index_chain(chain)

        # Register passive edges in graph
        if self._graph is not None:
            verbs = [step.verb for step in chain.steps]
            nouns_per_step = [list(step.nouns) for step in chain.steps]
            await self._graph.register_chain(verbs, nouns_per_step)

        return chain

    async def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[tuple[VerbChain, float]]:
        """意味的類似度でチェーンを検索（スコアリング付き）."""

        def _search_db() -> list[tuple[VerbChain, float]]:
            # 全チェーン+埋め込みを取得
            rows = self._db.execute(
                """SELECT vc.id, vc.document, vc.steps_json, vc.all_verbs, vc.all_nouns,
                          vc.timestamp, vc.emotion, vc.importance, vc.source, vc.context,
                          vce.vector
                   FROM verb_chains vc
                   JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id"""
            ).fetchall()

            if not rows:
                return []

            # クエリ埋め込み
            query_vec = np.array(self._embedding_fn.encode_query([query])[0])

            # チェーンとベクトルを分離
            chains: list[VerbChain] = []
            vecs: list[np.ndarray] = []
            for row in rows:
                chain_id = row[0]
                metadata = {
                    "steps_json": row[2],
                    "all_verbs": row[3],
                    "all_nouns": row[4],
                    "timestamp": row[5],
                    "emotion": row[6],
                    "importance": row[7],
                    "source": row[8],
                    "context": row[9],
                }
                chains.append(VerbChain.from_metadata(chain_id, metadata))
                vecs.append(decode_vector(row[10]))

            # バッチ cosine similarity
            corpus = np.stack(vecs)
            sims = cosine_similarity(query_vec, corpus)

            scored: list[tuple[VerbChain, float]] = []
            now = datetime.now(timezone.utc)

            for i, chain in enumerate(chains):
                semantic_distance = 1.0 - float(sims[i])
                time_decay = calculate_time_decay(chain.timestamp, now)
                emotion_boost = calculate_emotion_boost(chain.emotion)
                importance_boost = calculate_importance_boost(chain.importance)
                final_score = calculate_final_score(
                    semantic_distance=semantic_distance,
                    time_decay=time_decay,
                    emotion_boost=emotion_boost,
                    importance_boost=importance_boost,
                )
                scored.append((chain, final_score))

            scored.sort(key=lambda x: x[1])
            return scored[:n_results]

        return await asyncio.to_thread(_search_db)

    async def find_by_verb(self, verb: str) -> list[VerbChain]:
        """転置インデックスで動詞からチェーンを検索."""
        chain_ids = list(self._verb_to_chain_ids.get(verb, set()))
        if not chain_ids:
            return []
        return await self._get_chains_by_ids(chain_ids)

    async def find_by_noun(self, noun: str) -> list[VerbChain]:
        """転置インデックスで名詞からチェーンを検索."""
        chain_ids = list(self._noun_to_chain_ids.get(noun, set()))
        if not chain_ids:
            return []
        return await self._get_chains_by_ids(chain_ids)

    async def bump_chain_edges(self, chain: VerbChain) -> None:
        """Reinforce graph edges when a chain is recalled (active weights)."""
        if self._graph is None:
            return
        verbs = [step.verb for step in chain.steps]
        nouns_per_step = [list(step.nouns) for step in chain.steps]
        await self._graph.register_chain(verbs, nouns_per_step, delta_override=WEIGHTS_RECALL)

    async def expand_from_fragment(
        self,
        verb: str | None = None,
        noun: str | None = None,
        depth: int = 2,
        n_results: int = 20,
    ) -> list[VerbChain]:
        """1つの動詞/名詞から関連チェーンを発見.

        グラフがある場合: weight上位のノードをdepth段展開し、
        到達したノードから転置インデックスでチェーンID収集、weight順にスコアリング。
        グラフがない場合: 旧実装（芋づる式全展開）にフォールバック。
        """
        if self._graph is None:
            return await self._expand_from_fragment_legacy(verb=verb, noun=noun, depth=depth)

        depth = max(1, min(5, depth))

        # Graph-based expansion: collect nodes by following weighted edges
        visited_nodes: set[tuple[str, str]] = set()  # (type, surface_form)
        # Each node has an accumulated score from graph traversal
        node_scores: dict[tuple[str, str], float] = {}

        # Seed nodes
        seeds: list[tuple[str, str]] = []
        if verb:
            seeds.append(("verb", verb))
        if noun:
            seeds.append(("noun", noun))

        current_nodes = seeds
        for node in current_nodes:
            visited_nodes.add(node)
            node_scores[node] = 1.0  # seed nodes have score 1.0

        for _ in range(depth):
            if not current_nodes:
                break
            next_nodes: list[tuple[str, str]] = []
            for node_type, surface_form in current_nodes:
                neighbors = await self._graph.query_neighbors(
                    node_type, surface_form, limit=10
                )
                parent_score = node_scores.get((node_type, surface_form), 0.0)
                for n_type, n_form, weight in neighbors:
                    key = (n_type, n_form)
                    propagated = parent_score * weight
                    if key not in visited_nodes:
                        visited_nodes.add(key)
                        node_scores[key] = propagated
                        next_nodes.append(key)
                    else:
                        # Update score if this path is stronger
                        if propagated > node_scores.get(key, 0.0):
                            node_scores[key] = propagated
            current_nodes = next_nodes

        # Collect chain IDs from reached nodes via inverted index
        chain_id_scores: dict[str, float] = {}
        for (n_type, n_form), score in node_scores.items():
            if n_type == "verb":
                for cid in self._verb_to_chain_ids.get(n_form, set()):
                    chain_id_scores[cid] = max(chain_id_scores.get(cid, 0.0), score)
            elif n_type == "noun":
                for cid in self._noun_to_chain_ids.get(n_form, set()):
                    chain_id_scores[cid] = max(chain_id_scores.get(cid, 0.0), score)

        if not chain_id_scores:
            return []

        # Sort by score descending, take top n_results
        sorted_ids = sorted(chain_id_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in sorted_ids[:n_results]]

        return await self._get_chains_by_ids(top_ids)

    async def _expand_from_fragment_legacy(
        self,
        verb: str | None = None,
        noun: str | None = None,
        depth: int = 2,
    ) -> list[VerbChain]:
        """Legacy: 転置インデックスのみの芋づる式展開（グラフなし時のフォールバック）."""
        depth = max(1, min(5, depth))
        visited_chain_ids: set[str] = set()
        result_chains: list[VerbChain] = []

        # 初期シードとなるチェーンID群を収集
        seed_ids: set[str] = set()
        if verb:
            seed_ids.update(self._verb_to_chain_ids.get(verb, set()))
        if noun:
            seed_ids.update(self._noun_to_chain_ids.get(noun, set()))

        current_ids = seed_ids

        for _ in range(depth):
            if not current_ids:
                break

            # チェーンを取得
            new_ids = current_ids - visited_chain_ids
            if not new_ids:
                break

            chains = await self._get_chains_by_ids(list(new_ids))
            visited_chain_ids.update(new_ids)
            result_chains.extend(chains)

            # 次の展開: 取得したチェーンの動詞・名詞から新しいチェーンIDを収集
            next_ids: set[str] = set()
            for chain in chains:
                for step in chain.steps:
                    next_ids.update(self._verb_to_chain_ids.get(step.verb, set()))
                    for n in step.nouns:
                        next_ids.update(self._noun_to_chain_ids.get(n, set()))

            current_ids = next_ids - visited_chain_ids

        return result_chains

    async def _get_chains_by_ids(self, chain_ids: list[str]) -> list[VerbChain]:
        """IDリストからチェーンを取得."""
        if not chain_ids:
            return []

        def _fetch() -> list[VerbChain]:
            placeholders = ",".join("?" for _ in chain_ids)
            rows = self._db.execute(
                f"""SELECT id, document, steps_json, all_verbs, all_nouns,
                           timestamp, emotion, importance, source, context
                    FROM verb_chains WHERE id IN ({placeholders})""",
                chain_ids,
            ).fetchall()

            chains: list[VerbChain] = []
            for row in rows:
                metadata = {
                    "steps_json": row[1] if isinstance(row, tuple) else row["steps_json"],
                    "all_verbs": row[3] if isinstance(row, tuple) else row["all_verbs"],
                    "all_nouns": row[4] if isinstance(row, tuple) else row["all_nouns"],
                    "timestamp": row[5] if isinstance(row, tuple) else row["timestamp"],
                    "emotion": row[6] if isinstance(row, tuple) else row["emotion"],
                    "importance": row[7] if isinstance(row, tuple) else row["importance"],
                    "source": row[8] if isinstance(row, tuple) else row["source"],
                    "context": row[9] if isinstance(row, tuple) else row["context"],
                }
                chain_id = row[0] if isinstance(row, tuple) else row["id"]
                chains.append(VerbChain.from_metadata(chain_id, metadata))
            return chains

        return await asyncio.to_thread(_fetch)

    async def get_all(self) -> list[VerbChain]:
        """全チェーンを取得."""

        def _fetch_all() -> list[VerbChain]:
            rows = self._db.execute(
                """SELECT id, document, steps_json, all_verbs, all_nouns,
                          timestamp, emotion, importance, source, context
                   FROM verb_chains"""
            ).fetchall()

            chains: list[VerbChain] = []
            for row in rows:
                metadata = {
                    "steps_json": row[2] if isinstance(row, tuple) else row["steps_json"],
                    "all_verbs": row[3] if isinstance(row, tuple) else row["all_verbs"],
                    "all_nouns": row[4] if isinstance(row, tuple) else row["all_nouns"],
                    "timestamp": row[5] if isinstance(row, tuple) else row["timestamp"],
                    "emotion": row[6] if isinstance(row, tuple) else row["emotion"],
                    "importance": row[7] if isinstance(row, tuple) else row["importance"],
                    "source": row[8] if isinstance(row, tuple) else row["source"],
                    "context": row[9] if isinstance(row, tuple) else row["context"],
                }
                chain_id = row[0] if isinstance(row, tuple) else row["id"]
                chains.append(VerbChain.from_metadata(chain_id, metadata))
            return chains

        return await asyncio.to_thread(_fetch_all)


def crystallize_buffer(
    entries: list[dict[str, Any]],
    emotion: str = "neutral",
    importance: int = 3,
    min_verbs: int = 2,
) -> list[VerbChain]:
    """sensory_bufferエントリ群からVerbChainを生成.

    連続するエントリ間で共有名詞があれば同じチェーンにグループ化。
    共有名詞がなくなったら新しいチェーンを開始。

    Args:
        entries: sensory_buffer.jsonlのパース済みエントリリスト
        emotion: チェーンに付与する感情
        importance: チェーンに付与する重要度
        min_verbs: この数未満のステップのチェーンは破棄

    Returns:
        生成されたVerbChainリスト
    """
    if not entries:
        return []

    # エントリをVerbStep群にマップ（動詞がないエントリはスキップ）
    steps_with_nouns: list[tuple[list[VerbStep], set[str]]] = []
    for entry in entries:
        verbs = entry.get("v", [])
        nouns = entry.get("w", [])
        if not verbs:
            continue
        entry_steps = [VerbStep(verb=v, nouns=tuple(nouns)) for v in verbs]
        steps_with_nouns.append((entry_steps, set(nouns)))

    if not steps_with_nouns:
        return []

    # 共有名詞でグループ化
    groups: list[list[tuple[list[VerbStep], set[str]]]] = []
    current_group: list[tuple[list[VerbStep], set[str]]] = [steps_with_nouns[0]]

    for i in range(1, len(steps_with_nouns)):
        prev_nouns = current_group[-1][1]
        curr_nouns = steps_with_nouns[i][1]
        if prev_nouns & curr_nouns:  # 共有名詞あり → 同じグループ
            current_group.append(steps_with_nouns[i])
        else:
            groups.append(current_group)
            current_group = [steps_with_nouns[i]]
    groups.append(current_group)

    # グループをVerbChainに変換
    chains: list[VerbChain] = []
    timestamp = datetime.now(timezone.utc).isoformat()

    for group in groups:
        all_steps: list[VerbStep] = []
        for entry_steps, _ in group:
            all_steps.extend(entry_steps)

        if len(all_steps) < min_verbs:
            continue

        chain = VerbChain(
            id=str(uuid.uuid4()),
            steps=tuple(all_steps),
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            source="buffer",
            context="",
        )
        chains.append(chain)

    return chains
