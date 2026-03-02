"""VerbChainStore: 動詞チェーン記憶の保存・検索・連想展開 (SQLite backend)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .chive import ChiVeEmbedding
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
        chive: ChiVeEmbedding,
        graph: MemoryGraph | None = None,
    ):
        self._db = db
        self._chive = chive
        self._graph = graph
        # 転置インデックス（インメモリ）
        self._noun_to_chain_ids: dict[str, set[str]] = defaultdict(set)
        self._verb_to_chain_ids: dict[str, set[str]] = defaultdict(set)
        self._bigram_to_chain_ids: dict[str, set[str]] = defaultdict(set)

    async def initialize(self) -> None:
        """起動時に転置インデックスを復元（metaテーブルから永続化済みデータを使用）."""
        import logging

        logger = logging.getLogger(__name__)

        def _load_persisted_index() -> tuple[dict | None, int, int]:
            """metaテーブルから永続化済みインデックスを読み込む."""
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = 'verb_chain_inverted_index'"
            ).fetchone()
            if row is None:
                return None, 0, 0

            try:
                data = json.loads(row[0])
                saved_chain_count = data.get("chain_count", 0)
                current_chain_count = self._db.execute(
                    "SELECT COUNT(*) FROM verb_chains"
                ).fetchone()[0]
                return data, saved_chain_count, current_chain_count
            except (json.JSONDecodeError, KeyError, TypeError):
                return None, 0, 0

        data, saved_count, current_count = await asyncio.to_thread(_load_persisted_index)

        if data is not None:
            # Restore from persisted JSON
            verb_index = data.get("verb_to_chain_ids", {})
            noun_index = data.get("noun_to_chain_ids", {})
            bigram_index = data.get("bigram_to_chain_ids", {})
            for v, ids in verb_index.items():
                self._verb_to_chain_ids[v] = set(ids)
            for n, ids in noun_index.items():
                self._noun_to_chain_ids[n] = set(ids)
            for bg, ids in bigram_index.items():
                self._bigram_to_chain_ids[bg] = set(ids)

            logger.info(
                "VerbChainStore: restored inverted index from meta "
                "(saved_chains=%d, current_chains=%d)",
                saved_count, current_count,
            )

            # Check for new chains added since last persist
            if current_count > saved_count:
                indexed_ids: set[str] = set()
                for ids in self._verb_to_chain_ids.values():
                    indexed_ids.update(ids)
                for ids in self._noun_to_chain_ids.values():
                    indexed_ids.update(ids)

                def _find_new_chains() -> list[sqlite3.Row]:
                    rows = self._db.execute(
                        "SELECT id, all_verbs, all_nouns, steps_json FROM verb_chains"
                    ).fetchall()
                    return [
                        r for r in rows
                        if (r["id"] if isinstance(r, sqlite3.Row) else r[0])
                        not in indexed_ids
                    ]

                new_rows = await asyncio.to_thread(_find_new_chains)
                for row in new_rows:
                    chain_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
                    verbs_str = row["all_verbs"] if isinstance(row, sqlite3.Row) else row[1]
                    nouns_str = row["all_nouns"] if isinstance(row, sqlite3.Row) else row[2]
                    steps_json = row["steps_json"] if isinstance(row, sqlite3.Row) else row[3]

                    for v in verbs_str.split(","):
                        v = v.strip()
                        if v:
                            self._verb_to_chain_ids[v].add(chain_id)
                    for n in nouns_str.split(","):
                        n = n.strip()
                        if n:
                            self._noun_to_chain_ids[n].add(chain_id)
                    self._index_bigrams_from_json(chain_id, steps_json)

                if new_rows:
                    logger.info(
                        "VerbChainStore: indexed %d new chains incrementally",
                        len(new_rows),
                    )
                    await self._persist_index()
        else:
            # No persisted data - full build from DB
            logger.info("VerbChainStore: no persisted index found, building from scratch")

            rows = await asyncio.to_thread(
                lambda: self._db.execute(
                    "SELECT id, all_verbs, all_nouns, steps_json FROM verb_chains"
                ).fetchall()
            )

            for row in rows:
                chain_id = row["id"] if isinstance(row, sqlite3.Row) else row[0]
                verbs_str = row["all_verbs"] if isinstance(row, sqlite3.Row) else row[1]
                nouns_str = row["all_nouns"] if isinstance(row, sqlite3.Row) else row[2]
                steps_json = row["steps_json"] if isinstance(row, sqlite3.Row) else row[3]

                for v in verbs_str.split(","):
                    v = v.strip()
                    if v:
                        self._verb_to_chain_ids[v].add(chain_id)
                for n in nouns_str.split(","):
                    n = n.strip()
                    if n:
                        self._noun_to_chain_ids[n].add(chain_id)
                self._index_bigrams_from_json(chain_id, steps_json)

            if rows:
                await self._persist_index()
                logger.info(
                    "VerbChainStore: built and persisted index for %d chains",
                    len(rows),
                )

        # フローベクトルのバックフィル
        await self._backfill_flow_vectors()

    async def _backfill_flow_vectors(self) -> None:
        """flow_vector + delta_vector のバックフィル（chiVe 2ベクトル）."""
        import logging
        logger = logging.getLogger(__name__)

        FLOW_VERSION = "4"  # chiVe 2-vector version

        def _check_migration_needed() -> bool:
            row = self._db.execute(
                "SELECT value FROM meta WHERE key = 'flow_vector_version'"
            ).fetchone()
            return row is None or row[0] != FLOW_VERSION

        needs_migration = await asyncio.to_thread(_check_migration_needed)

        if needs_migration:
            def _null_all():
                self._db.execute("UPDATE verb_chain_embeddings SET flow_vector = NULL, delta_vector = NULL")
                self._db.commit()
            await asyncio.to_thread(_null_all)
            logger.info("VerbChainStore: migrating all flow/delta vectors (v%s)", FLOW_VERSION)

        def _find_missing() -> list[tuple[str, str, str]]:
            """chain_id, all_verbs, all_nouns のタプルを返す."""
            rows = self._db.execute(
                """SELECT vc.id, vc.all_verbs, vc.all_nouns
                   FROM verb_chains vc
                   JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id
                   WHERE vce.flow_vector IS NULL"""
            ).fetchall()
            return [(r[0], r[1] or "", r[2] or "") for r in rows]

        missing = await asyncio.to_thread(_find_missing)
        if not missing:
            if needs_migration:
                def _set_version():
                    self._db.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES ('flow_vector_version', ?)",
                        (FLOW_VERSION,),
                    )
                    self._db.commit()
                await asyncio.to_thread(_set_version)
            return

        logger.info("VerbChainStore: backfilling flow/delta vectors for %d chains", len(missing))

        def _update_flow_vectors() -> int:
            count = 0
            for chain_id, verbs_str, nouns_str in missing:
                verbs = [v.strip() for v in verbs_str.split(",") if v.strip()]
                nouns = [n.strip() for n in nouns_str.split(",") if n.strip()]
                if not verbs:
                    continue
                flow_vec, delta_vec = self._chive.encode_chain(verbs, nouns)
                concat_vec = np.concatenate([flow_vec, delta_vec])
                self._db.execute(
                    "UPDATE verb_chain_embeddings SET vector = ?, flow_vector = ?, delta_vector = ? WHERE chain_id = ?",
                    (encode_vector(concat_vec), encode_vector(flow_vec), encode_vector(delta_vec), chain_id),
                )
                count += 1
            self._db.commit()
            return count

        count = await asyncio.to_thread(_update_flow_vectors)

        def _set_version_final():
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('flow_vector_version', ?)",
                (FLOW_VERSION,),
            )
            self._db.commit()
        await asyncio.to_thread(_set_version_final)
        logger.info("VerbChainStore: backfilled %d flow/delta vectors (v%s)", count, FLOW_VERSION)

    async def _persist_index(self) -> None:
        """転置インデックスをmetaテーブルにJSON保存."""
        def _save() -> None:
            chain_count = self._db.execute(
                "SELECT COUNT(*) FROM verb_chains"
            ).fetchone()[0]

            data = {
                "verb_to_chain_ids": {
                    k: sorted(v) for k, v in self._verb_to_chain_ids.items()
                },
                "noun_to_chain_ids": {
                    k: sorted(v) for k, v in self._noun_to_chain_ids.items()
                },
                "bigram_to_chain_ids": {
                    k: sorted(v) for k, v in self._bigram_to_chain_ids.items()
                },
                "chain_count": chain_count,
            }
            self._db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('verb_chain_inverted_index', ?)",
                (json.dumps(data, ensure_ascii=False),),
            )
            self._db.commit()

        await asyncio.to_thread(_save)

    def _index_chain(self, chain: VerbChain) -> None:
        """転置インデックスにチェーンを追加."""
        for step in chain.steps:
            self._verb_to_chain_ids[step.verb].add(chain.id)
            for noun in step.nouns:
                self._noun_to_chain_ids[noun].add(chain.id)
        for i in range(len(chain.steps) - 1):
            bigram_key = f"{chain.steps[i].verb}→{chain.steps[i+1].verb}"
            self._bigram_to_chain_ids[bigram_key].add(chain.id)

    def _index_bigrams_from_json(self, chain_id: str, steps_json: str) -> None:
        """steps_json文字列からバイグラムを抽出してインデックスに追加."""
        try:
            steps_raw = json.loads(steps_json)
            verbs = [s.get("verb", "") for s in steps_raw if s.get("verb")]
            for i in range(len(verbs) - 1):
                bigram_key = f"{verbs[i]}→{verbs[i+1]}"
                self._bigram_to_chain_ids[bigram_key].add(chain_id)
        except (json.JSONDecodeError, TypeError):
            pass

    async def save(self, chain: VerbChain, category_id: int | None = None) -> VerbChain:
        """チェーンをSQLiteに保存（chiVe 2ベクトル）."""
        document = chain.to_document()
        meta = chain.to_metadata()

        # 2ベクトル計算
        verb_texts = [step.verb for step in chain.steps]
        noun_texts = list({n for step in chain.steps for n in step.nouns})
        flow_vec, delta_vec = self._chive.encode_chain(verb_texts, noun_texts)
        concat_vec = np.concatenate([flow_vec, delta_vec])

        def _insert() -> None:
            # Decay freshness only for post-consolidation memories
            cutoff = self._db.execute(
                "SELECT COALESCE((SELECT value FROM meta WHERE key = 'last_consolidated_at'), '')"
            ).fetchone()[0]
            if cutoff:
                self._db.execute(
                    "UPDATE memories SET freshness = MAX(0.01, freshness - 0.003) WHERE timestamp > ?",
                    (cutoff,),
                )
                self._db.execute(
                    "UPDATE verb_chains SET freshness = MAX(0.01, freshness - 0.003) WHERE timestamp > ?",
                    (cutoff,),
                )
            else:
                self._db.execute("UPDATE memories SET freshness = MAX(0.01, freshness - 0.003)")
                self._db.execute("UPDATE verb_chains SET freshness = MAX(0.01, freshness - 0.003)")
            self._db.execute(
                """INSERT OR IGNORE INTO verb_chains
                   (id, document, steps_json, all_verbs, all_nouns,
                    timestamp, emotion, importance, source, context, category_id, freshness)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    category_id,
                    1.0,
                ),
            )
            vec_bytes = encode_vector(concat_vec)
            flow_bytes = encode_vector(flow_vec)
            delta_bytes = encode_vector(delta_vec)
            self._db.execute(
                "INSERT OR IGNORE INTO verb_chain_embeddings (chain_id, vector, flow_vector, delta_vector) VALUES (?,?,?,?)",
                (chain.id, vec_bytes, flow_bytes, delta_bytes),
            )
            self._db.commit()

        await asyncio.to_thread(_insert)
        self._index_chain(chain)

        # Persist updated inverted index
        await self._persist_index()

        # Register passive edges in graph
        if self._graph is not None:
            verbs = [step.verb for step in chain.steps]
            nouns_per_step = [list(step.nouns) for step in chain.steps]
            await self._graph.register_chain(verbs, nouns_per_step)

            if category_id is not None:
                await self._graph.assign_chain_nodes_to_category(
                    verbs, nouns_per_step, category_id
                )

        return chain

    async def search(
        self,
        query: str,
        n_results: int = 5,
        category_id: int | None = None,
        flow_weight: float = 0.6,
    ) -> list[tuple[VerbChain, float]]:
        """2軸コサイン類似度でチェーンを検索（スコアリング付き）.

        flow_weight: flow軸の重み (0.0-1.0)。残りがdelta軸。
        """

        def _search_db() -> list[tuple[VerbChain, float]]:
            # チェーン+埋め込みを取得
            if category_id is not None:
                rows = self._db.execute(
                    """WITH RECURSIVE cat_tree(id) AS (
                           SELECT id FROM categories WHERE id = ?
                           UNION ALL
                           SELECT c.id FROM categories c
                           JOIN cat_tree ct ON c.parent_id = ct.id
                       )
                       SELECT vc.id, vc.document, vc.steps_json, vc.all_verbs, vc.all_nouns,
                              vc.timestamp, vc.emotion, vc.importance, vc.source, vc.context,
                              vce.vector, vc.freshness, vce.flow_vector, vce.delta_vector
                       FROM verb_chains vc
                       JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id
                       WHERE vc.category_id IN (SELECT id FROM cat_tree)""",
                    (category_id,),
                ).fetchall()
            else:
                rows = self._db.execute(
                    """SELECT vc.id, vc.document, vc.steps_json, vc.all_verbs, vc.all_nouns,
                              vc.timestamp, vc.emotion, vc.importance, vc.source, vc.context,
                              vce.vector, vc.freshness, vce.flow_vector, vce.delta_vector
                       FROM verb_chains vc
                       JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id"""
                ).fetchall()

            if not rows:
                return []

            # クエリ 2ベクトル
            q_flow, q_delta = self._chive.encode_text(query)

            # チェーンとベクトルを分離
            chains: list[VerbChain] = []
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
                    "freshness": row[11] if len(row) > 11 else 1.0,
                }
                chains.append(VerbChain.from_metadata(chain_id, metadata))

            scored: list[tuple[VerbChain, float]] = []
            now = datetime.now(timezone.utc)

            for i, chain in enumerate(chains):
                row = rows[i]
                flow_blob = row[12] if len(row) > 12 else None
                delta_blob = row[13] if len(row) > 13 else None

                if flow_blob and delta_blob:
                    m_flow = decode_vector(bytes(flow_blob))
                    m_delta = decode_vector(bytes(delta_blob))
                    flow_sim = float(cosine_similarity(q_flow, m_flow.reshape(1, -1))[0])
                    delta_sim = float(cosine_similarity(q_delta, m_delta.reshape(1, -1))[0])
                    sim = flow_weight * flow_sim + (1.0 - flow_weight) * delta_sim
                else:
                    # Legacy fallback (skip if dimension mismatch)
                    legacy_vec = decode_vector(bytes(row[10]))
                    q_concat = np.concatenate([q_flow, q_delta])
                    if legacy_vec.shape[0] != q_concat.shape[0]:
                        continue
                    sim = float(cosine_similarity(q_concat, legacy_vec.reshape(1, -1))[0])

                semantic_distance = 1.0 - sim
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

    async def find_by_bigram(self, verb1: str, verb2: str) -> list[VerbChain]:
        """バイグラムインデックスで動詞ペアからチェーンを検索."""
        bigram_key = f"{verb1}→{verb2}"
        chain_ids = list(self._bigram_to_chain_ids.get(bigram_key, set()))
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
        verb2: str | None = None,
        depth: int = 2,
        n_results: int = 20,
        category_id: int | None = None,
    ) -> tuple[list[VerbChain], list[str], list[str]]:
        """1つの動詞/名詞から関連チェーンを発見."""
        if self._graph is None:
            return await self._expand_from_fragment_legacy(verb=verb, noun=noun, depth=depth)

        depth = max(1, min(5, depth))

        visited_nodes: set[tuple[str, str]] = set()
        node_scores: dict[tuple[str, str], float] = {}

        seeds: list[tuple[str, str]] = []
        if verb:
            seeds.append(("verb", verb))
        if verb2:
            seeds.append(("verb", verb2))
        if noun:
            seeds.append(("noun", noun))

        bigram_chain_ids: set[str] = set()
        if verb and verb2:
            bigram_key = f"{verb}→{verb2}"
            bigram_chain_ids = set(self._bigram_to_chain_ids.get(bigram_key, set()))

        current_nodes = seeds
        for node in current_nodes:
            visited_nodes.add(node)
            node_scores[node] = 1.0

        for _ in range(depth):
            if not current_nodes:
                break
            next_nodes: list[tuple[str, str]] = []
            for node_type, surface_form in current_nodes:
                neighbors = await self._graph.query_neighbors(
                    node_type, surface_form, limit=10,
                    category_id=category_id,
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
                        if propagated > node_scores.get(key, 0.0):
                            node_scores[key] = propagated
            current_nodes = next_nodes

        chain_id_scores: dict[str, float] = {}
        for (n_type, n_form), score in node_scores.items():
            if n_type == "verb":
                for cid in self._verb_to_chain_ids.get(n_form, set()):
                    chain_id_scores[cid] = max(chain_id_scores.get(cid, 0.0), score)
            elif n_type == "noun":
                for cid in self._noun_to_chain_ids.get(n_form, set()):
                    chain_id_scores[cid] = max(chain_id_scores.get(cid, 0.0), score)

        BIGRAM_BONUS = 1.5
        for cid in bigram_chain_ids:
            chain_id_scores[cid] = max(
                chain_id_scores.get(cid, 0.0), BIGRAM_BONUS
            )

        if not chain_id_scores:
            return [], [], []

        sorted_ids = sorted(chain_id_scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = [cid for cid, _ in sorted_ids[:n_results]]

        chains = await self._get_chains_by_ids(top_ids)

        if category_id is not None:
            cat_ids = self._get_descendant_category_ids(category_id)
            chains = [c for c in chains if self._chain_has_category(c.id, cat_ids)]

        v_verbs = [sf for t, sf in visited_nodes if t == "verb"]
        v_nouns = [sf for t, sf in visited_nodes if t == "noun"]
        return chains, v_verbs, v_nouns

    def _get_descendant_category_ids(self, category_id: int) -> set[int]:
        rows = self._db.execute(
            """WITH RECURSIVE cat_tree(id) AS (
                   SELECT id FROM categories WHERE id = ?
                   UNION ALL
                   SELECT c.id FROM categories c
                   JOIN cat_tree ct ON c.parent_id = ct.id
               )
               SELECT id FROM cat_tree""",
            (category_id,),
        ).fetchall()
        return {int(r[0]) if isinstance(r, tuple) else int(r["id"]) for r in rows}

    def _chain_has_category(self, chain_id: str, cat_ids: set[int]) -> bool:
        row = self._db.execute(
            "SELECT category_id FROM verb_chains WHERE id = ?",
            (chain_id,),
        ).fetchone()
        if row is None:
            return False
        cid = row[0] if isinstance(row, tuple) else row["category_id"]
        return cid is not None and int(cid) in cat_ids

    async def _expand_from_fragment_legacy(
        self,
        verb: str | None = None,
        noun: str | None = None,
        depth: int = 2,
    ) -> tuple[list[VerbChain], list[str], list[str]]:
        """Legacy: 転置インデックスのみの芋づる式展開."""
        depth = max(1, min(5, depth))
        visited_chain_ids: set[str] = set()
        result_chains: list[VerbChain] = []

        seed_ids: set[str] = set()
        if verb:
            seed_ids.update(self._verb_to_chain_ids.get(verb, set()))
        if noun:
            seed_ids.update(self._noun_to_chain_ids.get(noun, set()))

        current_ids = seed_ids

        for _ in range(depth):
            if not current_ids:
                break
            new_ids = current_ids - visited_chain_ids
            if not new_ids:
                break
            chains = await self._get_chains_by_ids(list(new_ids))
            visited_chain_ids.update(new_ids)
            result_chains.extend(chains)
            next_ids: set[str] = set()
            for chain in chains:
                for step in chain.steps:
                    next_ids.update(self._verb_to_chain_ids.get(step.verb, set()))
                    for n in step.nouns:
                        next_ids.update(self._noun_to_chain_ids.get(n, set()))
            current_ids = next_ids - visited_chain_ids

        all_verbs: set[str] = set()
        all_nouns: set[str] = set()
        if verb:
            all_verbs.add(verb)
        if noun:
            all_nouns.add(noun)
        for chain in result_chains:
            for step in chain.steps:
                all_verbs.add(step.verb)
                all_nouns.update(step.nouns)
        return result_chains, list(all_verbs), list(all_nouns)

    async def _get_chains_by_ids(self, chain_ids: list[str]) -> list[VerbChain]:
        """IDリストからチェーンを取得."""
        if not chain_ids:
            return []

        def _fetch() -> list[VerbChain]:
            placeholders = ",".join("?" for _ in chain_ids)
            rows = self._db.execute(
                f"""SELECT id, document, steps_json, all_verbs, all_nouns,
                           timestamp, emotion, importance, source, context, freshness
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
                    "freshness": (row[10] if isinstance(row, tuple) else row["freshness"]) or 1.0,
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
                          timestamp, emotion, importance, source, context, freshness
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
                    "freshness": (row[10] if isinstance(row, tuple) else row["freshness"]) or 1.0,
                }
                chain_id = row[0] if isinstance(row, tuple) else row["id"]
                chains.append(VerbChain.from_metadata(chain_id, metadata))
            return chains

        return await asyncio.to_thread(_fetch_all)


def crystallize_buffer(
    entries: list[dict[str, Any]],
    emotion: str = "8",
    importance: int = 3,
    min_verbs: int = 2,
    merge_threshold: float = 0.2,
) -> list[VerbChain]:
    """sensory_bufferエントリ群からVerbChainを生成."""
    if not entries:
        return []

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

    groups: list[list[tuple[list[VerbStep], set[str]]]] = []
    current_group: list[tuple[list[VerbStep], set[str]]] = [steps_with_nouns[0]]

    for i in range(1, len(steps_with_nouns)):
        prev_nouns = current_group[-1][1]
        curr_nouns = steps_with_nouns[i][1]
        shared = prev_nouns & curr_nouns
        smaller = min(len(prev_nouns), len(curr_nouns))
        if smaller > 0 and len(shared) / smaller >= merge_threshold:
            current_group.append(steps_with_nouns[i])
        else:
            groups.append(current_group)
            current_group = [steps_with_nouns[i]]
    groups.append(current_group)

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
