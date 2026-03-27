"""Weighted Memory Graph for verb/noun co-occurrence edges."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone

# Edge weights assigned when a chain is first saved (passive)
# High initial weights so new memories stand out; consolidation normalises fast
WEIGHTS_INITIAL = {"vv": 0.3, "vn": 0.2, "nn": 0.1}
# Edge weights added when a chain is recalled (active reinforcement)
WEIGHTS_RECALL = {"vv": 0.3, "vn": 0.2, "nn": 0.1}

DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.01


class MemoryGraph:
    """Weighted graph over verb/noun nodes stored in SQLite."""

    def __init__(self, db: sqlite3.Connection):
        self._db = db

    # ── Node helpers ─────────────────────────────

    def _get_or_create_node_sync(self, node_type: str, surface_form: str) -> int:
        self._db.execute(
            "INSERT OR IGNORE INTO graph_nodes (type, surface_form) VALUES (?, ?)",
            (node_type, surface_form),
        )
        row = self._db.execute(
            "SELECT id FROM graph_nodes WHERE type = ? AND surface_form = ?",
            (node_type, surface_form),
        ).fetchone()
        return int(row[0]) if isinstance(row, tuple) else int(row["id"])

    # ── Edge helpers ─────────────────────────────

    def _bump_edge_sync(
        self, from_id: int, to_id: int, link_type: str, delta: float
    ) -> None:
        row = self._db.execute(
            "SELECT weight FROM graph_edges WHERE from_id = ? AND to_id = ?",
            (from_id, to_id),
        ).fetchone()
        if row is None:
            new_weight = min(1.0, delta)
        else:
            current = float(row[0]) if isinstance(row, tuple) else float(row["weight"])
            new_weight = min(1.0, current + delta)
        self._db.execute(
            """INSERT INTO graph_edges (from_id, to_id, weight, link_type)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(from_id, to_id) DO UPDATE SET weight = excluded.weight""",
            (from_id, to_id, new_weight, link_type),
        )

    # ── Public API ───────────────────────────────

    async def register_chain(
        self,
        verbs: list[str],
        nouns_per_step: list[list[str]],
        delta_override: dict[str, float] | None = None,
    ) -> None:
        """Register edges from a verb chain (passive or active weights)."""
        weights = delta_override if delta_override is not None else WEIGHTS_INITIAL

        def _work() -> None:
            # vv edges: consecutive verb pairs
            verb_node_ids: list[int] = []
            for v in verbs:
                verb_node_ids.append(self._get_or_create_node_sync("verb", v))

            for i in range(len(verb_node_ids) - 1):
                self._bump_edge_sync(
                    verb_node_ids[i], verb_node_ids[i + 1], "vv", weights["vv"]
                )

            # vn and nn edges per step
            for i, nouns in enumerate(nouns_per_step):
                if i >= len(verb_node_ids):
                    break
                v_id = verb_node_ids[i]
                noun_ids: list[int] = []
                for n in nouns:
                    n_id = self._get_or_create_node_sync("noun", n)
                    noun_ids.append(n_id)
                    # vn: verb -> noun
                    self._bump_edge_sync(v_id, n_id, "vn", weights["vn"])

                # nn: noun <-> noun (bidirectional) within same step
                for a in range(len(noun_ids)):
                    for b in range(a + 1, len(noun_ids)):
                        self._bump_edge_sync(
                            noun_ids[a], noun_ids[b], "nn", weights["nn"]
                        )
                        self._bump_edge_sync(
                            noun_ids[b], noun_ids[a], "nn", weights["nn"]
                        )

            self._db.commit()

        await asyncio.to_thread(_work)

    async def query_neighbors(
        self,
        node_type: str,
        surface_form: str,
        limit: int = 20,
        category_id: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Return (type, surface_form, weight) of neighbours sorted by weight DESC.

        If category_id is given, only return neighbours whose nodes belong to
        that category or any of its descendant categories.
        """

        def _query() -> list[tuple[str, str, float]]:
            row = self._db.execute(
                "SELECT id FROM graph_nodes WHERE type = ? AND surface_form = ?",
                (node_type, surface_form),
            ).fetchone()
            if row is None:
                return []
            node_id = int(row[0]) if isinstance(row, tuple) else int(row["id"])

            if category_id is not None:
                # Use recursive CTE to get all descendant category IDs
                rows = self._db.execute(
                    """WITH RECURSIVE cat_tree(id) AS (
                           SELECT id FROM categories WHERE id = ?
                           UNION ALL
                           SELECT c.id FROM categories c
                           JOIN cat_tree ct ON c.parent_id = ct.id
                       )
                       SELECT n.type, n.surface_form, e.weight
                       FROM graph_edges e
                       JOIN graph_nodes n ON n.id = e.to_id
                       JOIN node_categories nc ON nc.node_id = e.to_id
                       WHERE e.from_id = ? AND nc.category_id IN (SELECT id FROM cat_tree)
                       ORDER BY e.weight DESC
                       LIMIT ?""",
                    (category_id, node_id, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    """SELECT n.type, n.surface_form, e.weight
                       FROM graph_edges e
                       JOIN graph_nodes n ON n.id = e.to_id
                       WHERE e.from_id = ?
                       ORDER BY e.weight DESC
                       LIMIT ?""",
                    (node_id, limit),
                ).fetchall()

            result: list[tuple[str, str, float]] = []
            for r in rows:
                if isinstance(r, tuple):
                    result.append((r[0], r[1], float(r[2])))
                else:
                    result.append((r["type"], r["surface_form"], float(r["weight"])))
            return result

        return await asyncio.to_thread(_query)

    # ── Category API ─────────────────────────────

    async def create_category(
        self, name: str, parent_id: int | None = None
    ) -> int:
        """Create a category and return its ID."""

        def _create() -> int:
            now = datetime.now(timezone.utc).isoformat()
            self._db.execute(
                "INSERT INTO categories (name, parent_id, created_at) VALUES (?, ?, ?)",
                (name, parent_id, now),
            )
            row = self._db.execute(
                "SELECT id FROM categories WHERE name = ? AND parent_id IS ?",
                (name, parent_id),
            ).fetchone()
            self._db.commit()
            return int(row[0]) if isinstance(row, tuple) else int(row["id"])

        return await asyncio.to_thread(_create)

    async def list_categories(self) -> list[dict]:
        """Return all categories as a flat list of dicts."""

        def _list() -> list[dict]:
            rows = self._db.execute(
                "SELECT id, name, parent_id, created_at FROM categories ORDER BY id"
            ).fetchall()
            result: list[dict] = []
            for r in rows:
                if isinstance(r, tuple):
                    result.append({"id": r[0], "name": r[1], "parent_id": r[2], "created_at": r[3]})
                else:
                    result.append({
                        "id": r["id"], "name": r["name"],
                        "parent_id": r["parent_id"], "created_at": r["created_at"],
                    })
            return result

        return await asyncio.to_thread(_list)

    async def get_category_node_ids(self, category_id: int) -> set[int]:
        """Return all node IDs belonging to a category or its descendants (recursive CTE)."""

        def _get() -> set[int]:
            rows = self._db.execute(
                """WITH RECURSIVE cat_tree(id) AS (
                       SELECT id FROM categories WHERE id = ?
                       UNION ALL
                       SELECT c.id FROM categories c
                       JOIN cat_tree ct ON c.parent_id = ct.id
                   )
                   SELECT nc.node_id FROM node_categories nc
                   WHERE nc.category_id IN (SELECT id FROM cat_tree)""",
                (category_id,),
            ).fetchall()
            return {int(r[0]) if isinstance(r, tuple) else int(r["node_id"]) for r in rows}

        return await asyncio.to_thread(_get)

    async def assign_node_category(self, node_id: int, category_id: int) -> None:
        """Assign a node to a category."""

        def _assign() -> None:
            self._db.execute(
                "INSERT OR IGNORE INTO node_categories (node_id, category_id) VALUES (?, ?)",
                (node_id, category_id),
            )
            self._db.commit()

        await asyncio.to_thread(_assign)

    async def assign_chain_nodes_to_category(
        self,
        verbs: list[str],
        nouns_per_step: list[list[str]],
        category_id: int,
    ) -> None:
        """Assign all nodes from a chain's verbs/nouns to a category."""

        def _assign() -> None:
            for v in verbs:
                row = self._db.execute(
                    "SELECT id FROM graph_nodes WHERE type = 'verb' AND surface_form = ?",
                    (v,),
                ).fetchone()
                if row:
                    nid = int(row[0]) if isinstance(row, tuple) else int(row["id"])
                    self._db.execute(
                        "INSERT OR IGNORE INTO node_categories (node_id, category_id) VALUES (?, ?)",
                        (nid, category_id),
                    )
            for nouns in nouns_per_step:
                for n in nouns:
                    row = self._db.execute(
                        "SELECT id FROM graph_nodes WHERE type = 'noun' AND surface_form = ?",
                        (n,),
                    ).fetchone()
                    if row:
                        nid = int(row[0]) if isinstance(row, tuple) else int(row["id"])
                        self._db.execute(
                            "INSERT OR IGNORE INTO node_categories (node_id, category_id) VALUES (?, ?)",
                            (nid, category_id),
                        )
            self._db.commit()

        await asyncio.to_thread(_assign)

    async def get_path_strength(
        self, verbs: list[str], nouns: list[str]
    ) -> float:
        """指定された動詞/名詞間のグラフエッジ重み合計を返す。"""

        def _calc() -> float:
            total = 0.0

            # Collect node IDs
            verb_ids: list[int] = []
            for v in verbs:
                row = self._db.execute(
                    "SELECT id FROM graph_nodes WHERE type = 'verb' AND surface_form = ?",
                    (v,),
                ).fetchone()
                if row:
                    verb_ids.append(int(row[0]) if isinstance(row, tuple) else int(row["id"]))

            noun_ids: list[int] = []
            for n in nouns:
                row = self._db.execute(
                    "SELECT id FROM graph_nodes WHERE type = 'noun' AND surface_form = ?",
                    (n,),
                ).fetchone()
                if row:
                    noun_ids.append(int(row[0]) if isinstance(row, tuple) else int(row["id"]))

            all_ids = verb_ids + noun_ids
            if len(all_ids) < 2:
                return 0.0

            # Sum weights of all edges between these nodes
            placeholders = ",".join("?" for _ in all_ids)
            rows = self._db.execute(
                f"""SELECT SUM(weight) FROM graph_edges
                    WHERE from_id IN ({placeholders}) AND to_id IN ({placeholders})""",
                all_ids + all_ids,
            ).fetchone()

            if rows and rows[0] is not None:
                total = float(rows[0])

            return total

        return await asyncio.to_thread(_calc)

    async def consolidate(
        self,
        decay_factor: float = DECAY_FACTOR,
        prune_threshold: float = PRUNE_THRESHOLD,
    ) -> dict[str, int]:
        """Decay all edges, normalize, and prune weak ones."""

        def _work() -> dict[str, int]:
            # Count before
            total_before = self._db.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0]

            # Decay
            self._db.execute(
                "UPDATE graph_edges SET weight = weight * ?", (decay_factor,)
            )

            # Normalize: pull each edge 10% toward per-from_id average
            self._db.execute(
                """CREATE TEMP TABLE _avg_weights AS
                   SELECT from_id, AVG(weight) AS avg_w FROM graph_edges GROUP BY from_id"""
            )
            self._db.execute(
                "CREATE INDEX _avg_weights_idx ON _avg_weights(from_id)"
            )
            self._db.execute(
                """UPDATE graph_edges
                   SET weight = weight + 0.1 * (
                       (SELECT avg_w FROM _avg_weights WHERE from_id = graph_edges.from_id) - weight
                   )"""
            )
            self._db.execute("DROP TABLE _avg_weights")

            # Prune
            self._db.execute(
                "DELETE FROM graph_edges WHERE weight < ?", (prune_threshold,)
            )

            # Prune orphan nodes (no edges)
            self._db.execute(
                """DELETE FROM graph_nodes WHERE id NOT IN (
                       SELECT from_id FROM graph_edges
                       UNION SELECT to_id FROM graph_edges
                   )"""
            )

            self._db.commit()

            total_after = self._db.execute(
                "SELECT COUNT(*) FROM graph_edges"
            ).fetchone()[0]

            return {
                "graph_decayed": total_before,
                "graph_pruned": total_before - total_after,
                "graph_remaining": total_after,
            }

        return await asyncio.to_thread(_work)
