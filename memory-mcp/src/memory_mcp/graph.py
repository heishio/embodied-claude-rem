"""Weighted Memory Graph for verb/noun co-occurrence edges."""

from __future__ import annotations

import asyncio
import sqlite3

# Edge weights assigned when a chain is first saved (passive)
WEIGHTS_INITIAL = {"vv": 0.1, "vn": 0.08, "nn": 0.05}
# Edge weights added when a chain is recalled (active reinforcement)
WEIGHTS_RECALL = {"vv": 0.3, "vn": 0.2, "nn": 0.1}

DECAY_FACTOR = 0.8
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
    ) -> list[tuple[str, str, float]]:
        """Return (type, surface_form, weight) of neighbours sorted by weight DESC."""

        def _query() -> list[tuple[str, str, float]]:
            row = self._db.execute(
                "SELECT id FROM graph_nodes WHERE type = ? AND surface_form = ?",
                (node_type, surface_form),
            ).fetchone()
            if row is None:
                return []
            node_id = int(row[0]) if isinstance(row, tuple) else int(row["id"])

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

            # Normalize: pull outliers 10% toward mean per from_id
            rows = self._db.execute(
                """SELECT from_id, AVG(weight) as avg_w
                   FROM graph_edges GROUP BY from_id"""
            ).fetchall()
            for r in rows:
                from_id = r[0] if isinstance(r, tuple) else r["from_id"]
                avg_w = float(r[1] if isinstance(r, tuple) else r["avg_w"])
                # Pull each edge 10% toward the average
                self._db.execute(
                    """UPDATE graph_edges
                       SET weight = weight + 0.1 * (? - weight)
                       WHERE from_id = ?""",
                    (avg_w, from_id),
                )

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
