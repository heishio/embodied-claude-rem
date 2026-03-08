"""SQLite + numpy backed memory storage (Phase 11: ChromaDB → SQLite+numpy)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .bm25 import BM25Index
from .chive import ChiVeEmbedding
from .config import MemoryConfig
from .consolidation import ConsolidationEngine
from .hopfield import HopfieldRecallResult, ModernHopfieldNetwork
from .normalizer import get_reading, normalize_japanese
from .predictive import (
    PredictiveDiagnostics,
    calculate_context_relevance,
    calculate_novelty_score,
    calculate_prediction_error,
)
from .scoring import EMOTION_BOOST_MAP as EMOTION_BOOST_MAP  # re-export
from .scoring import (
    calculate_emotion_boost,
    calculate_final_score,
    calculate_importance_boost,
    calculate_time_decay,
)
from .types import (
    CameraPosition,
    Episode,
    Memory,
    MemorySearchResult,
    MemoryStats,
    ScoredMemory,
    SensoryData,
)
from .vector import cosine_similarity, decode_vector, encode_vector
from .working_memory import WorkingMemoryBuffer
from .workspace import (
    WorkspaceCandidate,
    diversity_score,
    select_workspace_candidates,
)

# ──────────────────────────────────────────────
# DDL
# ──────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    emotion TEXT NOT NULL DEFAULT '8',
    importance INTEGER NOT NULL DEFAULT 3,
    category TEXT NOT NULL DEFAULT 'daily',
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT NOT NULL DEFAULT '',
    linked_ids TEXT NOT NULL DEFAULT '',
    episode_id TEXT,
    sensory_data TEXT NOT NULL DEFAULT '',
    camera_position TEXT,
    tags TEXT NOT NULL DEFAULT '',
    links TEXT NOT NULL DEFAULT '',
    novelty_score REAL NOT NULL DEFAULT 0.0,
    prediction_error REAL NOT NULL DEFAULT 0.0,
    activation_count INTEGER NOT NULL DEFAULT 0,
    last_activated TEXT NOT NULL DEFAULT '',
    reading TEXT,
    freshness REAL NOT NULL DEFAULT 1.0,
    level INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_emotion    ON memories(emotion);
CREATE INDEX IF NOT EXISTS idx_memories_category   ON memories(category);
CREATE INDEX IF NOT EXISTS idx_memories_timestamp  ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);

CREATE TABLE IF NOT EXISTS embeddings (
    memory_id TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    flow_vector BLOB,
    delta_vector BLOB
);

CREATE TABLE IF NOT EXISTS episodes (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT,
    memory_ids TEXT NOT NULL DEFAULT '',
    participants TEXT NOT NULL DEFAULT '',
    location_context TEXT,
    summary TEXT NOT NULL DEFAULT '',
    emotion TEXT NOT NULL DEFAULT '8',
    importance INTEGER NOT NULL DEFAULT 3
);

CREATE TABLE IF NOT EXISTS verb_chains (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    steps_json TEXT NOT NULL,
    all_verbs TEXT NOT NULL,
    all_nouns TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    emotion TEXT NOT NULL DEFAULT '8',
    importance INTEGER NOT NULL DEFAULT 3,
    source TEXT NOT NULL DEFAULT 'buffer',
    context TEXT NOT NULL DEFAULT '',
    freshness REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS verb_chain_embeddings (
    chain_id TEXT PRIMARY KEY REFERENCES verb_chains(id) ON DELETE CASCADE,
    vector BLOB NOT NULL,
    flow_vector BLOB,
    delta_vector BLOB
);

CREATE TABLE IF NOT EXISTS graph_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK(type IN ('verb', 'noun')),
    surface_form TEXT NOT NULL,
    UNIQUE(type, surface_form)
);

CREATE TABLE IF NOT EXISTS graph_edges (
    from_id INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    to_id INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    weight REAL NOT NULL DEFAULT 0.0,
    link_type TEXT NOT NULL CHECK(link_type IN ('vv', 'vn', 'nn')),
    PRIMARY KEY (from_id, to_id)
);
CREATE INDEX IF NOT EXISTS idx_graph_edges_from ON graph_edges(from_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_to ON graph_edges(to_id);
CREATE INDEX IF NOT EXISTS idx_graph_edges_weight ON graph_edges(weight);

CREATE TABLE IF NOT EXISTS categories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    parent_id INTEGER REFERENCES categories(id),
    created_at TEXT NOT NULL,
    UNIQUE(name, parent_id)
);

CREATE TABLE IF NOT EXISTS node_categories (
    node_id INTEGER NOT NULL REFERENCES graph_nodes(id) ON DELETE CASCADE,
    category_id INTEGER NOT NULL REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (node_id, category_id)
);

CREATE TABLE IF NOT EXISTS recall_index (
    word TEXT NOT NULL,
    target_id TEXT NOT NULL,
    target_type TEXT NOT NULL DEFAULT 'memory',
    similarity REAL NOT NULL,
    content_preview TEXT NOT NULL DEFAULT '',
    flow_sim REAL,
    delta_sim REAL,
    PRIMARY KEY (word, target_id, target_type)
);
CREATE INDEX IF NOT EXISTS idx_recall_word ON recall_index(word);

CREATE TABLE IF NOT EXISTS composite_members (
    composite_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    contribution_weight REAL DEFAULT 1.0,
    PRIMARY KEY (composite_id, member_id)
);
CREATE INDEX IF NOT EXISTS idx_composite_members_member ON composite_members(member_id);

CREATE TABLE IF NOT EXISTS boundary_layers (
    composite_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    layer_index INTEGER NOT NULL,
    is_edge INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (composite_id, member_id, layer_index)
);
CREATE INDEX IF NOT EXISTS idx_boundary_layers_composite ON boundary_layers(composite_id);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS template_biases (
    chain_id TEXT NOT NULL,
    bias_weight REAL NOT NULL DEFAULT 0.0,
    update_count INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL,
    PRIMARY KEY (chain_id)
);

CREATE TABLE IF NOT EXISTS composite_axes (
    composite_id TEXT PRIMARY KEY,
    axis_vector BLOB NOT NULL,
    explained_variance_ratio REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS composite_intersections (
    composite_a TEXT NOT NULL,
    composite_b TEXT NOT NULL,
    intersection_type TEXT NOT NULL CHECK(intersection_type IN ('parallel', 'transversal')),
    axis_cosine REAL NOT NULL DEFAULT 0.0,
    shared_member_ids TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (composite_a, composite_b)
);
CREATE INDEX IF NOT EXISTS idx_intersections_type ON composite_intersections(intersection_type);
"""

# ──────────────────────────────────────────────
# Row → Memory helpers
# ──────────────────────────────────────────────


def _parse_sensory_data(sensory_data_json: str) -> tuple[SensoryData, ...]:
    if not sensory_data_json:
        return ()
    try:
        data_list = json.loads(sensory_data_json)
        return tuple(SensoryData.from_dict(d) for d in data_list)
    except (json.JSONDecodeError, KeyError, TypeError):
        return ()


def _parse_camera_position(camera_position_json: str | None) -> CameraPosition | None:
    if not camera_position_json:
        return None
    try:
        data = json.loads(camera_position_json)
        return CameraPosition.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _parse_tags(tags_str: str) -> tuple[str, ...]:
    if not tags_str:
        return ()
    return tuple(tag.strip() for tag in tags_str.split(",") if tag.strip())


def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Convert a SQLite Row from the memories table to a Memory object."""
    episode_id_raw = row["episode_id"]
    episode_id = episode_id_raw if episode_id_raw else None

    return Memory(
        id=row["id"],
        content=row["content"],
        timestamp=row["timestamp"],
        emotion=row["emotion"],
        importance=row["importance"],
        category=row["category"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"] or "",
        episode_id=episode_id,
        sensory_data=_parse_sensory_data(row["sensory_data"] or ""),
        camera_position=_parse_camera_position(row["camera_position"]),
        tags=_parse_tags(row["tags"] or ""),
        novelty_score=float(row["novelty_score"] or 0.0),
        prediction_error=float(row["prediction_error"] or 0.0),
        activation_count=int(row["activation_count"] or 0),
        last_activated=row["last_activated"] or "",
        freshness=float(row["freshness"]) if row["freshness"] is not None else 1.0,
        level=int(row["level"]) if row["level"] is not None else 0,
    )


def _row_to_episode(row: sqlite3.Row) -> Episode:
    """Convert a SQLite Row from the episodes table to an Episode object."""
    memory_ids_raw = row["memory_ids"] or ""
    participants_raw = row["participants"] or ""
    return Episode(
        id=row["id"],
        title=row["title"],
        start_time=row["start_time"],
        end_time=row["end_time"] or None,
        memory_ids=tuple(memory_ids_raw.split(",") if memory_ids_raw else []),
        participants=tuple(participants_raw.split(",") if participants_raw else []),
        location_context=row["location_context"] or None,
        summary=row["summary"] or "",
        emotion=row["emotion"],
        importance=int(row["importance"]),
    )


# ──────────────────────────────────────────────
# MemoryStore
# ──────────────────────────────────────────────


class MemoryStore:
    """SQLite + numpy memory storage (Phase 11)."""

    def __init__(self, config: MemoryConfig, chive: ChiVeEmbedding | None = None):
        self._config = config
        self._db: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._working_memory = WorkingMemoryBuffer(capacity=20)
        self._consolidation_engine = ConsolidationEngine()
        self._hopfield = ModernHopfieldNetwork(beta=4.0, n_iters=3)
        self._chive = chive
        self._bm25_index = BM25Index()

    # ── Connection ──────────────────────────────

    async def connect(self) -> None:
        """Open SQLite database and create tables."""
        async with self._lock:
            if self._db is None:
                db_path = self._config.db_path

                def _open() -> sqlite3.Connection:
                    conn = sqlite3.connect(db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA foreign_keys = ON")
                    conn.execute("PRAGMA journal_mode = WAL")
                    for stmt in _DDL.strip().split(";"):
                        stmt = stmt.strip()
                        if stmt:
                            conn.execute(stmt)
                    # Migration: add category_id to verb_chains if missing
                    cols = [r[1] for r in conn.execute("PRAGMA table_info(verb_chains)").fetchall()]
                    if "category_id" not in cols:
                        conn.execute("ALTER TABLE verb_chains ADD COLUMN category_id INTEGER REFERENCES categories(id)")
                    # Migration: emotion labels → numbers (1-8)
                    _emotion_map = {
                        "happy": "1", "sad": "2", "surprised": "3", "moved": "4",
                        "excited": "5", "nostalgic": "6", "curious": "7", "neutral": "8",
                    }
                    for old, new in _emotion_map.items():
                        conn.execute("UPDATE memories SET emotion = ? WHERE emotion = ?", (new, old))
                        conn.execute("UPDATE verb_chains SET emotion = ? WHERE emotion = ?", (new, old))
                        conn.execute("UPDATE episodes SET emotion = ? WHERE emotion = ?", (new, old))
                    # Migration: add freshness column
                    mem_cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
                    if "freshness" not in mem_cols:
                        conn.execute("ALTER TABLE memories ADD COLUMN freshness REAL NOT NULL DEFAULT 1.0")
                        # Initialize freshness for existing memories based on age
                        now = datetime.now(timezone.utc)
                        for row in conn.execute("SELECT id, timestamp FROM memories WHERE freshness = 1.0").fetchall():
                            try:
                                ts = datetime.fromisoformat(row[1])
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                age_days = (now - ts).days
                                if age_days > 0:
                                    freshness = max(0.01, 0.85 ** age_days)
                                    conn.execute("UPDATE memories SET freshness = ? WHERE id = ?", (freshness, row[0]))
                            except (ValueError, TypeError):
                                pass
                    # Migration: add level column to memories
                    if "level" not in mem_cols:
                        conn.execute("ALTER TABLE memories ADD COLUMN level INTEGER NOT NULL DEFAULT 0")
                    vc_cols = [r[1] for r in conn.execute("PRAGMA table_info(verb_chains)").fetchall()]
                    if "freshness" not in vc_cols:
                        conn.execute("ALTER TABLE verb_chains ADD COLUMN freshness REAL NOT NULL DEFAULT 1.0")
                        now = datetime.now(timezone.utc)
                        for row in conn.execute("SELECT id, timestamp FROM verb_chains WHERE freshness = 1.0").fetchall():
                            try:
                                ts = datetime.fromisoformat(row[1])
                                if ts.tzinfo is None:
                                    ts = ts.replace(tzinfo=timezone.utc)
                                age_days = (now - ts).days
                                if age_days > 0:
                                    freshness = max(0.01, 0.85 ** age_days)
                                    conn.execute("UPDATE verb_chains SET freshness = ? WHERE id = ?", (freshness, row[0]))
                            except (ValueError, TypeError):
                                pass
                    # Migration: create boundary_layers table if missing
                    existing_tables = {r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()}
                    if "boundary_layers" not in existing_tables:
                        conn.execute("""CREATE TABLE IF NOT EXISTS boundary_layers (
                            composite_id TEXT NOT NULL,
                            member_id TEXT NOT NULL,
                            layer_index INTEGER NOT NULL,
                            is_edge INTEGER NOT NULL DEFAULT 0,
                            PRIMARY KEY (composite_id, member_id, layer_index)
                        )""")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_boundary_layers_composite ON boundary_layers(composite_id)")
                    # Migration: create template_biases table if missing
                    if "template_biases" not in existing_tables:
                        conn.execute("""CREATE TABLE IF NOT EXISTS template_biases (
                            chain_id TEXT NOT NULL,
                            bias_weight REAL NOT NULL DEFAULT 0.0,
                            update_count INTEGER NOT NULL DEFAULT 0,
                            last_updated TEXT NOT NULL,
                            PRIMARY KEY (chain_id)
                        )""")
                    # Migration: create composite_axes table if missing
                    if "composite_axes" not in existing_tables:
                        conn.execute("""CREATE TABLE IF NOT EXISTS composite_axes (
                            composite_id TEXT PRIMARY KEY,
                            axis_vector BLOB NOT NULL,
                            explained_variance_ratio REAL NOT NULL DEFAULT 0.0
                        )""")
                    # Migration: create composite_intersections table if missing
                    if "composite_intersections" not in existing_tables:
                        conn.execute("""CREATE TABLE IF NOT EXISTS composite_intersections (
                            composite_a TEXT NOT NULL,
                            composite_b TEXT NOT NULL,
                            intersection_type TEXT NOT NULL CHECK(intersection_type IN ('parallel', 'transversal')),
                            axis_cosine REAL NOT NULL DEFAULT 0.0,
                            shared_member_ids TEXT NOT NULL DEFAULT '',
                            PRIMARY KEY (composite_a, composite_b)
                        )""")
                        conn.execute("CREATE INDEX IF NOT EXISTS idx_intersections_type ON composite_intersections(intersection_type)")
                    # Migration: add flow_vector column to verb_chain_embeddings
                    vce_cols = [r[1] for r in conn.execute("PRAGMA table_info(verb_chain_embeddings)").fetchall()]
                    if "flow_vector" not in vce_cols:
                        conn.execute("ALTER TABLE verb_chain_embeddings ADD COLUMN flow_vector BLOB")
                    # Migration: add delta_vector column to verb_chain_embeddings
                    if "delta_vector" not in vce_cols:
                        conn.execute("ALTER TABLE verb_chain_embeddings ADD COLUMN delta_vector BLOB")
                    # Migration: add flow_vector/delta_vector columns to embeddings
                    emb_cols = [r[1] for r in conn.execute("PRAGMA table_info(embeddings)").fetchall()]
                    if "flow_vector" not in emb_cols:
                        conn.execute("ALTER TABLE embeddings ADD COLUMN flow_vector BLOB")
                    if "delta_vector" not in emb_cols:
                        conn.execute("ALTER TABLE embeddings ADD COLUMN delta_vector BLOB")
                    # Migration: add flow_sim/delta_sim columns to recall_index
                    ri_cols = [r[1] for r in conn.execute("PRAGMA table_info(recall_index)").fetchall()]
                    if ri_cols and "flow_sim" not in ri_cols:
                        conn.execute("ALTER TABLE recall_index ADD COLUMN flow_sim REAL")
                    if ri_cols and "delta_sim" not in ri_cols:
                        conn.execute("ALTER TABLE recall_index ADD COLUMN delta_sim REAL")
                    conn.commit()
                    return conn

                self._db = await asyncio.to_thread(_open)

    async def disconnect(self) -> None:
        """Close the SQLite connection."""
        async with self._lock:
            if self._db is not None:
                await asyncio.to_thread(self._db.close)
                self._db = None

    def _ensure_connected(self) -> sqlite3.Connection:
        if self._db is None:
            raise RuntimeError("MemoryStore not connected. Call connect() first.")
        return self._db

    @property
    def db(self) -> sqlite3.Connection:
        """Expose the raw DB connection (for VerbChainStore sharing)."""
        return self._ensure_connected()

    @property
    def chive(self) -> ChiVeEmbedding:
        """Expose the chiVe embedding (for VerbChainStore sharing)."""
        assert self._chive is not None, "ChiVeEmbedding not initialized"
        return self._chive

    # ── Embedding helpers ───────────────────────

    def _encode_text_sync(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text into (flow_vector, delta_vector) using chiVe. Sync version."""
        assert self._chive is not None
        return self._chive.encode_text(text)

    async def _encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Encode text into (flow_vector, delta_vector) using chiVe."""
        return await asyncio.to_thread(self._encode_text_sync, text)

    # ── Fetch helpers ───────────────────────────

    def _fetch_memory_by_id(self, db: sqlite3.Connection, memory_id: str) -> Memory | None:
        row = db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            return None
        return _row_to_memory(row)

    def _fetch_memories_by_ids_sync(self, db: sqlite3.Connection, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = db.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids
        ).fetchall()
        return [_row_to_memory(row) for row in rows]

    # ── Save ────────────────────────────────────

    async def save(
        self,
        content: str,
        emotion: str = "8",
        importance: int = 3,
        category: str = "daily",
        episode_id: str | None = None,
        sensory_data: tuple[SensoryData, ...] = (),
        camera_position: CameraPosition | None = None,
        tags: tuple[str, ...] = (),
    ) -> Memory:
        """Save a new memory."""
        db = self._ensure_connected()
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        importance = max(1, min(5, importance))

        memory = Memory(
            id=memory_id,
            content=content,
            timestamp=timestamp,
            emotion=emotion,
            importance=importance,
            category=category,
            episode_id=episode_id,
            sensory_data=sensory_data,
            camera_position=camera_position,
            tags=tags,
        )

        normalized_content = normalize_japanese(content)
        reading = get_reading(content)

        flow_vec, delta_vec = await self._encode_text(normalized_content)
        # Legacy 'vector' column: concat of flow + delta for backward compat
        concat_vec = np.concatenate([flow_vec, delta_vec])
        vector_blob = encode_vector(concat_vec)
        flow_blob = encode_vector(flow_vec)
        delta_blob = encode_vector(delta_vec)

        def _insert() -> None:
            meta = memory.to_metadata()
            # Throttled age-proportional freshness decay
            last_decay = db.execute(
                "SELECT COALESCE((SELECT value FROM meta WHERE key = 'last_freshness_decay_at'), '')"
            ).fetchone()[0]
            should_decay = True
            if last_decay:
                try:
                    last_dt = datetime.fromisoformat(last_decay)
                    if last_dt.tzinfo is None:
                        last_dt = last_dt.replace(tzinfo=timezone.utc)
                    elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds()
                    should_decay = elapsed >= 3600  # 1 hour throttle
                except ValueError:
                    pass
            if should_decay:
                # Age-proportional: newer memories decay slower, older decay faster
                # rate = min(0.01, 0.0003 * age_days) per tick
                cutoff = db.execute(
                    "SELECT COALESCE((SELECT value FROM meta WHERE key = 'last_consolidated_at'), '')"
                ).fetchone()[0]
                age_decay_sql = (
                    "SET freshness = MAX(0.01, freshness * (1.0 - MIN(0.01, "
                    "0.0003 * MAX(0, julianday('now') - julianday(timestamp)))))"
                )
                if cutoff:
                    db.execute(f"UPDATE memories {age_decay_sql} WHERE timestamp > ?", (cutoff,))
                    db.execute(f"UPDATE verb_chains {age_decay_sql} WHERE timestamp > ?", (cutoff,))
                else:
                    db.execute(f"UPDATE memories {age_decay_sql}")
                    db.execute(f"UPDATE verb_chains {age_decay_sql}")
                db.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_freshness_decay_at', ?)",
                    (datetime.now(timezone.utc).isoformat(),),
                )
            db.execute(
                """INSERT INTO memories (
                    id, content, normalized_content, timestamp,
                    emotion, importance, category, access_count, last_accessed,
                    linked_ids, episode_id, sensory_data, camera_position,
                    tags, links, novelty_score, prediction_error,
                    activation_count, last_activated, reading, freshness, level
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory_id, content, normalized_content, timestamp,
                    emotion, importance, category,
                    meta.get("access_count", 0), meta.get("last_accessed", ""),
                    "", episode_id or None,
                    meta.get("sensory_data", ""),
                    meta.get("camera_position") or None,
                    meta.get("tags", ""), "",
                    0.0, 0.0, 0, "", reading, 1.0, 0,
                ),
            )
            db.execute(
                "INSERT INTO embeddings (memory_id, vector, flow_vector, delta_vector) VALUES (?,?,?,?)",
                (memory_id, vector_blob, flow_blob, delta_blob),
            )
            db.commit()

        await asyncio.to_thread(_insert)
        self._bm25_index.mark_dirty()
        await self._working_memory.add(memory)
        return memory

    # ── Vector search helpers ───────────────────

    async def _vector_search(
        self,
        query: str,
        n_results: int,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        flow_weight: float = 0.6,
    ) -> list[tuple[Memory, float]]:
        """Return (memory, cosine_distance) pairs using 2-vector similarity.

        Similarity = flow_weight * cos(flow) + (1-flow_weight) * cos(delta)
        Distance = 1 - similarity
        """
        db = self._ensure_connected()
        normalized_query = normalize_japanese(query)
        q_flow, q_delta = await self._encode_text(normalized_query)

        # Build WHERE clause for filters
        conditions: list[str] = []
        params: list[Any] = []
        if emotion_filter:
            conditions.append("m.emotion = ?")
            params.append(emotion_filter)
        if category_filter:
            conditions.append("m.category = ?")
            params.append(category_filter)
        if date_from:
            conditions.append("m.timestamp >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("m.timestamp <= ?")
            params.append(date_to)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = (
            f"SELECT m.*, e.vector, e.flow_vector, e.delta_vector "
            f"FROM memories m JOIN embeddings e ON m.id = e.memory_id {where_clause}"
        )

        def _query() -> list[tuple[sqlite3.Row, bytes | None, bytes | None]]:
            rows = db.execute(sql, params).fetchall()
            return [
                (
                    row,
                    bytes(row["flow_vector"]) if row["flow_vector"] else None,
                    bytes(row["delta_vector"]) if row["delta_vector"] else None,
                )
                for row in rows
            ]

        rows_with_vecs = await asyncio.to_thread(_query)
        if not rows_with_vecs:
            return []

        # Compute 2-axis similarity for each memory
        scored: list[tuple[int, float]] = []
        for i, (row, flow_blob, delta_blob) in enumerate(rows_with_vecs):
            if flow_blob and delta_blob:
                m_flow = decode_vector(flow_blob)
                m_delta = decode_vector(delta_blob)
                flow_sim = float(cosine_similarity(q_flow, m_flow.reshape(1, -1))[0])
                delta_sim = float(cosine_similarity(q_delta, m_delta.reshape(1, -1))[0])
                sim = flow_weight * flow_sim + (1.0 - flow_weight) * delta_sim
            else:
                # Legacy: fallback to concat vector (skip if dimension mismatch)
                legacy_vec = decode_vector(bytes(row["vector"]))
                q_concat = np.concatenate([q_flow, q_delta])
                if legacy_vec.shape[0] != q_concat.shape[0]:
                    continue
                sim = float(cosine_similarity(q_concat, legacy_vec.reshape(1, -1))[0])
            scored.append((i, sim))

        scored.sort(key=lambda t: t[1], reverse=True)
        scored = scored[:n_results]

        results: list[tuple[Memory, float]] = []
        for idx, sim in scored:
            row = rows_with_vecs[idx][0]
            memory = _row_to_memory(row)
            distance = float(1.0 - sim)
            results.append((memory, distance))

        return results

    # ── search ──────────────────────────────────

    async def search(
        self,
        query: str,
        n_results: int = 5,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        flow_weight: float = 0.6,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity."""
        pairs = await self._vector_search(
            query=query,
            n_results=n_results,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
            flow_weight=flow_weight,
        )
        return [MemorySearchResult(memory=m, distance=d) for m, d in pairs]

    # ── search_with_scoring ─────────────────────

    async def search_with_scoring(
        self,
        query: str,
        n_results: int = 5,
        use_time_decay: bool = True,
        use_emotion_boost: bool = True,
        decay_half_life_days: float = 30.0,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        flow_weight: float = 0.6,
    ) -> list[ScoredMemory]:
        """Search with time decay + emotion boost scoring."""
        fetch_count = min(n_results * 3, 50)
        pairs = await self._vector_search(
            query=query,
            n_results=fetch_count,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
            flow_weight=flow_weight,
        )

        scored_results: list[ScoredMemory] = []
        now = datetime.now(timezone.utc)

        for memory, semantic_distance in pairs:
            time_decay = (
                calculate_time_decay(memory.timestamp, now, decay_half_life_days)
                if use_time_decay
                else 1.0
            )
            emotion_boost = calculate_emotion_boost(memory.emotion) if use_emotion_boost else 0.0
            importance_boost = calculate_importance_boost(memory.importance)
            final_score = calculate_final_score(
                semantic_distance=semantic_distance,
                time_decay=time_decay,
                emotion_boost=emotion_boost,
                importance_boost=importance_boost,
            )
            scored_results.append(
                ScoredMemory(
                    memory=memory,
                    semantic_distance=semantic_distance,
                    time_decay_factor=time_decay,
                    emotion_boost=emotion_boost,
                    importance_boost=importance_boost,
                    final_score=final_score,
                )
            )

        # Phase 9: BM25 hybrid re-ranking
        if self._config.enable_bm25 and scored_results:
            if self._bm25_index.is_dirty:
                all_memories = await self.get_all()
                await asyncio.to_thread(
                    self._bm25_index.build,
                    [(m.id, m.content) for m in all_memories],
                )
            result_ids = [sr.memory.id for sr in scored_results]
            bm25_scores = self._bm25_index.scores(query, result_ids)
            query_reading = get_reading(query)
            bm25_weight = 0.2
            reading_weight = 0.15
            reranked: list[ScoredMemory] = []
            for sr in scored_results:
                boost = bm25_scores.get(sr.memory.id, 0.0) * bm25_weight
                if query_reading:
                    doc_reading = get_reading(sr.memory.content) or ""
                    if doc_reading and query_reading == doc_reading:
                        boost += reading_weight
                reranked.append(
                    ScoredMemory(
                        memory=sr.memory,
                        semantic_distance=sr.semantic_distance,
                        time_decay_factor=sr.time_decay_factor,
                        emotion_boost=sr.emotion_boost,
                        importance_boost=sr.importance_boost,
                        final_score=sr.final_score - boost,
                    )
                )
            scored_results = reranked

        scored_results.sort(key=lambda x: x.final_score)
        return scored_results[:n_results]

    # ── recall ──────────────────────────────────

    async def recall(
        self,
        context: str,
        n_results: int = 3,
        flow_weight: float = 0.6,
        emotion_filter: str | None = None,
        category_filter: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[MemorySearchResult]:
        """Recall using hybrid semantic + Hopfield scoring."""
        pool_size = min(n_results * 3, 20)
        scored_results = await self.search_with_scoring(
            query=context, n_results=pool_size, use_time_decay=True, use_emotion_boost=True,
            flow_weight=flow_weight,
            emotion_filter=emotion_filter,
            category_filter=category_filter,
            date_from=date_from,
            date_to=date_to,
        )
        if not scored_results:
            return []

        try:
            hopfield_results = await self.hopfield_recall(query=context, n_results=pool_size, auto_load=True)
            hopfield_scores: dict[str, float] = {r.memory_id: r.hopfield_score for r in hopfield_results}
        except Exception:
            hopfield_scores = {}

        hopfield_weight = 0.15
        blended: list[tuple[ScoredMemory, float]] = []
        for sr in scored_results:
            h_score = hopfield_scores.get(sr.memory.id, 0.0)
            h_boost = max(0.0, h_score) * hopfield_weight
            blended.append((sr, sr.final_score - h_boost))

        blended.sort(key=lambda x: x[1])
        return [
            MemorySearchResult(memory=sr.memory, distance=blended_score)
            for sr, blended_score in blended[:n_results]
        ]

    # ── list_recent ─────────────────────────────

    async def list_recent(self, limit: int = 10, category_filter: str | None = None) -> list[Memory]:
        """List recent memories sorted by timestamp descending."""
        db = self._ensure_connected()

        def _fetch() -> list[sqlite3.Row]:
            if category_filter:
                return db.execute(
                    "SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                    (category_filter, limit),
                ).fetchall()
            return db.execute(
                "SELECT * FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()

        rows = await asyncio.to_thread(_fetch)
        return [_row_to_memory(row) for row in rows]

    # ── get_stats ───────────────────────────────

    async def get_stats(self) -> MemoryStats:
        """Get statistics about stored memories."""
        db = self._ensure_connected()

        def _fetch() -> tuple[list[sqlite3.Row], str | None, str | None]:
            rows = db.execute("SELECT emotion, category, timestamp FROM memories").fetchall()
            oldest = db.execute("SELECT MIN(timestamp) FROM memories").fetchone()[0]
            newest = db.execute("SELECT MAX(timestamp) FROM memories").fetchone()[0]
            return rows, oldest, newest

        rows, oldest, newest = await asyncio.to_thread(_fetch)

        by_category: dict[str, int] = {}
        by_emotion: dict[str, int] = {}
        for row in rows:
            cat = row["category"] or "daily"
            emo = row["emotion"] or "8"
            by_category[cat] = by_category.get(cat, 0) + 1
            by_emotion[emo] = by_emotion.get(emo, 0) + 1

        return MemoryStats(
            total_count=len(rows),
            by_category=by_category,
            by_emotion=by_emotion,
            oldest_timestamp=oldest,
            newest_timestamp=newest,
        )

    # ── get_by_id / get_by_ids ──────────────────

    async def get_by_id(self, memory_id: str) -> Memory | None:
        db = self._ensure_connected()

        def _fetch() -> Memory | None:
            return self._fetch_memory_by_id(db, memory_id)

        return await asyncio.to_thread(_fetch)

    async def get_by_ids(self, memory_ids: list[str]) -> list[Memory]:
        if not memory_ids:
            return []
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            return self._fetch_memories_by_ids_sync(db, memory_ids)

        return await asyncio.to_thread(_fetch)

    # ── get_all ─────────────────────────────────

    async def get_all(self) -> list[Memory]:
        """Return all memories."""
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            rows = db.execute("SELECT * FROM memories").fetchall()
            return [_row_to_memory(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    # ── update_access ───────────────────────────

    # ── freshness ───────────────────────────────

    async def decay_all_freshness(self, amount: float = 0.003) -> None:
        """全記憶・チェーンの freshness を age 比例で減少させる。

        amount は後方互換のために残すが、実際は age 比例の乗算decay を使う。
        rate = min(0.01, 0.0003 * age_days) per tick。
        """
        db = self._ensure_connected()

        def _decay() -> None:
            age_decay_sql = (
                "SET freshness = MAX(0.01, freshness * (1.0 - MIN(0.01, "
                "0.0003 * MAX(0, julianday('now') - julianday(timestamp)))))"
            )
            db.execute(f"UPDATE memories {age_decay_sql}")
            db.execute(f"UPDATE verb_chains {age_decay_sql}")
            db.commit()

        await asyncio.to_thread(_decay)

    async def consolidate_freshness(self, factor: float = 0.92) -> None:
        """コンソリデート時に全 freshness を割合減衰させ、タイムスタンプを記録。"""
        db = self._ensure_connected()

        def _consolidate() -> None:
            db.execute("UPDATE memories SET freshness = MAX(0.01, freshness * ?)", (factor,))
            db.execute("UPDATE verb_chains SET freshness = MAX(0.01, freshness * ?)", (factor,))
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('last_consolidated_at', ?)",
                (now,),
            )
            db.commit()

        await asyncio.to_thread(_consolidate)

    # ── update_diary_content ───────────────────

    async def update_diary_content(
        self,
        memory_id: str,
        amendment: str,
        emotion: str | None = None,
        importance: int | None = None,
    ) -> Memory | None:
        """Amend a diary entry with strikethrough + appended note.

        First amendment: ~~original~~ + [追記 timestamp] amendment
        Subsequent amendments: append [追記 timestamp] amendment to the end.
        Embedding is recomputed for the full updated text.
        """
        db = self._ensure_connected()

        existing = await self.get_by_id(memory_id)
        if existing is None:
            return None

        # Build updated content
        old_content = existing.content
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

        if old_content.startswith("~~"):
            # Already amended — just append a new amendment line
            new_content = f"{old_content}\n\n[追記 {now_str}] {amendment}"
        else:
            # First amendment — wrap original in strikethrough
            new_content = f"~~{old_content}~~\n\n[追記 {now_str}] {amendment}"

        # Normalize and embed the full new content
        normalized = normalize_japanese(new_content)
        reading = get_reading(new_content)
        flow_vec, delta_vec = await self._encode_text(normalized)
        concat_vec = np.concatenate([flow_vec, delta_vec])
        vector_blob = encode_vector(concat_vec)
        flow_blob = encode_vector(flow_vec)
        delta_blob = encode_vector(delta_vec)

        # Determine updated emotion/importance
        new_emotion = emotion if emotion is not None else existing.emotion
        new_importance = max(1, min(5, importance)) if importance is not None else existing.importance

        def _update() -> None:
            db.execute(
                """UPDATE memories
                   SET content = ?,
                       normalized_content = ?,
                       reading = ?,
                       emotion = ?,
                       importance = ?
                   WHERE id = ?""",
                (new_content, normalized, reading, new_emotion, new_importance, memory_id),
            )
            db.execute(
                "UPDATE embeddings SET vector = ?, flow_vector = ?, delta_vector = ? WHERE memory_id = ?",
                (vector_blob, flow_blob, delta_blob, memory_id),
            )
            db.commit()

        await asyncio.to_thread(_update)
        self._bm25_index.mark_dirty()

        # Return updated memory
        return await self.get_by_id(memory_id)

    # ── update_access ──────────────────────────

    async def update_access(self, memory_id: str) -> None:
        db = self._ensure_connected()

        def _update() -> None:
            db.execute(
                """UPDATE memories
                   SET access_count = access_count + 1,
                       last_accessed = ?
                   WHERE id = ?""",
                (datetime.now(timezone.utc).isoformat(), memory_id),
            )
            db.commit()

        await asyncio.to_thread(_update)

    # ── update_episode_id ───────────────────────

    async def update_episode_id(self, memory_id: str, episode_id: str) -> None:
        db = self._ensure_connected()
        ep_val = episode_id if episode_id else None

        def _update() -> None:
            result = db.execute(
                "UPDATE memories SET episode_id = ? WHERE id = ?",
                (ep_val, memory_id),
            )
            if result.rowcount == 0:
                raise ValueError(f"Memory not found: {memory_id}")
            db.commit()

        await asyncio.to_thread(_update)

    # ── update_memory_fields ────────────────────

    async def update_memory_fields(self, memory_id: str, **fields: Any) -> bool:
        """Update arbitrary fields on a memory row."""
        if not fields:
            return True
        db = self._ensure_connected()

        valid_cols = {
            "access_count", "last_accessed", "episode_id",
            "sensory_data", "camera_position", "tags",
            "novelty_score", "prediction_error", "activation_count",
            "last_activated", "reading",
        }
        valid = {k: v for k, v in fields.items() if k in valid_cols}
        if not valid:
            return True

        set_clause = ", ".join(f"{k} = ?" for k in valid)
        values = list(valid.values()) + [memory_id]

        def _update() -> bool:
            result = db.execute(f"UPDATE memories SET {set_clause} WHERE id = ?", values)
            db.commit()
            return result.rowcount > 0

        return await asyncio.to_thread(_update)

    # ── record_activation ───────────────────────

    async def record_activation(
        self,
        memory_id: str,
        prediction_error: float | None = None,
    ) -> bool:
        memory = await self.get_by_id(memory_id)
        if memory is None:
            return False
        payload: dict[str, Any] = {
            "activation_count": memory.activation_count + 1,
            "last_activated": datetime.now(timezone.utc).isoformat(),
        }
        if prediction_error is not None:
            payload["prediction_error"] = max(0.0, min(1.0, prediction_error))
        return await self.update_memory_fields(memory_id, **payload)

    # ── Recall Index (pre-computed word→memory similarity) ──

    async def build_recall_index(self) -> int:
        """Build recall_index if needed (startup-optimized).

        Checks meta table for 'recall_index_built_at' timestamp and compares
        it against the latest memory/chain timestamp. If the index was built
        after the most recent data change, we skip the rebuild.

        This means:
        - Normal startup (no new data since last build): instant (~20ms)
        - First startup or after adding data: full rebuild (~10-20s)
        - Between startups, update_recall_index() keeps the index current

        Returns the number of entries written (0 if skipped).
        """
        import logging

        logger = logging.getLogger(__name__)
        db = self._ensure_connected()

        def _check() -> tuple[str | None, str | None, int]:
            # When was the index last built?
            row = db.execute(
                "SELECT value FROM meta WHERE key = 'recall_index_built_at'"
            ).fetchone()
            built_at = row[0] if row else None

            # What's the latest data timestamp?
            row = db.execute(
                "SELECT MAX(ts) FROM ("
                "  SELECT MAX(timestamp) as ts FROM memories"
                "  UNION ALL"
                "  SELECT MAX(timestamp) as ts FROM verb_chains"
                ")"
            ).fetchone()
            latest_data = row[0] if row and row[0] else None

            # How many entries exist?
            entry_count = db.execute(
                "SELECT COUNT(*) FROM recall_index"
            ).fetchone()[0]

            return built_at, latest_data, entry_count

        built_at, latest_data, entry_count = await asyncio.to_thread(_check)

        # Skip if index exists and was built after the latest data
        if built_at and latest_data and built_at >= latest_data and entry_count > 0:
            logger.info(
                "build_recall_index: up to date (built_at=%s, latest_data=%s, %d entries), skipping",
                built_at[:19], latest_data[:19], entry_count,
            )
            return 0

        if not latest_data:
            logger.info("build_recall_index: no data yet, skipping")
            return 0

        logger.info(
            "build_recall_index: stale (built_at=%s, latest_data=%s), full rebuild",
            (built_at or "never")[:19], latest_data[:19],
        )
        return await self.rebuild_recall_index_full()

    async def rebuild_recall_index_full(self) -> int:
        """Full rebuild of the recall_index table (DELETE + re-create).

        Used by the rebuild_recall_index tool. Collects all vocabulary,
        encodes each word, computes cosine similarity against all embeddings,
        and stores top-20 per word.

        Returns the number of word→target entries inserted.
        """
        import logging

        logger = logging.getLogger(__name__)
        db = self._ensure_connected()

        # 1. Collect vocabulary from graph_nodes and verb_chains
        def _collect_vocab() -> set[str]:
            words: set[str] = set()
            rows = db.execute(
                "SELECT surface_form FROM graph_nodes WHERE type IN ('verb', 'noun')"
            ).fetchall()
            for row in rows:
                w = row[0].strip()
                if w:
                    words.add(w)
            rows = db.execute("SELECT all_nouns, all_verbs FROM verb_chains").fetchall()
            for row in rows:
                for field in (row[0], row[1]):
                    if field:
                        for w in field.split(","):
                            w = w.strip()
                            if w:
                                words.add(w)
            return words

        vocab = await asyncio.to_thread(_collect_vocab)

        if not vocab:
            logger.warning("rebuild_recall_index_full: no vocabulary found, skipping")
            return 0

        vocab_list = sorted(vocab)
        logger.info("rebuild_recall_index_full: %d vocabulary words collected", len(vocab_list))

        # 2. Encode all vocabulary words via chiVe
        assert self._chive is not None
        word_vecs_map = await asyncio.to_thread(self._chive.batch_get, vocab_list)
        # Filter out OOV words
        valid_words = [w for w in vocab_list if w in word_vecs_map]
        if not valid_words:
            logger.warning("rebuild_recall_index_full: all vocabulary OOV in chiVe, skipping")
            return 0
        word_vecs_np = np.array([word_vecs_map[w] for w in valid_words], dtype=np.float32)

        # 3. Load all memory flow_vectors and delta_vectors
        def _load_memory_vecs() -> tuple[
            list[str], list[str], np.ndarray | None, np.ndarray | None
        ]:
            rows = db.execute(
                "SELECT e.memory_id, m.content, e.flow_vector, e.delta_vector "
                "FROM embeddings e JOIN memories m ON e.memory_id = m.id "
                "WHERE e.flow_vector IS NOT NULL"
            ).fetchall()
            if not rows:
                return [], [], None, None
            ids = [r[0] for r in rows]
            previews = [r[1][:20] if r[1] else "" for r in rows]
            flow = np.array(
                [decode_vector(bytes(r[2])) for r in rows], dtype=np.float32
            )
            delta = None
            delta_rows = [r for r in rows if r[3] is not None]
            if delta_rows:
                # Build delta array aligned with ids (0 for missing)
                delta_map = {r[0]: decode_vector(bytes(r[3])) for r in delta_rows}
                dim = len(next(iter(delta_map.values())))
                delta_list = [delta_map.get(id_, np.zeros(dim, dtype=np.float32)) for id_ in ids]
                delta = np.array(delta_list, dtype=np.float32)
            return ids, previews, flow, delta

        mem_ids, mem_previews, mem_flow_vecs, mem_delta_vecs = await asyncio.to_thread(
            _load_memory_vecs
        )

        # 4. Load all verb_chain flow_vectors and delta_vectors
        def _load_chain_vecs() -> tuple[
            list[str], list[str], np.ndarray | None, np.ndarray | None
        ]:
            rows = db.execute(
                "SELECT ce.chain_id, vc.context, ce.flow_vector, ce.delta_vector "
                "FROM verb_chain_embeddings ce "
                "JOIN verb_chains vc ON ce.chain_id = vc.id "
                "WHERE ce.flow_vector IS NOT NULL"
            ).fetchall()
            if not rows:
                return [], [], None, None
            ids = [r[0] for r in rows]
            previews = [r[1][:20] if r[1] else "" for r in rows]
            flow = np.array(
                [decode_vector(bytes(r[2])) for r in rows], dtype=np.float32
            )
            delta = None
            delta_rows = [r for r in rows if r[3] is not None]
            if delta_rows:
                delta_map = {r[0]: decode_vector(bytes(r[3])) for r in delta_rows}
                dim = len(next(iter(delta_map.values())))
                delta_list = [delta_map.get(id_, np.zeros(dim, dtype=np.float32)) for id_ in ids]
                delta = np.array(delta_list, dtype=np.float32)
            return ids, previews, flow, delta

        chain_ids, chain_previews, chain_flow_vecs, chain_delta_vecs = (
            await asyncio.to_thread(_load_chain_vecs)
        )

        if mem_flow_vecs is None and chain_flow_vecs is None:
            logger.warning("rebuild_recall_index_full: no flow vectors found, skipping")
            return 0

        # 5. Compute similarities (batch matrix ops via numpy)
        TOP_K = 20

        # Merge all targets into single arrays for batch computation
        all_ids: list[str] = []
        all_types: list[str] = []
        all_previews: list[str] = []
        flow_arrays: list[np.ndarray] = []
        delta_arrays: list[np.ndarray] = []
        delta_mask_parts: list[np.ndarray] = []  # per-target: True if real delta exists

        if mem_flow_vecs is not None:
            n_mem = len(mem_ids)
            all_ids.extend(mem_ids)
            all_types.extend(["memory"] * n_mem)
            all_previews.extend(mem_previews)
            flow_arrays.append(mem_flow_vecs)
            if mem_delta_vecs is not None:
                delta_arrays.append(mem_delta_vecs)
                delta_mask_parts.append(np.ones(n_mem, dtype=bool))
            else:
                delta_arrays.append(np.zeros_like(mem_flow_vecs))
                delta_mask_parts.append(np.zeros(n_mem, dtype=bool))

        if chain_flow_vecs is not None:
            n_chain = len(chain_ids)
            all_ids.extend(chain_ids)
            all_types.extend(["chain"] * n_chain)
            all_previews.extend(chain_previews)
            flow_arrays.append(chain_flow_vecs)
            if chain_delta_vecs is not None:
                delta_arrays.append(chain_delta_vecs)
                delta_mask_parts.append(np.ones(n_chain, dtype=bool))
            else:
                delta_arrays.append(np.zeros_like(chain_flow_vecs))
                delta_mask_parts.append(np.zeros(n_chain, dtype=bool))

        if not flow_arrays:
            return 0

        all_flow = np.concatenate(flow_arrays, axis=0)  # (N_targets, dim)
        all_delta = np.concatenate(delta_arrays, axis=0)
        delta_mask = np.concatenate(delta_mask_parts)  # (N_targets,)
        has_any_delta = delta_mask.any()

        # Normalize for cosine similarity via matrix multiply
        # word_vecs_np: (N_words, dim), all_flow: (N_targets, dim)
        def _normalize_rows(m: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            return m / norms

        word_normed = _normalize_rows(word_vecs_np)
        flow_normed = _normalize_rows(all_flow)

        # (N_words, N_targets) similarity matrix
        flow_sim_matrix = word_normed @ flow_normed.T

        if has_any_delta:
            delta_normed = _normalize_rows(all_delta)
            delta_sim_matrix = word_normed @ delta_normed.T
            # Targets with delta: 0.6*flow + 0.4*delta; without: flow only (preserves old behavior)
            combined_matrix = np.where(
                delta_mask[np.newaxis, :],
                0.6 * flow_sim_matrix + 0.4 * delta_sim_matrix,
                flow_sim_matrix,
            )
        else:
            delta_sim_matrix = None
            combined_matrix = flow_sim_matrix

        # Extract top-K per word using argpartition (faster than full sort)
        n_targets = combined_matrix.shape[1]
        k = min(TOP_K, n_targets)

        entries: list[tuple[str, str, str, float, str, float | None, float | None]] = []

        for i, word in enumerate(valid_words):
            row = combined_matrix[i]
            if k < n_targets:
                top_indices = np.argpartition(row, -k)[-k:]
                top_indices = top_indices[np.argsort(row[top_indices])[::-1]]
            else:
                top_indices = np.argsort(row)[::-1][:k]

            for j in top_indices:
                j_int = int(j)
                f_sim = float(flow_sim_matrix[i, j_int])
                d_sim = float(delta_sim_matrix[i, j_int]) if delta_sim_matrix is not None else None
                entries.append((
                    word, all_ids[j_int], all_types[j_int],
                    float(row[j_int]), all_previews[j_int],
                    f_sim, d_sim,
                ))

        # 6. Write to database (clear and rebuild)
        def _write_index() -> int:
            db.execute("DELETE FROM recall_index")
            db.executemany(
                "INSERT OR REPLACE INTO recall_index "
                "(word, target_id, target_type, similarity, content_preview, "
                "flow_sim, delta_sim) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                entries,
            )
            now = datetime.now(timezone.utc).isoformat()
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('recall_index_built_at', ?)",
                (now,),
            )
            db.commit()
            return len(entries)

        count = await asyncio.to_thread(_write_index)
        logger.info("rebuild_recall_index_full: %d entries written", count)
        return count

    async def update_recall_index(self, memory_id: str, target_type: str = "memory") -> int:
        """Incrementally update recall_index for a single new memory/chain.

        Uses chiVe flow_vector and delta_vector for similarity against vocabulary words.
        Returns the number of entries added/updated.
        """
        db = self._ensure_connected()

        # Load the new flow_vector and delta_vector
        def _load_vectors() -> tuple[bytes | None, bytes | None]:
            table = "embeddings" if target_type == "memory" else "verb_chain_embeddings"
            id_col = "memory_id" if target_type == "memory" else "chain_id"
            row = db.execute(
                f"SELECT flow_vector, delta_vector FROM {table} WHERE {id_col} = ?",
                (memory_id,),
            ).fetchone()
            if not row or not row[0]:
                return None, None
            flow_blob = bytes(row[0])
            delta_blob = bytes(row[1]) if row[1] else None
            return flow_blob, delta_blob

        flow_blob, delta_blob = await asyncio.to_thread(_load_vectors)
        if flow_blob is None:
            return 0

        new_flow_vec = decode_vector(flow_blob)
        new_delta_vec = decode_vector(delta_blob) if delta_blob else None

        # Load content preview
        def _load_preview() -> str:
            if target_type == "memory":
                row = db.execute(
                    "SELECT content FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                return row[0][:20] if row and row[0] else ""
            else:
                row = db.execute(
                    "SELECT context FROM verb_chains WHERE id = ?", (memory_id,)
                ).fetchone()
                return row[0][:20] if row and row[0] else ""

        preview = await asyncio.to_thread(_load_preview)

        # Get all vocabulary words from recall_index (distinct words)
        def _get_vocab() -> list[str]:
            rows = db.execute("SELECT DISTINCT word FROM recall_index").fetchall()
            return [r[0] for r in rows]

        vocab_list = await asyncio.to_thread(_get_vocab)
        if not vocab_list:
            return 0

        # Encode vocabulary words via chiVe
        assert self._chive is not None
        word_vecs_map = await asyncio.to_thread(self._chive.batch_get, vocab_list)
        valid_words = [w for w in vocab_list if w in word_vecs_map]
        if not valid_words:
            return 0
        word_vecs_np = np.array([word_vecs_map[w] for w in valid_words], dtype=np.float32)

        # Compute similarities
        flow_sims = cosine_similarity(new_flow_vec, word_vecs_np)
        delta_sims = (
            cosine_similarity(new_delta_vec, word_vecs_np)
            if new_delta_vec is not None else None
        )

        TOP_K = 20

        def _update_entries() -> int:
            count = 0
            for i, word in enumerate(valid_words):
                f_sim = float(flow_sims[i])
                d_sim = float(delta_sims[i]) if delta_sims is not None else None
                combined = 0.6 * f_sim + 0.4 * d_sim if d_sim is not None else f_sim

                rows = db.execute(
                    "SELECT similarity FROM recall_index "
                    "WHERE word = ? ORDER BY similarity ASC LIMIT 1",
                    (word,),
                ).fetchall()

                existing_count = db.execute(
                    "SELECT COUNT(*) FROM recall_index WHERE word = ?", (word,)
                ).fetchone()[0]

                if existing_count < TOP_K:
                    db.execute(
                        "INSERT OR REPLACE INTO recall_index "
                        "(word, target_id, target_type, similarity, content_preview, "
                        "flow_sim, delta_sim) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (word, memory_id, target_type, combined, preview, f_sim, d_sim),
                    )
                    count += 1
                elif rows and combined > rows[0][0]:
                    db.execute(
                        "DELETE FROM recall_index WHERE word = ? AND similarity = ("
                        "SELECT MIN(similarity) FROM recall_index WHERE word = ?)",
                        (word, word),
                    )
                    db.execute(
                        "INSERT OR REPLACE INTO recall_index "
                        "(word, target_id, target_type, similarity, content_preview, "
                        "flow_sim, delta_sim) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (word, memory_id, target_type, combined, preview, f_sim, d_sim),
                    )
                    count += 1
            db.commit()
            return count

        updated = await asyncio.to_thread(_update_entries)
        return updated

    # ── search_important_memories ───────────────

    async def search_important_memories(
        self,
        min_importance: int = 4,
        min_access_count: int = 5,
        since: str | None = None,
        n_results: int = 10,
    ) -> list[Memory]:
        db = self._ensure_connected()

        def _fetch() -> list[Memory]:
            conditions = [
                "importance >= ?",
                "access_count >= ?",
            ]
            params: list[Any] = [min_importance, min_access_count]
            if since:
                conditions.append("last_accessed >= ?")
                params.append(since)
            where = " AND ".join(conditions)
            rows = db.execute(
                f"SELECT * FROM memories WHERE {where} ORDER BY last_accessed DESC LIMIT ?",
                params + [n_results],
            ).fetchall()
            return [_row_to_memory(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    # ── get_working_memory ──────────────────────

    def get_working_memory(self) -> WorkingMemoryBuffer:
        return self._working_memory

    # ── Episode CRUD ────────────────────────────

    async def save_episode(self, episode: Episode) -> None:
        """Persist an Episode to the episodes table."""
        db = self._ensure_connected()

        def _insert() -> None:
            db.execute(
                """INSERT INTO episodes
                   (id, title, start_time, end_time, memory_ids, participants,
                    location_context, summary, emotion, importance)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    episode.id,
                    episode.title,
                    episode.start_time,
                    episode.end_time or None,
                    ",".join(episode.memory_ids),
                    ",".join(episode.participants),
                    episode.location_context,
                    episode.summary,
                    episode.emotion,
                    episode.importance,
                ),
            )
            db.commit()

        await asyncio.to_thread(_insert)

    async def get_episode_by_id(self, episode_id: str) -> Episode | None:
        db = self._ensure_connected()

        def _fetch() -> Episode | None:
            row = db.execute("SELECT * FROM episodes WHERE id = ?", (episode_id,)).fetchone()
            if row is None:
                return None
            return _row_to_episode(row)

        return await asyncio.to_thread(_fetch)

    async def search_episodes(self, query: str, n_results: int = 5) -> list[Episode]:
        """Search episodes by title/summary (LIKE search)."""
        db = self._ensure_connected()
        pattern = f"%{query}%"

        def _fetch() -> list[Episode]:
            rows = db.execute(
                """SELECT * FROM episodes
                   WHERE title LIKE ? OR summary LIKE ?
                   ORDER BY start_time DESC LIMIT ?""",
                (pattern, pattern, n_results),
            ).fetchall()
            return [_row_to_episode(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    async def list_all_episodes(self) -> list[Episode]:
        db = self._ensure_connected()

        def _fetch() -> list[Episode]:
            rows = db.execute("SELECT * FROM episodes ORDER BY start_time DESC").fetchall()
            return [_row_to_episode(row) for row in rows]

        return await asyncio.to_thread(_fetch)

    async def delete_episode(self, episode_id: str) -> None:
        db = self._ensure_connected()

        def _delete() -> None:
            db.execute("DELETE FROM episodes WHERE id = ?", (episode_id,))
            db.commit()

        await asyncio.to_thread(_delete)

    # ── Divergent recall ─────────────────────────

    async def recall_divergent(
        self,
        context: str,
        n_results: int = 5,
        max_branches: int = 3,
        max_depth: int = 3,
        temperature: float = 0.7,
        include_diagnostics: bool = False,
        record_activation: bool = True,
    ) -> tuple[list[MemorySearchResult], dict[str, Any]]:
        n_results = max(1, min(20, n_results))
        seed_size = max(3, min(25, n_results * 3))
        seeds = await self.search_with_scoring(query=context, n_results=seed_size)
        if not seeds:
            return [], {}

        seed_memories = [item.memory for item in seeds]

        distance_map = {item.memory.id: item.semantic_distance for item in seeds}
        all_candidates: dict[str, Memory] = {}
        for memory in seed_memories:
            all_candidates[memory.id] = memory

        # ── Boundary-aware composite edge expansion ──
        # spread 後に level=1 の composite を抽出して edge メンバー + 隣接 composite を展開
        composite_ids = [
            mid for mid, mem in all_candidates.items() if mem.level >= 1
        ]
        if composite_ids:
            normalized_query = normalize_japanese(context)
            q_flow, q_delta = await self._encode_text(normalized_query)
            query_vec = np.concatenate([q_flow, q_delta])
            edge_memories = await self.expand_composite_edges(composite_ids, query_vec)
            for mem in edge_memories:
                if mem.id not in all_candidates:
                    all_candidates[mem.id] = mem

        # ── Boundary fuzziness scores ──
        all_ids = list(all_candidates.keys())
        boundary_scores = await self.get_member_boundary_scores(all_ids)

        # ── Intersection boost scores ──
        intersection_boosts = await self.get_intersection_nodes(all_ids)

        workspace_candidates: list[WorkspaceCandidate] = []
        prediction_errors: list[float] = []
        novelty_scores: list[float] = []

        for memory in all_candidates.values():
            semantic_distance = distance_map.get(memory.id)
            if semantic_distance is None:
                relevance = calculate_context_relevance(context, memory)
            else:
                relevance = 1.0 / (1.0 + max(0.0, semantic_distance))

            prediction_error = calculate_prediction_error(context, memory)
            novelty = calculate_novelty_score(memory, prediction_error)
            emotion_boost = calculate_emotion_boost(memory.emotion)
            normalized_emotion = max(0.0, min(1.0, emotion_boost / 0.4))

            # boundary_score + intersection_boost
            base_boundary = boundary_scores.get(memory.id, 0.0)
            i_boost = intersection_boosts.get(memory.id, 0.0)
            combined_boundary = base_boundary + i_boost

            prediction_errors.append(prediction_error)
            novelty_scores.append(novelty)
            workspace_candidates.append(
                WorkspaceCandidate(
                    memory=memory,
                    relevance=relevance,
                    novelty=novelty,
                    prediction_error=prediction_error,
                    emotion_boost=normalized_emotion,
                    boundary_score=combined_boundary,
                )
            )

        selected = select_workspace_candidates(
            candidates=workspace_candidates,
            max_results=n_results,
            temperature=temperature,
        )

        results: list[MemorySearchResult] = []
        selected_memories: list[Memory] = []
        for candidate, utility in selected:
            selected_memories.append(candidate.memory)
            if record_activation:
                await self.record_activation(
                    candidate.memory.id,
                    prediction_error=candidate.prediction_error,
                )
                await self.update_memory_fields(
                    candidate.memory.id,
                    novelty_score=candidate.novelty,
                    prediction_error=candidate.prediction_error,
                )
            score_distance = max(0.0, 1.0 - utility)
            results.append(MemorySearchResult(memory=candidate.memory, distance=score_distance))

        if not include_diagnostics:
            return results, {}

        diagnostics = self._build_divergent_diagnostics(
            context=context,
            selected=selected_memories,
            prediction_errors=prediction_errors,
            novelty_scores=novelty_scores,
        )
        return results, diagnostics

    async def get_association_diagnostics(self, context: str, sample_size: int = 20) -> dict[str, Any]:
        n_results = max(3, min(20, sample_size))
        _, diagnostics = await self.recall_divergent(
            context=context,
            n_results=n_results,
            max_branches=4,
            max_depth=3,
            include_diagnostics=True,
            record_activation=False,
        )
        return diagnostics

    async def consolidate_memories(
        self,
        window_hours: int = 24,
        max_replay_events: int = 200,
        link_update_strength: float = 0.2,
        synthesize: bool = True,
        n_layers: int = 3,
        graph: "Any | None" = None,
    ) -> dict[str, int]:
        stats = await self._consolidation_engine.run(
            store=self,
            window_hours=window_hours,
            max_replay_events=max_replay_events,
            link_update_strength=link_update_strength,
        )
        result = stats.to_dict()

        if synthesize:
            # Clean up stale composite_axes from previous model migrations
            if self._chive is not None:
                expected_dim = self._chive.vector_size * 2  # flow + delta
                await self.cleanup_stale_composite_axes(expected_dim)

            # Level 0 → 1
            synth_stats = await self._consolidation_engine.synthesize_composites(
                store=self,
                source_level=0,
                target_level=1,
                similarity_threshold=0.75,
            )
            result.update(synth_stats)

            # 孤立救出 (level=0)
            rescue_stats_0 = await self._consolidation_engine.rescue_orphans(
                store=self, level=0,
            )
            result["orphans_rescued_l0"] = rescue_stats_0["orphans_rescued"]

            # クラスタ重なり検出
            overlap_stats = await self._consolidation_engine.detect_overlap(
                store=self,
            )
            result.update(overlap_stats)

            # Level 1 → 2 (閾値を下げる)
            synth_stats_2 = await self._consolidation_engine.synthesize_composites(
                store=self,
                source_level=1,
                target_level=2,
                similarity_threshold=0.55,
                min_group_size=2,
            )
            result["l2_composites_created"] = synth_stats_2["composites_created"]

            # Boundary layers computation (全 level の composite に対して)
            boundary_stats = await self._consolidation_engine.compute_boundary_layers(
                store=self,
                graph=graph,
                n_layers=n_layers,
            )
            result.update(boundary_stats)

            # Intersection detection (after boundary layers)
            intersection_stats = await self._consolidation_engine.detect_intersections(
                store=self,
            )
            result.update(intersection_stats)

        return result

    async def fetch_memories_with_vectors_by_level(
        self,
        level: int = 0,
        min_freshness: float = 0.1,
    ) -> list[tuple[Memory, np.ndarray]]:
        """指定 level かつ freshness > min_freshness の記憶とベクトル(flow+delta concat)を取得。"""
        db = self._ensure_connected()

        def _fetch() -> list[tuple[sqlite3.Row, bytes | None, bytes | None, bytes]]:
            rows = db.execute(
                """SELECT m.*, e.vector, e.flow_vector, e.delta_vector FROM memories m
                   JOIN embeddings e ON e.memory_id = m.id
                   WHERE m.level = ? AND m.freshness > ?""",
                (level, min_freshness),
            ).fetchall()
            return [
                (row, row["flow_vector"], row["delta_vector"], bytes(row["vector"]))
                for row in rows
            ]

        raw = await asyncio.to_thread(_fetch)
        results: list[tuple[Memory, np.ndarray]] = []
        for row, flow_blob, delta_blob, legacy_blob in raw:
            mem = _row_to_memory(row)
            if flow_blob and delta_blob:
                flow = decode_vector(bytes(flow_blob))
                delta = decode_vector(bytes(delta_blob))
                vec = np.concatenate([flow, delta])
            else:
                vec = decode_vector(legacy_blob)
            results.append((mem, vec))
        return results

    async def fetch_level0_memories_with_vectors(
        self,
        min_freshness: float = 0.1,
    ) -> list[tuple[Memory, np.ndarray]]:
        """level=0 の記憶とベクトルを取得（後方互換ラッパー）。"""
        return await self.fetch_memories_with_vectors_by_level(
            level=0, min_freshness=min_freshness,
        )

    async def get_existing_composite_members(self) -> list[frozenset[str]]:
        """既存の合成記憶のメンバー構成をすべて取得。"""
        db = self._ensure_connected()

        def _fetch() -> list[frozenset[str]]:
            composites: dict[str, set[str]] = {}
            for row in db.execute("SELECT composite_id, member_id FROM composite_members").fetchall():
                cid = row["composite_id"]
                if cid not in composites:
                    composites[cid] = set()
                composites[cid].add(row["member_id"])
            return [frozenset(members) for members in composites.values()]

        return await asyncio.to_thread(_fetch)

    async def fetch_orphan_memories(
        self,
        level: int = 0,
        min_freshness: float = 0.1,
    ) -> list[tuple[Memory, np.ndarray]]:
        """指定 level でどの composite にも属さない孤立記憶を取得。"""
        db = self._ensure_connected()

        def _fetch() -> list[tuple[sqlite3.Row, bytes | None, bytes | None, bytes]]:
            rows = db.execute(
                """SELECT m.*, e.vector, e.flow_vector, e.delta_vector FROM memories m
                   JOIN embeddings e ON e.memory_id = m.id
                   LEFT JOIN composite_members cm ON cm.member_id = m.id
                   WHERE m.level = ? AND m.freshness > ?
                     AND cm.composite_id IS NULL""",
                (level, min_freshness),
            ).fetchall()
            return [
                (row, row["flow_vector"], row["delta_vector"], bytes(row["vector"]))
                for row in rows
            ]

        raw = await asyncio.to_thread(_fetch)
        results: list[tuple[Memory, np.ndarray]] = []
        for row, flow_blob, delta_blob, legacy_blob in raw:
            mem = _row_to_memory(row)
            if flow_blob and delta_blob:
                flow = decode_vector(bytes(flow_blob))
                delta = decode_vector(bytes(delta_blob))
                vec = np.concatenate([flow, delta])
            else:
                legacy_vec = decode_vector(legacy_blob)
                if legacy_vec.shape[0] != self._chive.vector_size * 2:
                    continue  # skip unmigrated 768-dim vectors
                vec = legacy_vec
            results.append((mem, vec))
        return results

    async def save_composite(
        self,
        member_ids: list[str],
        vector: np.ndarray,
        emotion: str,
        importance: int,
        freshness: float,
        category: str,
        axis_vector: np.ndarray | None = None,
        explained_variance_ratio: float | None = None,
        level: int = 1,
    ) -> str:
        """合成記憶を保存し、composite_members に関係を登録。

        vector is the full concat (flow+delta) centroid.
        We split it into flow_vector and delta_vector for the embeddings table.
        """
        db = self._ensure_connected()
        composite_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        vector_blob = encode_vector(vector.tolist())
        axis_blob = encode_vector(axis_vector.tolist()) if axis_vector is not None else None

        # Split the vector into flow and delta halves if it has the right dimension
        chive_dim = self._chive.vector_size if self._chive else 300
        if len(vector) == chive_dim * 2:
            flow_blob = encode_vector(vector[:chive_dim].tolist())
            delta_blob = encode_vector(vector[chive_dim:].tolist())
        else:
            flow_blob = None
            delta_blob = None

        def _insert() -> None:
            db.execute(
                """INSERT INTO memories (
                    id, content, normalized_content, timestamp,
                    emotion, importance, category, access_count, last_accessed,
                    linked_ids, episode_id, sensory_data, camera_position,
                    tags, links, novelty_score, prediction_error,
                    activation_count, last_activated, reading, freshness, level
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    composite_id, "", "", timestamp,
                    emotion, importance, category,
                    0, "",
                    "", None,
                    "", None,
                    "", "",
                    0.0, 0.0, 0, "", None, freshness, level,
                ),
            )
            db.execute(
                "INSERT INTO embeddings (memory_id, vector, flow_vector, delta_vector) VALUES (?,?,?,?)",
                (composite_id, vector_blob, flow_blob, delta_blob),
            )
            for mid in member_ids:
                db.execute(
                    "INSERT INTO composite_members (composite_id, member_id) VALUES (?,?)",
                    (composite_id, mid),
                )
            if axis_blob is not None and explained_variance_ratio is not None:
                db.execute(
                    """INSERT OR REPLACE INTO composite_axes
                       (composite_id, axis_vector, explained_variance_ratio)
                       VALUES (?,?,?)""",
                    (composite_id, axis_blob, explained_variance_ratio),
                )
            db.commit()

        await asyncio.to_thread(_insert)
        return composite_id

    # ── Boundary layers ──────────────────────────

    async def clear_boundary_layers(self, composite_id: str | None = None) -> None:
        """boundary_layers をクリア。composite_id 指定時はそのcompositeのみ。"""
        db = self._ensure_connected()

        def _clear() -> None:
            if composite_id:
                db.execute("DELETE FROM boundary_layers WHERE composite_id = ?", (composite_id,))
            else:
                db.execute("DELETE FROM boundary_layers")
            db.commit()

        await asyncio.to_thread(_clear)

    async def save_boundary_layers(
        self,
        composite_id: str,
        layers: list[tuple[str, int, int]],
    ) -> None:
        """boundary_layers を一括保存。layers: list of (member_id, layer_index, is_edge)。"""
        db = self._ensure_connected()

        def _save() -> None:
            db.executemany(
                "INSERT OR REPLACE INTO boundary_layers (composite_id, member_id, layer_index, is_edge) VALUES (?,?,?,?)",
                [(composite_id, mid, lidx, edge) for mid, lidx, edge in layers],
            )
            db.commit()

        await asyncio.to_thread(_save)

    async def fetch_composite_with_vectors(
        self, composite_id: str
    ) -> list[tuple[str, np.ndarray]]:
        """compositeのメンバーIDとベクトル(flow+delta concat)を取得。"""
        db = self._ensure_connected()

        def _fetch() -> list[tuple[str, bytes | None, bytes | None, bytes]]:
            rows = db.execute(
                """SELECT cm.member_id, e.flow_vector, e.delta_vector, e.vector
                   FROM composite_members cm
                   JOIN embeddings e ON e.memory_id = cm.member_id
                   WHERE cm.composite_id = ?""",
                (composite_id,),
            ).fetchall()
            return [
                (row["member_id"], row["flow_vector"], row["delta_vector"], bytes(row["vector"]))
                for row in rows
            ]

        raw = await asyncio.to_thread(_fetch)
        results: list[tuple[str, np.ndarray]] = []
        expected_dim = self._chive.vector_size * 2
        for mid, flow_blob, delta_blob, legacy_blob in raw:
            if flow_blob and delta_blob:
                flow = decode_vector(bytes(flow_blob))
                delta = decode_vector(bytes(delta_blob))
                vec = np.concatenate([flow, delta])
            else:
                legacy_vec = decode_vector(legacy_blob)
                if legacy_vec.shape[0] != expected_dim:
                    continue
                vec = legacy_vec
            results.append((mid, vec))
        return results

    async def fetch_composite_centroid(self, composite_id: str) -> np.ndarray | None:
        """compositeの重心ベクトル(flow+delta concat)を取得。"""
        db = self._ensure_connected()

        def _fetch() -> tuple[bytes | None, bytes | None, bytes | None]:
            row = db.execute(
                "SELECT flow_vector, delta_vector, vector FROM embeddings WHERE memory_id = ?",
                (composite_id,),
            ).fetchone()
            if not row:
                return None, None, None
            return (
                bytes(row["flow_vector"]) if row["flow_vector"] else None,
                bytes(row["delta_vector"]) if row["delta_vector"] else None,
                bytes(row["vector"]) if row["vector"] else None,
            )

        flow_blob, delta_blob, legacy_blob = await asyncio.to_thread(_fetch)
        if flow_blob and delta_blob:
            flow = decode_vector(flow_blob)
            delta = decode_vector(delta_blob)
            return np.concatenate([flow, delta])
        if legacy_blob:
            legacy_vec = decode_vector(legacy_blob)
            if legacy_vec.shape[0] != self._chive.vector_size * 2:
                return None  # skip unmigrated 768-dim vector
            return legacy_vec
        return None

    async def fetch_all_composite_ids(self, level: int | None = None) -> list[str]:
        """composite IDを取得。level=None で全 composite、level=N で指定レベルのみ。"""
        db = self._ensure_connected()

        def _fetch() -> list[str]:
            if level is None:
                rows = db.execute(
                    "SELECT DISTINCT id FROM memories WHERE level >= 1"
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT DISTINCT id FROM memories WHERE level = ?",
                    (level,),
                ).fetchall()
            return [row["id"] for row in rows]

        return await asyncio.to_thread(_fetch)

    async def fetch_all_composite_axes(self) -> dict[str, np.ndarray]:
        """全compositeの主成分軸を {composite_id: axis_vector} で返す。"""
        db = self._ensure_connected()

        def _fetch() -> dict[str, np.ndarray]:
            rows = db.execute(
                "SELECT composite_id, axis_vector FROM composite_axes"
            ).fetchall()
            result: dict[str, np.ndarray] = {}
            for row in rows:
                result[row["composite_id"]] = decode_vector(bytes(row["axis_vector"]))
            return result

        return await asyncio.to_thread(_fetch)

    async def cleanup_stale_composite_axes(self, expected_dim: int) -> int:
        """Delete composite_axes with mismatched dimensions (e.g. after model migration).

        Returns number of stale composites removed.
        """
        db = self._ensure_connected()

        def _cleanup() -> int:
            rows = db.execute(
                "SELECT composite_id, axis_vector FROM composite_axes"
            ).fetchall()
            stale_ids = []
            for row in rows:
                vec = decode_vector(bytes(row["axis_vector"]))
                if len(vec) != expected_dim:
                    stale_ids.append(row["composite_id"])
            if stale_ids:
                for cid in stale_ids:
                    db.execute("DELETE FROM composite_axes WHERE composite_id=?", (cid,))
                    db.execute("DELETE FROM composite_members WHERE composite_id=?", (cid,))
                    db.execute(
                        "DELETE FROM composite_intersections "
                        "WHERE composite_a=? OR composite_b=?",
                        (cid, cid),
                    )
                db.commit()
                logger.warning(
                    "Cleaned up %d stale composite_axes (expected %d-dim)",
                    len(stale_ids), expected_dim,
                )
            return len(stale_ids)

        return await asyncio.to_thread(_cleanup)

    async def fetch_all_composite_member_sets(self) -> dict[str, set[str]]:
        """全compositeのメンバーIDセットを {composite_id: set(member_ids)} で返す。"""
        db = self._ensure_connected()

        def _fetch() -> dict[str, set[str]]:
            composites: dict[str, set[str]] = {}
            for row in db.execute("SELECT composite_id, member_id FROM composite_members").fetchall():
                cid = row["composite_id"]
                if cid not in composites:
                    composites[cid] = set()
                composites[cid].add(row["member_id"])
            return composites

        return await asyncio.to_thread(_fetch)

    async def save_intersections(
        self,
        intersections: list[tuple[str, str, str, float, list[str]]],
    ) -> None:
        """composite_intersections を一括保存。既存データをクリアして入れ替え。"""
        db = self._ensure_connected()

        def _save() -> None:
            db.execute("DELETE FROM composite_intersections")
            if intersections:
                db.executemany(
                    """INSERT INTO composite_intersections
                       (composite_a, composite_b, intersection_type, axis_cosine, shared_member_ids)
                       VALUES (?,?,?,?,?)""",
                    [
                        (a, b, itype, cos, ",".join(shared))
                        for a, b, itype, cos, shared in intersections
                    ],
                )
            db.commit()

        await asyncio.to_thread(_save)

    async def get_intersection_nodes(
        self, memory_ids: list[str]
    ) -> dict[str, float]:
        """memory_idsの中から交差ノードを検出し、ブーストスコアを返す。

        transversal: +0.8, parallel: +0.3
        """
        if not memory_ids:
            return {}
        db = self._ensure_connected()

        def _calc() -> dict[str, float]:
            rows = db.execute(
                "SELECT intersection_type, shared_member_ids FROM composite_intersections"
            ).fetchall()
            if not rows:
                return {}

            id_set = set(memory_ids)
            scores: dict[str, float] = {}
            for row in rows:
                itype = row["intersection_type"]
                shared_str = row["shared_member_ids"]
                if not shared_str:
                    continue
                shared = shared_str.split(",")
                boost = 0.8 if itype == "transversal" else 0.3
                for mid in shared:
                    mid = mid.strip()
                    if mid in id_set:
                        # 最大ブーストを保持
                        scores[mid] = max(scores.get(mid, 0.0), boost)
            return scores

        return await asyncio.to_thread(_calc)

    # ── Boundary-aware recall helpers ────────────

    async def get_member_boundary_scores(
        self, memory_ids: list[str]
    ) -> dict[str, float]:
        """memory_ids の fuzziness スコアを取得。

        fuzziness = 全レイヤーで edge に分類された回数 / 全レイヤー数。
        0.0=常にcore, 1.0=常にedge。boundary_layers に存在しない ID は結果に含まない。
        """
        if not memory_ids:
            return {}
        db = self._ensure_connected()

        def _calc() -> dict[str, float]:
            placeholders = ",".join("?" for _ in memory_ids)
            rows = db.execute(
                f"""SELECT bl.member_id,
                           CAST(SUM(bl.is_edge) AS REAL) / COUNT(*) as fuzziness
                    FROM boundary_layers bl
                    WHERE bl.member_id IN ({placeholders})
                    GROUP BY bl.member_id""",
                memory_ids,
            ).fetchall()
            return {row["member_id"]: row["fuzziness"] for row in rows}

        return await asyncio.to_thread(_calc)

    async def select_active_boundary_layer(
        self, path_vec: np.ndarray, n_layers_max: int = 10,
    ) -> int:
        """経路ベクトルに最も aligned な boundary layer を選択。

        各レイヤーの edge メンバーの flow_vector 平均と path_vec の cos sim を計算。
        最も高い layer_index を返す。edge データがなければ 0 を返す。
        """
        db = self._ensure_connected()

        def _select() -> int:
            layer_rows = db.execute(
                "SELECT DISTINCT layer_index FROM boundary_layers ORDER BY layer_index"
            ).fetchall()
            if not layer_rows:
                return 0

            layer_indices = [r["layer_index"] for r in layer_rows][:n_layers_max]

            best_layer = 0
            best_sim = -2.0

            for lidx in layer_indices:
                edge_rows = db.execute(
                    """SELECT e.flow_vector
                       FROM boundary_layers bl
                       JOIN embeddings e ON e.memory_id = bl.member_id
                       WHERE bl.layer_index = ? AND bl.is_edge = 1
                         AND e.flow_vector IS NOT NULL""",
                    (lidx,),
                ).fetchall()
                if not edge_rows:
                    continue

                edge_vecs = [decode_vector(bytes(r["flow_vector"])) for r in edge_rows]
                edge_mean = np.mean(edge_vecs, axis=0).reshape(1, -1)
                sim = float(cosine_similarity(path_vec, edge_mean)[0])

                if sim > best_sim:
                    best_sim = sim
                    best_layer = lidx

            return best_layer

        return await asyncio.to_thread(_select)

    async def get_chain_boundary_scores(
        self,
        chain_ids: list[str],
        layer_index: int | None = None,
    ) -> dict[str, float]:
        """verb chain の boundary 近接スコアを計算 (flow_vector ベース)。

        layer_index 指定時: そのレイヤーの edge メンバーとの類似度ベース
        layer_index=None: fuzziness（全レイヤー横断）ベース

        Returns: {chain_id: boundary_score} (0.0〜1.0)
        """
        if not chain_ids:
            return {}

        db = self._ensure_connected()

        def _calc() -> dict[str, float]:
            placeholders = ",".join("?" for _ in chain_ids)
            chain_rows = db.execute(
                f"""SELECT chain_id, flow_vector FROM verb_chain_embeddings
                    WHERE chain_id IN ({placeholders}) AND flow_vector IS NOT NULL""",
                chain_ids,
            ).fetchall()
            if not chain_rows:
                return {}

            chain_vecs = {
                r["chain_id"]: decode_vector(bytes(r["flow_vector"]))
                for r in chain_rows
            }

            if layer_index is not None:
                edge_rows = db.execute(
                    """SELECT e.flow_vector
                       FROM boundary_layers bl
                       JOIN embeddings e ON e.memory_id = bl.member_id
                       WHERE bl.layer_index = ? AND bl.is_edge = 1
                         AND e.flow_vector IS NOT NULL""",
                    (layer_index,),
                ).fetchall()
            else:
                edge_rows = db.execute(
                    """SELECT e.flow_vector
                       FROM embeddings e
                       WHERE e.flow_vector IS NOT NULL AND e.memory_id IN (
                           SELECT bl.member_id
                           FROM boundary_layers bl
                           GROUP BY bl.member_id
                           HAVING CAST(SUM(bl.is_edge) AS REAL) / COUNT(*) > 0.5
                       )"""
                ).fetchall()

            if not edge_rows:
                return {cid: 0.0 for cid in chain_vecs}

            edge_vecs = np.stack([decode_vector(bytes(r["flow_vector"])) for r in edge_rows])

            result: dict[str, float] = {}
            for cid, cvec in chain_vecs.items():
                sims = cosine_similarity(cvec, edge_vecs)
                result[cid] = float(np.max(sims))

            return result

        return await asyncio.to_thread(_calc)

    async def find_adjacent_composites(
        self,
        composite_id: str,
        query_vec: np.ndarray,
        n_results: int = 3,
    ) -> list[tuple[str, float]]:
        """composite の edge メンバーの flow_vector 平均から隣接 composite を発見。

        Returns: [(adjacent_composite_id, similarity), ...] 自身を除外。
        """
        db = self._ensure_connected()

        def _find() -> list[tuple[str, float]]:
            edge_rows = db.execute(
                """SELECT e.flow_vector
                   FROM boundary_layers bl
                   JOIN embeddings e ON e.memory_id = bl.member_id
                   WHERE bl.composite_id = ? AND bl.layer_index = 0 AND bl.is_edge = 1
                     AND e.flow_vector IS NOT NULL""",
                (composite_id,),
            ).fetchall()
            if not edge_rows:
                return []

            edge_vecs = [decode_vector(bytes(r["flow_vector"])) for r in edge_rows]
            edge_mean = np.mean(edge_vecs, axis=0)

            other_rows = db.execute(
                """SELECT e.memory_id, e.flow_vector
                   FROM embeddings e
                   JOIN memories m ON m.id = e.memory_id
                   WHERE m.level >= 1 AND e.memory_id != ?
                     AND e.flow_vector IS NOT NULL""",
                (composite_id,),
            ).fetchall()
            if not other_rows:
                return []

            candidates: list[tuple[str, float]] = []
            for row in other_rows:
                other_vec = decode_vector(bytes(row["flow_vector"]))
                sim = float(cosine_similarity(edge_mean, other_vec.reshape(1, -1))[0])
                candidates.append((row["memory_id"], sim))

            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:n_results]

        return await asyncio.to_thread(_find)

    async def expand_composite_edges(
        self,
        composite_ids: list[str],
        query_vec: np.ndarray,
    ) -> list[Memory]:
        """composite の edge メンバーと隣接 composite の edge メンバーを展開して返す。
        横断的交差パートナーの共有メンバーも展開する。
        """
        if not composite_ids:
            return []

        all_member_ids: set[str] = set()
        db = self._ensure_connected()

        for cid in composite_ids:
            # edge メンバーを取得
            def _get_edges(comp_id: str = cid) -> list[str]:
                rows = db.execute(
                    """SELECT bl.member_id
                       FROM boundary_layers bl
                       WHERE bl.composite_id = ? AND bl.layer_index = 0 AND bl.is_edge = 1""",
                    (comp_id,),
                ).fetchall()
                return [row["member_id"] for row in rows]

            edge_ids = await asyncio.to_thread(_get_edges)
            all_member_ids.update(edge_ids)

            # 隣接 composite を発見
            adjacent = await self.find_adjacent_composites(cid, query_vec, n_results=3)
            for adj_id, _sim in adjacent:
                def _get_adj_edges(comp_id: str = adj_id) -> list[str]:
                    rows = db.execute(
                        """SELECT bl.member_id
                           FROM boundary_layers bl
                           WHERE bl.composite_id = ? AND bl.layer_index = 0 AND bl.is_edge = 1""",
                        (comp_id,),
                    ).fetchall()
                    return [row["member_id"] for row in rows]

                adj_edge_ids = await asyncio.to_thread(_get_adj_edges)
                all_member_ids.update(adj_edge_ids)

            # 横断的交差パートナーの共有メンバーを展開
            def _get_transversal_shared(comp_id: str = cid) -> list[str]:
                rows = db.execute(
                    """SELECT shared_member_ids FROM composite_intersections
                       WHERE intersection_type = 'transversal'
                         AND (composite_a = ? OR composite_b = ?)""",
                    (comp_id, comp_id),
                ).fetchall()
                ids: list[str] = []
                for row in rows:
                    shared_str = row["shared_member_ids"]
                    if shared_str:
                        ids.extend(mid.strip() for mid in shared_str.split(",") if mid.strip())
                return ids

            transversal_ids = await asyncio.to_thread(_get_transversal_shared)
            all_member_ids.update(transversal_ids)

        if not all_member_ids:
            return []

        return await self.get_by_ids(list(all_member_ids))

    async def fetch_verb_chain_templates(
        self,
    ) -> list[tuple[str, np.ndarray, list[str], list[str]]]:
        """全VerbChainの(chain_id, flow_vector, verbs_list, nouns_list)を取得。"""
        db = self._ensure_connected()

        def _fetch() -> list[tuple[str, bytes | None, bytes, str, str]]:
            rows = db.execute(
                """SELECT vc.id, vce.flow_vector, vce.vector, vc.all_verbs, vc.all_nouns
                   FROM verb_chains vc
                   JOIN verb_chain_embeddings vce ON vce.chain_id = vc.id"""
            ).fetchall()
            return [
                (
                    row["id"],
                    bytes(row["flow_vector"]) if row["flow_vector"] else None,
                    bytes(row["vector"]),
                    row["all_verbs"],
                    row["all_nouns"],
                )
                for row in rows
            ]

        raw = await asyncio.to_thread(_fetch)
        results: list[tuple[str, np.ndarray, list[str], list[str]]] = []
        for chain_id, flow_blob, legacy_blob, verbs_str, nouns_str in raw:
            if flow_blob:
                vec = decode_vector(flow_blob)
            else:
                legacy_vec = decode_vector(legacy_blob)
                if legacy_vec.shape[0] != self._chive.vector_size:
                    continue  # skip unmigrated 768-dim vectors
                vec = legacy_vec
            verbs = [v.strip() for v in verbs_str.split(",") if v.strip()]
            nouns = [n.strip() for n in nouns_str.split(",") if n.strip()]
            results.append((chain_id, vec, verbs, nouns))
        return results

    # ── Template Biases ──────────────────────────

    async def fetch_template_biases(self) -> dict[str, float]:
        """全バイアスを {chain_id: bias_weight} で返す。"""
        db = self._ensure_connected()

        def _fetch() -> dict[str, float]:
            rows = db.execute("SELECT chain_id, bias_weight FROM template_biases").fetchall()
            return {row["chain_id"]: row["bias_weight"] for row in rows}

        return await asyncio.to_thread(_fetch)

    async def save_template_biases(self, biases: list[tuple[str, float, int]]) -> None:
        """バッチ upsert (chain_id, bias_weight, update_count)。"""
        if not biases:
            return
        db = self._ensure_connected()
        now = datetime.now(timezone.utc).isoformat()

        def _save() -> None:
            db.executemany(
                """INSERT INTO template_biases (chain_id, bias_weight, update_count, last_updated)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(chain_id) DO UPDATE SET
                       bias_weight = excluded.bias_weight,
                       update_count = excluded.update_count,
                       last_updated = excluded.last_updated""",
                [(cid, weight, count, now) for cid, weight, count in biases],
            )
            db.commit()

        await asyncio.to_thread(_save)

    async def decay_template_biases(
        self, factor: float, prune_threshold: float
    ) -> dict[str, int]:
        """全バイアスを factor 倍に減衰し、prune_threshold 以下を刈り取る。"""
        db = self._ensure_connected()

        def _decay() -> dict[str, int]:
            db.execute(
                "UPDATE template_biases SET bias_weight = bias_weight * ?",
                (factor,),
            )
            cursor = db.execute(
                "DELETE FROM template_biases WHERE bias_weight <= ?",
                (prune_threshold,),
            )
            pruned = cursor.rowcount
            remaining = db.execute("SELECT COUNT(*) FROM template_biases").fetchone()[0]
            db.commit()
            return {"biases_decayed": remaining + pruned, "biases_pruned": pruned}

        return await asyncio.to_thread(_decay)

    # ── Hopfield ─────────────────────────────────

    async def hopfield_load(self) -> int:
        """Load all embeddings from SQLite into Hopfield network.

        Uses flow+delta concat (600-dim) for pattern storage.
        Falls back to legacy vector column if flow/delta not yet migrated.
        """
        db = self._ensure_connected()

        def _fetch() -> list[tuple[str, bytes | None, bytes | None, bytes, str]]:
            sql = (
                "SELECT e.memory_id, e.flow_vector, e.delta_vector, e.vector, m.normalized_content"
                " FROM embeddings e JOIN memories m ON m.id = e.memory_id"
            )
            return db.execute(sql).fetchall()

        rows = await asyncio.to_thread(_fetch)
        if not rows:
            self._hopfield.store([], [], [])
            return 0

        ids: list[str] = []
        embeddings: list[list[float]] = []
        contents: list[str] = []
        for r in rows:
            mem_id = r[0]
            flow_blob = r[1]
            delta_blob = r[2]
            legacy_blob = r[3]
            content = r[4]

            if flow_blob and delta_blob:
                flow = decode_vector(bytes(flow_blob))
                delta = decode_vector(bytes(delta_blob))
                vec = np.concatenate([flow, delta])
            else:
                legacy_vec = decode_vector(bytes(legacy_blob))
                if legacy_vec.shape[0] != self._chive.vector_size * 2:
                    continue  # skip unmigrated 768-dim vectors
                vec = legacy_vec

            ids.append(mem_id)
            embeddings.append(vec.tolist())
            contents.append(content)

        self._hopfield.store(embeddings, ids, contents)
        return self._hopfield.n_memories

    async def hopfield_recall(
        self,
        query: str,
        n_results: int = 5,
        beta: float | None = None,
        auto_load: bool = True,
    ) -> list[HopfieldRecallResult]:
        if auto_load and not self._hopfield.is_loaded:
            await self.hopfield_load()
        if not self._hopfield.is_loaded:
            return []

        original_beta = self._hopfield.beta
        if beta is not None:
            self._hopfield.beta = beta

        try:
            normalized_query = normalize_japanese(query)
            q_flow, q_delta = await self._encode_text(normalized_query)
            query_emb = np.concatenate([q_flow, q_delta]).tolist()
            _, similarities = self._hopfield.retrieve(query_emb)
            results = self._hopfield.recall_results(similarities, k=n_results)
        finally:
            self._hopfield.beta = original_beta

        return results

    # ── Diagnostics helper ───────────────────────

    def _build_divergent_diagnostics(
        self,
        context: str,
        selected: list[Memory],
        prediction_errors: list[float],
        novelty_scores: list[float],
    ) -> dict[str, Any]:
        avg_prediction_error = sum(prediction_errors) / len(prediction_errors) if prediction_errors else 0.0
        avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        predictive = PredictiveDiagnostics(
            avg_prediction_error=avg_prediction_error,
            avg_novelty=avg_novelty,
        )
        return {
            "context": context,
            "selected_count": len(selected),
            "diversity_score": diversity_score(selected),
            "avg_prediction_error": predictive.avg_prediction_error,
            "avg_novelty": predictive.avg_novelty,
        }

    # ── chiVe 2-vector migration ─────────────────

    async def migrate_to_chive_2vec(self) -> dict[str, int]:
        """Migrate all existing memories and verb_chains to chiVe 2-vector.

        - Re-encodes all memories via encode_text (sudachipy extraction → chiVe)
        - Re-encodes all verb_chains via encode_chain (all_verbs/all_nouns → chiVe)
        - Stores flow_vector + delta_vector + concat in each row
        - Marks completion in meta.embedding_schema = 'v2_chive'
        """
        import logging

        logger = logging.getLogger(__name__)
        db = self._ensure_connected()
        assert self._chive is not None

        # Check if already migrated
        TARGET_SCHEMA = "v2_chive_full"

        def _check() -> str | None:
            row = db.execute(
                "SELECT value FROM meta WHERE key = 'embedding_schema'"
            ).fetchone()
            return row[0] if row else None

        current_schema = await asyncio.to_thread(_check)
        if current_schema == TARGET_SCHEMA:
            logger.info("migrate_to_chive_2vec: already at %s, skipping", TARGET_SCHEMA)
            return {"memories_migrated": 0, "chains_migrated": 0}

        # If already at v2_chive (partial), skip memory/chain re-encoding
        skip_base_migration = current_schema == "v2_chive"

        # 1. Migrate memories (batch of 100) — skip if already done
        mem_count = 0
        chain_count = 0

        if skip_base_migration:
            logger.info("migrate_to_chive_2vec: base migration done, skipping to composites")
        else:
            mem_count, chain_count = await self._migrate_base_memories_and_chains(db, logger)

        # 2b-2c: composites + orphan cleanup (always run if not at full)
        comp_count, orphan_count = await self._migrate_composites_and_cleanup(db, logger)

        # 3. Mark completion
        def _mark_done() -> None:
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('embedding_schema', ?)",
                (TARGET_SCHEMA,),
            )
            # Invalidate recall_index so it gets rebuilt
            db.execute("DELETE FROM meta WHERE key = 'recall_index_built_at'")
            # Reset flow_vector_version so verb_chain backfill recalculates
            db.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('flow_vector_version', '4')"
            )
            db.commit()

        await asyncio.to_thread(_mark_done)

        logger.info(
            "migrate_to_chive_2vec: done. memories=%d, chains=%d, composites=%d, orphans_cleaned=%d",
            mem_count, chain_count, comp_count, orphan_count,
        )
        return {"memories_migrated": mem_count, "chains_migrated": chain_count}

    async def _migrate_base_memories_and_chains(
        self, db: sqlite3.Connection, logger: Any,
    ) -> tuple[int, int]:
        """Re-encode all memories and verb_chains with chiVe 2-vector."""
        def _get_memory_ids() -> list[str]:
            rows = db.execute("SELECT id FROM memories").fetchall()
            return [r[0] for r in rows]

        mem_ids = await asyncio.to_thread(_get_memory_ids)

        def _get_memory_content(mid: str) -> str:
            row = db.execute(
                "SELECT normalized_content FROM memories WHERE id = ?", (mid,)
            ).fetchone()
            return row[0] if row and row[0] else ""

        mem_count = 0
        for batch_start in range(0, len(mem_ids), 100):
            batch = mem_ids[batch_start:batch_start + 100]
            updates: list[tuple[bytes, bytes, bytes, str]] = []
            for mid in batch:
                content = await asyncio.to_thread(_get_memory_content, mid)
                if not content:
                    continue
                flow_vec, delta_vec = self._encode_text_sync(content)
                concat = np.concatenate([flow_vec, delta_vec])
                updates.append((
                    encode_vector(concat),
                    encode_vector(flow_vec),
                    encode_vector(delta_vec),
                    mid,
                ))

            def _batch_update(upd: list[tuple[bytes, bytes, bytes, str]] = updates) -> int:
                for vec_blob, flow_blob, delta_blob, mid in upd:
                    db.execute(
                        "UPDATE embeddings SET vector = ?, flow_vector = ?, delta_vector = ? WHERE memory_id = ?",
                        (vec_blob, flow_blob, delta_blob, mid),
                    )
                db.commit()
                return len(upd)

            count = await asyncio.to_thread(_batch_update)
            mem_count += count
            logger.info("migrate_to_chive_2vec: memories %d/%d", mem_count, len(mem_ids))

        # Migrate verb_chains (batch of 100)
        def _get_chain_data() -> list[tuple[str, str, str]]:
            rows = db.execute(
                "SELECT id, all_verbs, all_nouns FROM verb_chains"
            ).fetchall()
            return [(r[0], r[1] or "", r[2] or "") for r in rows]

        chain_data = await asyncio.to_thread(_get_chain_data)

        chain_count = 0
        for batch_start in range(0, len(chain_data), 100):
            batch = chain_data[batch_start:batch_start + 100]
            updates: list[tuple[bytes, bytes, bytes, str]] = []
            for chain_id, verbs_str, nouns_str in batch:
                verbs = [v.strip() for v in verbs_str.split(",") if v.strip()]
                nouns = [n.strip() for n in nouns_str.split(",") if n.strip()]
                if not verbs:
                    continue
                flow_vec, delta_vec = self._chive.encode_chain(verbs, nouns)
                concat = np.concatenate([flow_vec, delta_vec])
                updates.append((
                    encode_vector(concat),
                    encode_vector(flow_vec),
                    encode_vector(delta_vec),
                    chain_id,
                ))

            def _batch_update_chains(upd: list[tuple[bytes, bytes, bytes, str]] = updates) -> int:
                for vec_blob, flow_blob, delta_blob, cid in upd:
                    db.execute(
                        "UPDATE verb_chain_embeddings SET vector = ?, flow_vector = ?, delta_vector = ? WHERE chain_id = ?",
                        (vec_blob, flow_blob, delta_blob, cid),
                    )
                db.commit()
                return len(upd)

            count = await asyncio.to_thread(_batch_update_chains)
            chain_count += count
            logger.info("migrate_to_chive_2vec: chains %d/%d", chain_count, len(chain_data))

        return mem_count, chain_count

    async def _migrate_composites_and_cleanup(
        self, db: sqlite3.Connection, logger: Any,
    ) -> tuple[int, int]:
        """Migrate composite vectors from member centroids + cleanup orphans."""
        # Composite memories: re-compute from member centroids
        def _get_composites() -> list[tuple[str, list[str]]]:
            cids = db.execute(
                "SELECT DISTINCT composite_id FROM composite_members"
            ).fetchall()
            result = []
            for row in cids:
                cid = row[0]
                members = db.execute(
                    "SELECT member_id FROM composite_members WHERE composite_id = ?",
                    (cid,),
                ).fetchall()
                result.append((cid, [m[0] for m in members]))
            return result

        composites = await asyncio.to_thread(_get_composites)
        comp_count = 0
        for cid, member_ids in composites:
            def _get_member_vecs(mids: list[str] = member_ids) -> list[tuple[bytes, bytes]]:
                vecs = []
                for mid in mids:
                    row = db.execute(
                        "SELECT flow_vector, delta_vector FROM embeddings WHERE memory_id = ?",
                        (mid,),
                    ).fetchone()
                    if row and row[0] and row[1]:
                        vecs.append((bytes(row[0]), bytes(row[1])))
                return vecs

            member_vecs = await asyncio.to_thread(_get_member_vecs)
            if not member_vecs:
                continue

            flows = np.array([decode_vector(v[0]) for v in member_vecs], dtype=np.float32)
            deltas = np.array([decode_vector(v[1]) for v in member_vecs], dtype=np.float32)
            centroid_flow = flows.mean(axis=0)
            centroid_delta = deltas.mean(axis=0)
            norm_f = np.linalg.norm(centroid_flow)
            norm_d = np.linalg.norm(centroid_delta)
            if norm_f > 0:
                centroid_flow /= norm_f
            if norm_d > 0:
                centroid_delta /= norm_d
            concat = np.concatenate([centroid_flow, centroid_delta])

            def _update_composite(
                vec_blob: bytes = encode_vector(concat),
                flow_blob: bytes = encode_vector(centroid_flow),
                delta_blob: bytes = encode_vector(centroid_delta),
                composite_id: str = cid,
            ) -> None:
                db.execute(
                    "UPDATE embeddings SET vector = ?, flow_vector = ?, delta_vector = ? WHERE memory_id = ?",
                    (vec_blob, flow_blob, delta_blob, composite_id),
                )
                db.commit()

            await asyncio.to_thread(_update_composite)
            comp_count += 1

        logger.info("migrate_to_chive_2vec: composites %d/%d", comp_count, len(composites))

        # Clean up orphaned verb_chain_embeddings
        def _cleanup_orphans() -> int:
            cur = db.execute(
                "DELETE FROM verb_chain_embeddings WHERE chain_id NOT IN (SELECT id FROM verb_chains)"
            )
            db.commit()
            return cur.rowcount

        orphan_count = await asyncio.to_thread(_cleanup_orphans)
        logger.info("migrate_to_chive_2vec: cleaned %d orphaned verb_chain_embeddings", orphan_count)

        return comp_count, orphan_count
