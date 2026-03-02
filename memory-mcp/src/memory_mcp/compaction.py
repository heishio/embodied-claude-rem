"""Core memory compaction - extract and write memory essence to MEMORY.md."""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Section markers in MEMORY.md
SECTION_START = "## 記憶の核"
SECTION_END_PATTERN = re.compile(r"^## ", re.MULTILINE)

# Compaction parameters
TOP_FULL = 3  # Number of memories shown in full
TOP_FRAGMENTS = 20  # Number of memories shown as first+last sentence
TOP_GRAPH_NODES = 15  # Number of top graph noun nodes


def _extract_first_sentence(content: str, max_chars: int = 80) -> str:
    first_line = content.split("\n")[0]
    idx = first_line.find("。")
    if idx >= 0:
        return first_line[: idx + 1]
    if len(first_line) > max_chars:
        return first_line[:max_chars] + "..."
    return first_line


def _extract_last_sentence(content: str, max_chars: int = 80) -> str:
    lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    sentences = [s for s in last_line.split("。") if s.strip()]
    if not sentences:
        return last_line[:max_chars]
    result = sentences[-1].strip()
    if not result.endswith("。"):
        result += "。"
    if len(result) > max_chars:
        return result[:max_chars] + "..."
    return result


def _extract_first_last(content: str, max_chars: int = 80) -> str:
    first = _extract_first_sentence(content, max_chars)
    last = _extract_last_sentence(content, max_chars)
    if first == last or not last or last == "。":
        return first
    return f"{first} ... {last}"


def compact_core_memories(db_path: str, memory_md_path: str) -> dict:
    """Extract core memories and write to MEMORY.md.

    Returns stats about the compaction.
    """
    if not memory_md_path:
        return {"skipped": True, "reason": "no MEMORY_MD_PATH configured"}

    md_path = Path(memory_md_path)
    if not md_path.parent.exists():
        return {"skipped": True, "reason": f"directory not found: {md_path.parent}"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        ranked = _rank_memories(conn)
        graph_nodes = _get_graph_top_nodes(conn, TOP_GRAPH_NODES)
        section_text = _format_section(ranked, graph_nodes)
        _update_memory_md(md_path, section_text)

        return {
            "compacted": True,
            "total_memories": len(ranked),
            "top_full": min(TOP_FULL, len(ranked)),
            "top_fragments": min(TOP_FRAGMENTS, max(0, len(ranked) - TOP_FULL)),
            "graph_nodes": len(graph_nodes),
        }
    except Exception as e:
        logger.error(f"Compaction failed: {e}")
        return {"compacted": False, "error": str(e)}
    finally:
        conn.close()


def _rank_memories(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()

    cur.execute("""
        SELECT
            m.id,
            m.content,
            m.importance,
            m.freshness,
            m.category,
            COALESCE(co.edge_count, 0) as coactivation_edges,
            m.access_count,
            m.activation_count,
            COALESCE(bf.fuzziness, 0.0) as boundary_fuzziness
        FROM memories m
        LEFT JOIN (
            SELECT id, SUM(edge_count) as edge_count FROM (
                SELECT source_id as id, COUNT(*) as edge_count
                FROM coactivation GROUP BY source_id
                UNION ALL
                SELECT target_id as id, COUNT(*) as edge_count
                FROM coactivation GROUP BY target_id
            ) GROUP BY id
        ) co ON co.id = m.id
        LEFT JOIN (
            SELECT member_id,
                   CAST(SUM(is_edge) AS REAL) / COUNT(*) as fuzziness
            FROM boundary_layers GROUP BY member_id
        ) bf ON bf.member_id = m.id
        WHERE m.level = 0 AND m.category != 'technical'
        GROUP BY m.id
    """)

    # Template biases (for future use)
    cur2 = conn.cursor()
    cur2.execute(
        "SELECT chain_id, bias_weight FROM template_biases WHERE bias_weight > 0.0001"
    )
    bias_map = {r[0]: r[1] for r in cur2.fetchall()}

    scored = []
    for row in cur.fetchall():
        importance_score = (row["importance"] - 1) * 0.1
        edge_score = min(row["coactivation_edges"] / 10.0, 1.0)
        access_score = min(
            (row["access_count"] + row["activation_count"]) / 20.0, 1.0
        )
        fuzziness = row["boundary_fuzziness"]
        bias_score = bias_map.get(row["id"], 0.0) / 0.15

        composite = (
            importance_score * 1.0
            + edge_score * 0.5
            + access_score * 0.3
            + fuzziness * 0.2
            + bias_score * 0.3
        )

        scored.append(
            {
                "content": row["content"],
                "importance": row["importance"],
                "category": row["category"],
                "composite_score": composite,
            }
        )

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def _get_graph_top_nodes(conn: sqlite3.Connection, limit: int) -> list[tuple[str, float]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT gn.surface_form, SUM(ge.weight) as total_weight
        FROM graph_nodes gn
        JOIN graph_edges ge ON gn.id = ge.from_id OR gn.id = ge.to_id
        WHERE gn.type = 'noun'
        GROUP BY gn.id
        ORDER BY total_weight DESC
        LIMIT ?
    """,
        (limit,),
    )
    return [(r[0], r[1]) for r in cur.fetchall()]


def _format_section(ranked: list[dict], graph_nodes: list[tuple[str, float]]) -> str:
    lines = [SECTION_START, ""]

    # Graph nodes as keyword cloud
    if graph_nodes:
        node_strs = [n[0] for n in graph_nodes]
        lines.append(f"核語: {', '.join(node_strs)}")
        lines.append("")

    # TOP full memories
    for i, m in enumerate(ranked[:TOP_FULL]):
        content = m["content"]
        # Truncate very long memories
        content_lines = content.split("\n")
        if len(content_lines) > 8:
            content = "\n".join(content_lines[:8]) + "\n..."
        lines.append(f"**[{i+1}]** {content}")
        lines.append("")

    # Fragment memories (first + last sentence)
    if len(ranked) > TOP_FULL:
        fragments = []
        for m in ranked[TOP_FULL : TOP_FULL + TOP_FRAGMENTS]:
            fragment = _extract_first_last(m["content"])
            fragments.append(f"- {fragment}")
        lines.extend(fragments)
        lines.append("")

    return "\n".join(lines)


def _update_memory_md(md_path: Path, section_text: str) -> None:
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
    else:
        content = "# Embodied Claude - Memory\n\n"

    # Find and replace existing section, or append
    section_start_idx = content.find(SECTION_START)

    if section_start_idx >= 0:
        # Find the next ## heading after our section
        after_section = content[section_start_idx + len(SECTION_START) :]
        match = SECTION_END_PATTERN.search(after_section)
        if match:
            section_end_idx = section_start_idx + len(SECTION_START) + match.start()
            content = content[:section_start_idx] + section_text + content[section_end_idx:]
        else:
            # Our section is the last one
            content = content[:section_start_idx] + section_text
    else:
        # Append new section
        if not content.endswith("\n"):
            content += "\n"
        content += "\n" + section_text

    md_path.write_text(content, encoding="utf-8")
    logger.info(f"Updated MEMORY.md core section at {md_path}")
