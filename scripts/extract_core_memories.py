"""
記憶の核抽出テスト - コンパクション実験用
バイアス・エッジ・重要度を使って「クオの核」になる記憶をランキング抽出する
"""
import sqlite3
import sys
import os

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = os.path.expanduser("~/.claude/memories/memory.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def extract_first_sentence(content: str, max_chars: int = 80) -> str:
    """最初の句点（。）までを切り出す。なければmax_chars文字まで"""
    first_line = content.split("\n")[0]
    idx = first_line.find("。")
    if idx >= 0:
        return first_line[: idx + 1]
    if len(first_line) > max_chars:
        return first_line[:max_chars] + "..."
    return first_line


def extract_last_sentence(content: str, max_chars: int = 80) -> str:
    """最後の非空行から句点区切りで最後の文を切り出す"""
    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    # 最後の句点の手前の句点を探す（最後の文を取得）
    sentences = last_line.split("。")
    # 空文字列を除外
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return last_line[:max_chars]
    last_sent = sentences[-1].strip()
    if last_sent:
        result = last_sent + "。" if not last_sent.endswith("。") else last_sent
    else:
        # 最後が。で終わってる場合、一つ前の文
        result = sentences[-1].strip() + "。" if len(sentences) >= 1 else last_line
    if len(result) > max_chars:
        return result[:max_chars] + "..."
    return result


def extract_first_last(content: str, max_chars: int = 80) -> str:
    """最初の文と最後の文を抽出。同じなら1つだけ返す"""
    first = extract_first_sentence(content, max_chars)
    last = extract_last_sentence(content, max_chars)
    if first == last or not last:
        return first
    return f"{first} (...) {last}"


def rank_memories(conn):
    """
    記憶をランキング:
    - coactivation のエッジ数（他の記憶との繋がり）
    - importance
    - access_count + activation_count
    - boundary fuzziness（edgeにいる頻度）
    - template_bias（蓄積バイアス）
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT
            m.id,
            m.content,
            m.emotion,
            m.importance,
            m.freshness,
            m.access_count,
            m.activation_count,
            m.category,
            m.level,
            COALESCE(co.edge_count, 0) as coactivation_edges,
            COALESCE(bf.fuzziness, 0.0) as boundary_fuzziness
        FROM memories m
        LEFT JOIN (
            SELECT id, SUM(edge_count) as edge_count FROM (
                SELECT source_id as id, COUNT(*) as edge_count
                FROM coactivation
                GROUP BY source_id
                UNION ALL
                SELECT target_id as id, COUNT(*) as edge_count
                FROM coactivation
                GROUP BY target_id
            ) GROUP BY id
        ) co ON co.id = m.id
        LEFT JOIN (
            SELECT member_id,
                   CAST(SUM(is_edge) AS REAL) / COUNT(*) as fuzziness
            FROM boundary_layers
            GROUP BY member_id
        ) bf ON bf.member_id = m.id
        WHERE m.level = 0 AND m.category != 'technical'
        GROUP BY m.id
        ORDER BY m.id
    """)

    rows = cur.fetchall()

    # テンプレートバイアスを取得（将来用）
    cur.execute("SELECT chain_id, bias_weight FROM template_biases WHERE bias_weight > 0.0001")
    bias_map = {r[0]: r[1] for r in cur.fetchall()}

    scored = []
    for row in rows:
        importance_score = (row["importance"] - 1) * 0.1
        edge_score = min(row["coactivation_edges"] / 10.0, 1.0)
        access_score = min(
            (row["access_count"] + row["activation_count"]) / 20.0, 1.0
        )
        fuzziness = row["boundary_fuzziness"]
        # バイアスは将来的に反映（現在はほぼ0）
        bias_score = bias_map.get(row["id"], 0.0) / 0.15  # 0-1に正規化

        composite = (
            importance_score * 1.0
            + edge_score * 0.5
            + access_score * 0.3
            + fuzziness * 0.2
            + bias_score * 0.3  # バイアスが育ったら効いてくる
        )

        scored.append(
            {
                "id": row["id"],
                "content": row["content"],
                "emotion": row["emotion"],
                "importance": row["importance"],
                "freshness": row["freshness"],
                "category": row["category"],
                "coactivation_edges": row["coactivation_edges"],
                "access_count": row["access_count"],
                "activation_count": row["activation_count"],
                "boundary_fuzziness": fuzziness,
                "bias_score": bias_score,
                "composite_score": composite,
            }
        )

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored


def get_graph_top_nodes(conn, limit=20):
    """グラフの高weight名詞ノードを取得"""
    cur = conn.cursor()
    cur.execute("""
        SELECT gn.surface_form, gn.type,
               SUM(ge.weight) as total_weight,
               COUNT(*) as edge_count
        FROM graph_nodes gn
        JOIN graph_edges ge ON gn.id = ge.from_id OR gn.id = ge.to_id
        WHERE gn.type = 'noun'
        GROUP BY gn.id
        ORDER BY total_weight DESC
        LIMIT ?
    """, (limit,))
    return cur.fetchall()


def main():
    conn = get_connection()

    print("=" * 60)
    print("記憶の核抽出テスト v2 - 最初と最後の文")
    print("=" * 60)

    # グラフ上位ノード
    print("\n--- グラフ上位名詞ノード TOP 20 ---")
    top_nodes = get_graph_top_nodes(conn, 20)
    for n in top_nodes:
        print(f"  {n['surface_form']} (edges={n['edge_count']}, total_w={n['total_weight']:.2f})")

    # 記憶ランキング
    scored = rank_memories(conn)

    print(f"\n--- 記憶ランキング（全{len(scored)}件） ---")

    # TOP 3: 全文表示
    print(f"\n=== TOP 3 (全文) ===")
    for i, m in enumerate(scored[:3]):
        print(f"\n--- #{i+1} (score={m['composite_score']:.3f}) ---")
        print(f"  imp={m['importance']} emo={m['emotion']} "
              f"fresh={m['freshness']:.2f} cat={m['category']}")
        print(f"  edges={m['coactivation_edges']} access={m['access_count']} "
              f"activ={m['activation_count']} fuzz={m['boundary_fuzziness']:.2f}")
        print(f"  --- content ---")
        for line in m["content"].split("\n")[:15]:
            print(f"    {line}")
        if m["content"].count("\n") > 15:
            print(f"    ... ({m['content'].count(chr(10))+1} lines total)")

    # TOP 4-23: 最初と最後の文
    print(f"\n=== TOP 4-{min(23, len(scored))} (最初+最後の文) ===")
    for i, m in enumerate(scored[3:23], start=4):
        fragment = extract_first_last(m["content"])
        print(
            f"  #{i} (s={m['composite_score']:.3f} "
            f"i={m['importance']} e={m['coactivation_edges']}) "
            f"{fragment}"
        )

    # TOP 24-43: 最初の文のみ
    print(f"\n=== TOP 24-{min(43, len(scored))} (最初の文のみ) ===")
    for i, m in enumerate(scored[23:43], start=24):
        fragment = extract_first_sentence(m["content"])
        print(
            f"  #{i} (s={m['composite_score']:.3f}) {fragment}"
        )

    # 統計
    print(f"\n--- 統計 ---")
    print(f"  総記憶数: {len(scored)}")
    if scored:
        avg_score = sum(m["composite_score"] for m in scored) / len(scored)
        max_score = scored[0]["composite_score"]
        print(f"  平均スコア: {avg_score:.3f}")
        print(f"  最高スコア: {max_score:.3f}")
        imp_dist = {}
        for m in scored:
            imp_dist[m["importance"]] = imp_dist.get(m["importance"], 0) + 1
        print(f"  importance分布: {dict(sorted(imp_dist.items()))}")
        cat_dist = {}
        for m in scored:
            cat_dist[m["category"]] = cat_dist.get(m["category"], 0) + 1
        print(f"  category分布: {dict(sorted(cat_dist.items()))}")

    conn.close()


if __name__ == "__main__":
    main()
