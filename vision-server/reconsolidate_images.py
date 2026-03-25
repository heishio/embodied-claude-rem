"""画像compositeの再構築スクリプト（consolidateの画像部分だけ）

使い方:
  python reconsolidate_images.py          # composite再構築 + /reload-refs
  python reconsolidate_images.py --dry    # dry run（DB書き込みなし）
"""
import os
import sqlite3
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import urllib.request

DB_PATH = Path.home() / ".claude" / "memories" / "memory.db"
VISION_SERVER = "http://127.0.0.1:8100"
SIMILARITY_THRESHOLD = 0.75
MIN_GROUP_SIZE = 2
MAX_GROUP_SIZE = 8


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def encode_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def fetch_embeddings(conn):
    """person_ratio >= 0.1 の image_embeddings を取得"""
    rows = conn.execute(
        """SELECT id, delta_vector, flow_vector, face_vector, tag
           FROM image_embeddings
           WHERE person_ratio >= 0.1 AND delta_vector IS NOT NULL AND freshness >= 0.1"""
    ).fetchall()
    result = []
    for rid, dv, fv, fcv, tag in rows:
        entry = {
            "id": rid,
            "delta": decode_vector(dv),
            "flow": decode_vector(fv) if fv else None,
            "face": decode_vector(fcv) if fcv else None,
            "tag": tag,
        }
        result.append(entry)
    return result


def cluster_union_find(embeddings, threshold):
    """delta_vectorのコサイン類似度でUnion-Findクラスタリング"""
    n = len(embeddings)
    deltas = np.array([e["delta"] for e in embeddings])
    # L2正規化
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    deltas_n = deltas / np.clip(norms, 1e-10, None)
    # 類似度行列
    sim_matrix = deltas_n @ deltas_n.T

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                union(i, j)

    # グループ化
    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # MIN_GROUP_SIZE以上のグループだけ
    valid = {k: v for k, v in groups.items() if len(v) >= MIN_GROUP_SIZE}

    # MAX_GROUP_SIZE制限
    for k, members in valid.items():
        if len(members) > MAX_GROUP_SIZE:
            # グループ内類似度合計が高い順に上位k個
            sims = []
            for i in members:
                s = sum(sim_matrix[i, j] for j in members if j != i)
                sims.append((i, s))
            sims.sort(key=lambda x: x[1], reverse=True)
            valid[k] = [x[0] for x in sims[:MAX_GROUP_SIZE]]

    return list(valid.values())


def compute_centroids(embeddings, member_indices):
    """グループのdelta/flow/face重心を計算"""
    members = [embeddings[i] for i in member_indices]

    # delta centroid
    delta_vecs = np.array([m["delta"] for m in members])
    delta_mean = delta_vecs.mean(axis=0)
    delta_centroid = delta_mean / np.linalg.norm(delta_mean)

    # flow centroid
    flow_vecs = [m["flow"] for m in members if m["flow"] is not None]
    flow_centroid = None
    if flow_vecs:
        flow_mean = np.mean(flow_vecs, axis=0)
        flow_centroid = flow_mean / np.linalg.norm(flow_mean)

    # face centroid
    face_vecs = [m["face"] for m in members if m["face"] is not None]
    face_centroid = None
    if face_vecs:
        face_mean = np.mean(face_vecs, axis=0)
        face_centroid = face_mean / np.linalg.norm(face_mean)

    # tag (majority vote)
    tags = [m["tag"] for m in members if m["tag"] is not None]
    tag = Counter(tags).most_common(1)[0][0] if tags else None

    return delta_centroid, flow_centroid, face_centroid, tag


def save_composite(conn, member_ids, delta_c, flow_c, face_c, tag, member_count):
    """image_composites + composite_members に保存"""
    cid = f"img-{uuid.uuid4()}"
    now = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    conn.execute(
        """INSERT INTO image_composites
           (id, delta_centroid, flow_centroid, face_centroid, member_count, freshness, tag, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 1.0, ?, ?, ?)""",
        (
            cid,
            encode_vector(delta_c),
            encode_vector(flow_c) if flow_c is not None else None,
            encode_vector(face_c) if face_c is not None else None,
            member_count,
            tag,
            now,
            now,
        ),
    )
    for mid in member_ids:
        conn.execute(
            "INSERT OR IGNORE INTO composite_members (composite_id, member_id) VALUES (?, ?)",
            (cid, mid),
        )
    return cid


def get_existing_members(conn):
    """既存のcomposite_membersを取得（重複防止用）"""
    rows = conn.execute(
        "SELECT composite_id, member_id FROM composite_members WHERE composite_id LIKE 'img-%'"
    ).fetchall()
    existing = {}
    for cid, mid in rows:
        existing.setdefault(cid, set()).add(mid)
    return existing


def main():
    dry = "--dry" in sys.argv
    t0 = time.time()

    conn = sqlite3.connect(str(DB_PATH))

    # 1. データ取得
    embeddings = fetch_embeddings(conn)
    print(f"対象embedding: {len(embeddings)}件 (person_ratio >= 0.1)")

    if len(embeddings) < MIN_GROUP_SIZE:
        print("クラスタリングに必要な件数が不足")
        conn.close()
        return

    # 2. クラスタリング
    groups = cluster_union_find(embeddings, SIMILARITY_THRESHOLD)
    print(f"クラスタ: {len(groups)}個")

    # 3. 既存メンバー構成を取得（重複スキップ用）
    existing = get_existing_members(conn)
    existing_sets = [frozenset(v) for v in existing.values()]

    # 4. 旧composite削除 + 再構築
    if not dry:
        conn.execute("DELETE FROM composite_members WHERE composite_id LIKE 'img-%'")
        conn.execute("DELETE FROM image_composites WHERE id LIKE 'img-%'")
        conn.commit()
        print("旧image_composites削除")

    # 5. 新composite作成
    created = 0
    for group in groups:
        member_ids = [embeddings[i]["id"] for i in group]
        delta_c, flow_c, face_c, tag = compute_centroids(embeddings, group)

        if dry:
            tag_str = tag or "-"
            has_face = "Y" if face_c is not None else "N"
            print(f"  [{created}] members={len(group)} tag={tag_str} face={has_face}")
        else:
            cid = save_composite(conn, member_ids, delta_c, flow_c, face_c, tag, len(member_ids))
            tag_str = tag or "-"
            print(f"  {cid}: members={len(member_ids)} tag={tag_str}")
            created += 1

    if not dry:
        conn.commit()

    conn.close()

    elapsed = time.time() - t0
    print(f"\n完了: {created}個作成 ({elapsed:.1f}秒)")

    # 6. /reload-refs
    if not dry:
        try:
            r = urllib.request.urlopen(f"{VISION_SERVER}/reload-refs", timeout=3)
            import json
            res = json.loads(r.read())
            print(f"reload-refs: {res}")
        except Exception as e:
            print(f"reload-refs失敗（サーバー停止中?）: {e}")


if __name__ == "__main__":
    main()
