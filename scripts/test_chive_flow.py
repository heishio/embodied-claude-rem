"""chiVe vs E5 動詞フローベクトルの分布比較実験.

既存verb chainの動詞を取得し、chiVeとE5それぞれで
バイグラム平均フローベクトルを計算して、チェーン間の
コサイン類似度分布を比較する。
"""

import json
import os
import sqlite3
import sys

import numpy as np

# ── DB接続 ──
db_path = os.path.join(os.path.expanduser("~"), ".claude", "memories", "memory.db")
if not os.path.exists(db_path):
    print(f"ERROR: DB not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)

# 全チェーンの動詞リストを取得
rows = conn.execute("SELECT id, steps_json FROM verb_chains").fetchall()
print(f"Total chains: {len(rows)}")

chain_verbs: list[tuple[str, list[str]]] = []
all_unique_verbs: set[str] = set()

for row in rows:
    chain_id, steps_json = row[0], row[1]
    try:
        steps = json.loads(steps_json)
        verbs = [s.get("verb", "") for s in steps if s.get("verb")]
    except (json.JSONDecodeError, TypeError):
        verbs = []
    if verbs:
        chain_verbs.append((chain_id, verbs))
        all_unique_verbs.update(verbs)

conn.close()

print(f"Chains with verbs: {len(chain_verbs)}")
print(f"Unique verbs: {len(all_unique_verbs)}")
print(f"Sample verbs: {list(all_unique_verbs)[:20]}")


def compute_bigram_flow(verb_vecs_list: list[np.ndarray]) -> np.ndarray:
    """バイグラム平均フローベクトル（L2正規化済み）."""
    if len(verb_vecs_list) >= 2:
        bigrams = [
            (verb_vecs_list[i] + verb_vecs_list[i + 1]) / 2.0
            for i in range(len(verb_vecs_list) - 1)
        ]
        flow = np.mean(bigrams, axis=0)
    else:
        flow = verb_vecs_list[0].copy()
    norm = np.linalg.norm(flow)
    if norm > 0:
        flow = flow / norm
    return flow


def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    """N個のベクトルのコサイン類似度行列（上三角）."""
    # vecs: (N, D) normalized
    return vecs @ vecs.T


def analyze_distribution(name: str, flow_vecs: np.ndarray, sample_n: int = 500):
    """フローベクトルの類似度分布を分析."""
    n = len(flow_vecs)
    if n > sample_n:
        # ランダムサンプリング
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_n, replace=False)
        flow_vecs = flow_vecs[idx]
        n = sample_n

    sim_matrix = cosine_sim_matrix(flow_vecs)
    # 上三角のみ（対角除外）
    triu_idx = np.triu_indices(n, k=1)
    sims = sim_matrix[triu_idx]

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Vectors: {n}")
    print(f"  Dim: {flow_vecs.shape[1]}")
    print(f"  Similarity distribution:")
    print(f"    mean: {sims.mean():.4f}")
    print(f"    std:  {sims.std():.4f}")
    print(f"    min:  {sims.min():.4f}")
    print(f"    max:  {sims.max():.4f}")
    print(f"    5th:  {np.percentile(sims, 5):.4f}")
    print(f"    25th: {np.percentile(sims, 25):.4f}")
    print(f"    50th: {np.percentile(sims, 50):.4f}")
    print(f"    75th: {np.percentile(sims, 75):.4f}")
    print(f"    95th: {np.percentile(sims, 95):.4f}")


# ── chiVe ──
print("\n--- Loading chiVe mc90 ---")
chive_path = os.path.join(os.path.dirname(__file__), "chive", "chive-1.2-mc90_gensim-full", "chive-1.2-mc90.bin")
if not os.path.exists(chive_path):
    print(f"chiVe model not found at {chive_path}")
    chive_path = None

if chive_path:
    from gensim.models import Word2Vec

    model = Word2Vec.load(chive_path)
    kv = model.wv
    print(f"chiVe loaded: {len(kv)} words, {kv.vector_size}D")

    # 動詞のカバレッジ確認
    found = sum(1 for v in all_unique_verbs if v in kv)
    print(f"Verb coverage: {found}/{len(all_unique_verbs)} ({100*found/len(all_unique_verbs):.1f}%)")
    missing = [v for v in list(all_unique_verbs)[:50] if v not in kv]
    if missing:
        print(f"Missing verbs (sample): {missing[:10]}")

    # chiVeフローベクトル計算
    chive_flows: list[np.ndarray] = []
    skipped = 0
    for chain_id, verbs in chain_verbs:
        vecs = []
        for v in verbs:
            if v in kv:
                vecs.append(kv[v])
        if not vecs:
            skipped += 1
            continue
        flow = compute_bigram_flow(vecs)
        chive_flows.append(flow)

    print(f"chiVe flow vectors: {len(chive_flows)} (skipped {skipped})")
    if chive_flows:
        chive_flow_matrix = np.stack(chive_flows)
        analyze_distribution("chiVe mc90 - Bigram Average", chive_flow_matrix)


# ── E5 (現行) ──
print("\n--- Loading E5 ---")
try:
    from sentence_transformers import SentenceTransformer

    e5 = SentenceTransformer("intfloat/multilingual-e5-base")

    # 全ユニーク動詞をバッチエンベッド (passage prefix)
    unique_verb_list = sorted(all_unique_verbs)
    prefixed = [f"passage: {v}" for v in unique_verb_list]
    e5_verb_vecs = e5.encode(prefixed, normalize_embeddings=True, show_progress_bar=True)
    e5_verb_to_vec = {v: e5_verb_vecs[i] for i, v in enumerate(unique_verb_list)}

    # E5フローベクトル計算
    e5_flows: list[np.ndarray] = []
    for chain_id, verbs in chain_verbs:
        vecs = [e5_verb_to_vec[v] for v in verbs if v in e5_verb_to_vec]
        if not vecs:
            continue
        flow = compute_bigram_flow(vecs)
        e5_flows.append(flow)

    print(f"E5 flow vectors: {len(e5_flows)}")
    if e5_flows:
        e5_flow_matrix = np.stack(e5_flows)
        analyze_distribution("E5 multilingual-e5-base - Bigram Average", e5_flow_matrix)

except ImportError:
    print("sentence-transformers not available, skipping E5")


# ── 比較サマリ ──
print("\n" + "=" * 50)
print("  SUMMARY")
print("=" * 50)
print("std が大きいほど、フローベクトルの差別化が効いている")
print("目標: std > 0.05（最低限）、理想は std > 0.1")
