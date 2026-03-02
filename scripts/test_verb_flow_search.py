#!/usr/bin/env python
"""動詞フロー検索の実験スクリプト。

ユーザー発話から動詞を全部抜き出し、その流れでverb_chain_embeddingsを検索する。
"""
import sqlite3
import sys
import numpy as np

# ── 設定 ──
DB_PATH = "C:/Users/sep61/.claude/memories/memory.db"

# ── 入力テキスト ──
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
else:
    text = input("検索テキスト: ")

print(f"\n入力: {text}")

# ── sudachipyで動詞抽出 ──
from sudachipy import Dictionary
tokenizer = Dictionary().create()
tokens = tokenizer.tokenize(text)

VERB_STOPLIST = {"する", "ある", "いる", "なる", "できる", "れる", "られる", "せる", "させる"}

verbs = []
for t in tokens:
    pos = t.part_of_speech()
    if pos[0] == "動詞":
        lemma = t.dictionary_form()
        if lemma not in VERB_STOPLIST:
            verbs.append(lemma)

if not verbs:
    print("動詞が見つかりません")
    sys.exit(0)

flow_text = " → ".join(verbs)
print(f"動詞フロー: {flow_text}")

# ── embeddingモデルロード ──
print("\nモデルロード中...")
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("intfloat/multilingual-e5-base")

# クエリをエンコード
query_vec = model.encode(f"query: {flow_text}", normalize_embeddings=True)
print(f"クエリベクトル: {query_vec.shape}")

# ── DB検索 ──
conn = sqlite3.connect(DB_PATH)

# flow_vectorがあるチェーンを取得
rows = conn.execute("""
    SELECT vc.id, vc.all_verbs, vc.all_nouns, vc.context,
           vce.flow_vector, vce.vector
    FROM verb_chains vc
    JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id
    WHERE vce.flow_vector IS NOT NULL AND length(vce.flow_vector) > 0
""").fetchall()

print(f"\nflow_vectorありチェーン: {len(rows)}件")

# flow_vectorなしだけどvectorありのチェーンもチェック
rows_vec_only = conn.execute("""
    SELECT COUNT(*) FROM verb_chains vc
    JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id
    WHERE (vce.flow_vector IS NULL OR length(vce.flow_vector) = 0)
      AND vce.vector IS NOT NULL AND length(vce.vector) > 0
""").fetchone()
print(f"vectorのみチェーン: {rows_vec_only[0]}件")

# ── コサイン類似度で検索 ──
results = []
for row in rows:
    chain_id, all_verbs, all_nouns, context, flow_vec_blob, vec_blob = row

    # flow_vectorで検索
    flow_vec = np.frombuffer(bytes(flow_vec_blob), dtype=np.float32)
    if flow_vec.shape[0] != query_vec.shape[0]:
        continue
    sim = float(np.dot(query_vec, flow_vec))

    results.append({
        "id": chain_id[:12],
        "verbs": all_verbs,
        "nouns": all_nouns,
        "context": (context or "")[:60],
        "flow_sim": sim,
    })

# vectorでも検索（比較用）
results_vec = []
rows2 = conn.execute("""
    SELECT vc.id, vc.all_verbs, vc.all_nouns, vc.context, vce.vector
    FROM verb_chains vc
    JOIN verb_chain_embeddings vce ON vc.id = vce.chain_id
    WHERE vce.vector IS NOT NULL AND length(vce.vector) > 0
""").fetchall()

for row in rows2:
    chain_id, all_verbs, all_nouns, context, vec_blob = row
    vec = np.frombuffer(bytes(vec_blob), dtype=np.float32)
    if vec.shape[0] != query_vec.shape[0]:
        continue
    sim = float(np.dot(query_vec, vec))
    results_vec.append({
        "id": chain_id[:12],
        "verbs": all_verbs,
        "nouns": all_nouns,
        "context": (context or "")[:60],
        "vec_sim": sim,
    })

conn.close()

# ── 結果表示 ──
print("\n" + "=" * 70)
print("【flow_vector検索】（動詞フローの類似度）")
print("=" * 70)
results.sort(key=lambda x: x["flow_sim"], reverse=True)
for i, r in enumerate(results[:10]):
    print(f"\n{i+1}. sim={r['flow_sim']:.4f}")
    print(f"   verbs: {r['verbs']}")
    print(f"   nouns: {r['nouns']}")
    if r["context"]:
        print(f"   ctx:   {r['context']}")

print("\n" + "=" * 70)
print("【vector検索】（チェーン全体の類似度、比較用）")
print("=" * 70)
results_vec.sort(key=lambda x: x["vec_sim"], reverse=True)
for i, r in enumerate(results_vec[:10]):
    print(f"\n{i+1}. sim={r['vec_sim']:.4f}")
    print(f"   verbs: {r['verbs']}")
    print(f"   nouns: {r['nouns']}")
    if r["context"]:
        print(f"   ctx:   {r['context']}")
