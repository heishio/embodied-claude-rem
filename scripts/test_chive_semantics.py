"""chiVe動詞ベクトルの意味的妥当性チェック.

似た動詞が近く、違う動詞が遠いか？
"""

import os
import sys

from gensim.models import Word2Vec

chive_path = os.path.join(os.path.dirname(__file__), "chive", "chive-1.2-mc90_gensim-full", "chive-1.2-mc90.bin")
model = Word2Vec.load(chive_path)
kv = model.wv

print("=== 1. 動詞ペアの類似度 ===\n")

pairs = [
    # 似てるはず（知覚）
    ("見る", "聞く", "知覚同士"),
    ("見る", "眺める", "視覚同士"),
    ("聞く", "聴く", "聴覚同士"),
    # 似てるはず（感情）
    ("驚く", "怒る", "感情同士"),
    ("喜ぶ", "悲しむ", "感情同士"),
    ("笑う", "泣く", "感情反対"),
    # 似てるはず（移動）
    ("歩く", "走る", "移動同士"),
    ("行く", "来る", "移動反対"),
    # 似てるはず（思考）
    ("考える", "思う", "思考同士"),
    ("悩む", "迷う", "思考同士"),
    # 遠いはず（カテゴリ違い）
    ("見る", "食べる", "知覚vs摂取"),
    ("走る", "考える", "移動vs思考"),
    ("笑う", "書く", "感情vs行為"),
    ("寝る", "話す", "生理vs発話"),
]

for v1, v2, label in pairs:
    if v1 in kv and v2 in kv:
        sim = kv.similarity(v1, v2)
        print(f"  {v1} ↔ {v2}  ({label}): {sim:.4f}")
    else:
        missing = []
        if v1 not in kv:
            missing.append(v1)
        if v2 not in kv:
            missing.append(v2)
        print(f"  {v1} ↔ {v2}  ({label}): MISSING {missing}")

print("\n=== 2. 各動詞の最近傍 ===\n")

test_verbs = ["見る", "驚く", "話す", "歩く", "考える", "食べる", "怒る", "書く", "忘れる", "思い出す"]

for verb in test_verbs:
    if verb in kv:
        neighbors = kv.most_similar(verb, topn=8)
        neighbor_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbors)
        print(f"  {verb}: {neighbor_str}")
    else:
        print(f"  {verb}: NOT IN VOCAB")

print("\n=== 3. うちのDBで実際に使われてる動詞の最近傍 ===\n")

import json
import sqlite3

db_path = os.path.join(os.path.expanduser("~"), ".claude", "memories", "memory.db")
conn = sqlite3.connect(db_path)
rows = conn.execute("SELECT all_verbs FROM verb_chains").fetchall()
conn.close()

from collections import Counter
verb_counter = Counter()
for row in rows:
    for v in row[0].split(","):
        v = v.strip()
        if v:
            verb_counter[v] += 1

print("頻出動詞 Top20:")
for verb, count in verb_counter.most_common(20):
    in_chive = "✓" if verb in kv else "✗"
    if verb in kv:
        neighbors = kv.most_similar(verb, topn=5)
        neighbor_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbors)
        print(f"  [{in_chive}] {verb} (x{count}): {neighbor_str}")
    else:
        print(f"  [{in_chive}] {verb} (x{count}): NOT IN VOCAB")
