"""テスト: 画像全体embed vs composite delta_centroid"""
import sqlite3
import numpy as np
from pathlib import Path
from server import load_models, embed_image, decode_vector, cos_sim, DB_PATH
import cv2

load_models()

conn = sqlite3.connect(str(DB_PATH))
composites = conn.execute(
    "SELECT id, delta_centroid, tag, member_count FROM image_composites WHERE delta_centroid IS NOT NULL"
).fetchall()
print(f"=== image_composites: {len(composites)} 件 ===")
for c in composites:
    print(f"  {c[0][:12]}... tag={c[2]} members={c[3]}")

emb_rows = conn.execute(
    "SELECT capture_path, person_ratio, tag FROM image_embeddings ORDER BY timestamp DESC LIMIT 15"
).fetchall()
conn.close()

print()
print("=== 全体embed vs composite delta_centroid ===")
header = f"{'file':<45} {'p_ratio':>7} {'tag':>6}"
for c in composites:
    tag = c[2] or c[0][:8]
    header += f"  {tag:>8}"
print(header)
print("-" * len(header))

for path, pr, tag in emb_rows:
    img = cv2.imread(path)
    if img is None:
        continue
    whole_vec = embed_image(img)
    fname = Path(path).name
    line = f"{fname:<45} {pr:>7.3f} {(tag or '-'):>6}"
    for c in composites:
        dc = decode_vector(c[1])
        sim = cos_sim(whole_vec, dc)
        line += f"  {sim:>8.4f}"
    print(line)
