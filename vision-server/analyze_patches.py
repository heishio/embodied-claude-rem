"""パッチ特徴の類似度分布を分析するスクリプト"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import cv2
import time
from server import load_models, state, three_crop_batch_patches, decode_vector, DB_PATH
import sqlite3

load_models()

# ref centroids読み込み
conn = sqlite3.connect(str(DB_PATH))
rows = conn.execute(
    """SELECT tag, delta_centroid, face_centroid FROM image_composites
       WHERE tag IS NOT NULL AND delta_centroid IS NOT NULL AND id LIKE 'img-%'"""
).fetchall()
conn.close()

ref_delta = None
ref_face = None
for tag, dc, fc in rows:
    if tag == "シオ":
        d = decode_vector(dc)
        ref_delta = d / np.linalg.norm(d)
        if fc:
            f = decode_vector(fc)
            ref_face = f / np.linalg.norm(f)
        break

print(f"ref_delta: {ref_delta is not None}, ref_face: {ref_face is not None}\n")

# 分析対象
images = {
    "無人(210919)": "C:/tmp/wifi-cam-mcp/capture_20260313_210919.jpg",
    "シオ近距離(214949)": "C:/tmp/wifi-cam-mcp/capture_20260313_214949.jpg",
    "フリー写真+シオ(152804)": "C:/tmp/wifi-cam-mcp/capture_20260313_152804.jpg",
    "シオ遠距離(202921)": "C:/tmp/wifi-cam-mcp/capture_20260313_202921.jpg",
    "別人テスト(214717)": "C:/tmp/wifi-cam-mcp/capture_20260313_214717.jpg",
}

for label, path in images.items():
    img = cv2.imread(path)
    if img is None:
        print(f"{label}: 読み込み失敗")
        continue

    patches_2d = three_crop_batch_patches(img)  # [16, 40, 768]
    rows, cols, dim = patches_2d.shape
    patches_flat = patches_2d.reshape(-1, dim)
    norms = np.linalg.norm(patches_flat, axis=1, keepdims=True)
    patches_n = patches_flat / np.clip(norms, 1e-10, None)

    # delta類似度
    delta_sims = patches_n @ ref_delta
    # face類似度
    face_sims = patches_n @ ref_face if ref_face is not None else np.zeros(len(patches_n))

    # delta高パッチ(top 5%)のface類似度
    delta_threshold = np.percentile(delta_sims, 95)
    high_delta_mask = delta_sims >= delta_threshold
    face_of_high_delta = face_sims[high_delta_mask]

    print(f"=== {label} ===")
    print(f"  delta_sim: max={delta_sims.max():.4f}  mean={delta_sims.mean():.4f}  std={delta_sims.std():.4f}  p95={np.percentile(delta_sims, 95):.4f}")
    print(f"  face_sim:  max={face_sims.max():.4f}  mean={face_sims.mean():.4f}  std={face_sims.std():.4f}  p95={np.percentile(face_sims, 95):.4f}")
    print(f"  delta top5% ({high_delta_mask.sum()}patches) → face: max={face_of_high_delta.max():.4f}  mean={face_of_high_delta.mean():.4f}")

    # delta_simのヒストグラム（テキスト）
    bins = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    counts, _ = np.histogram(delta_sims, bins=bins)
    hist_str = " | ".join(f"{bins[i]:.2f}-{bins[i+1]:.2f}:{counts[i]:>3}" for i in range(len(counts)))
    print(f"  delta分布: {hist_str}")

    # face_simのヒストグラム
    bins_f = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    counts_f, _ = np.histogram(face_sims, bins=bins_f)
    hist_str_f = " | ".join(f"{bins_f[i]:.2f}-{bins_f[i+1]:.2f}:{counts_f[i]:>3}" for i in range(len(counts_f)))
    print(f"  face分布:  {hist_str_f}")
    print()
