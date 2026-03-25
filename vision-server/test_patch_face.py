"""テスト: パッチ特徴で顔領域を特定 → その特徴で人物検索"""
import sqlite3
import numpy as np
import cv2
import torch
from pathlib import Path
from server import load_models, state, decode_vector, cos_sim, DB_PATH

load_models()

# face_centroid を持つ composite を取得
conn = sqlite3.connect(str(DB_PATH))
composites = conn.execute(
    "SELECT id, face_centroid, delta_centroid, tag, member_count FROM image_composites WHERE face_centroid IS NOT NULL"
).fetchall()
print(f"=== face_centroid付き composites: {len(composites)} 件 ===")
for c in composites:
    print(f"  {c[0][:12]}... tag={c[3]} members={c[4]}")

# face_centroid がなければ、image_embeddings の face_vector を使う
if not composites:
    print("\nface_centroid付きcompositeなし。image_embeddingsのface_vectorを使う。")
    face_rows = conn.execute(
        "SELECT id, face_vector, tag FROM image_embeddings WHERE face_vector IS NOT NULL AND tag IS NOT NULL LIMIT 5"
    ).fetchall()
    if not face_rows:
        face_rows = conn.execute(
            "SELECT id, face_vector, tag FROM image_embeddings WHERE face_vector IS NOT NULL LIMIT 5"
        ).fetchall()
    print(f"face_vector付きembeddings: {len(face_rows)} 件")
    # 平均して参照ベクトルにする
    face_vecs = [decode_vector(r[1]) for r in face_rows]
    ref_face_vec = np.mean(face_vecs, axis=0)
    ref_face_vec = ref_face_vec / np.linalg.norm(ref_face_vec)
    ref_label = "avg_face"
else:
    ref_face_vec = decode_vector(composites[0][1])
    ref_label = composites[0][3] or composites[0][0][:8]

# delta_centroid も取得（人物全体の参照）
delta_composites = conn.execute(
    "SELECT id, delta_centroid, tag FROM image_composites WHERE delta_centroid IS NOT NULL AND tag IS NOT NULL LIMIT 1"
).fetchall()
ref_delta_vec = None
if delta_composites:
    ref_delta_vec = decode_vector(delta_composites[0][1])
conn.close()

print(f"\n参照ベクトル: {ref_label} ({len(ref_face_vec)}d)")

# テスト画像
test_images = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_20260313_*.jpg"))[-5:]
print(f"テスト画像: {len(test_images)} 枚\n")

from PIL import Image as PILImage

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    # DINOv2 パッチ特徴を取得（CLSスキップ → 256パッチ x 768d）
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    patches = out.last_hidden_state[0, 1:, :].cpu().numpy()  # [256, 768]

    # 各パッチとface参照ベクトルのコサイン類似度
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches_normed = patches / np.clip(norms, 1e-10, None)
    sims = patches_normed @ ref_face_vec  # [256]

    # 16x16グリッドに reshape
    sim_map = sims.reshape(16, 16)

    # 統計
    top_k = 10
    top_indices = np.argsort(sims)[-top_k:]
    top_sims = sims[top_indices]
    top_positions = [(idx // 16, idx % 16) for idx in top_indices]

    print(f"--- {img_path.name} ---")
    print(f"  パッチ類似度: min={sims.min():.4f} max={sims.max():.4f} mean={sims.mean():.4f}")
    print(f"  Top-{top_k}パッチ: sim={top_sims.mean():.4f} (range {top_sims.min():.4f}-{top_sims.max():.4f})")
    print(f"  Top位置 (row,col): {top_positions[-3:]}")

    # 高類似度パッチ（>閾値）の平均ベクトルで人物検索
    threshold = 0.5
    high_mask = sims > threshold
    n_high = high_mask.sum()
    print(f"  sim>{threshold} パッチ数: {n_high}/256")

    if n_high > 0 and ref_delta_vec is not None:
        # 高類似度パッチの平均ベクトル
        high_patches = patches[high_mask]
        region_vec = high_patches.mean(axis=0)
        region_vec = region_vec / np.linalg.norm(region_vec)

        # composite delta_centroid との類似度
        delta_sim = cos_sim(region_vec, ref_delta_vec)

        # 全体平均との比較
        whole_vec = patches.mean(axis=0)
        whole_vec = whole_vec / np.linalg.norm(whole_vec)
        whole_delta_sim = cos_sim(whole_vec, ref_delta_vec)

        print(f"  顔領域パッチ平均 vs delta: {delta_sim:.4f}")
        print(f"  全体パッチ平均 vs delta:   {whole_delta_sim:.4f}")
    print()
