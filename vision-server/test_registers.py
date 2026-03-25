"""比較: dinov2-base vs dinov2-with-registers-base
パッチ特徴の違い、2段階検索の精度差を見る
"""
import sqlite3
import numpy as np
import cv2
import torch
import time
from pathlib import Path
from server import load_models, state, decode_vector, cos_sim, DB_PATH
from transformers import AutoModel
from torchvision import transforms
from PIL import Image as PILImage

# --- 現行モデル (without registers) ---
load_models()
model_noreg = state.dino_model
transform = state.dino_transform

# --- レジスタ付きモデル ---
print("レジスタ付きモデルをロード中...")
model_reg = AutoModel.from_pretrained("facebook/dinov2-with-registers-base")
model_reg.eval()
if torch.cuda.is_available():
    model_reg = model_reg.cuda()
print("ロード完了\n")

# composites（現行モデルで作られたもの）
conn = sqlite3.connect(str(DB_PATH))
delta_rows = conn.execute(
    "SELECT id, delta_centroid, tag FROM image_composites WHERE delta_centroid IS NOT NULL AND tag IS NOT NULL"
).fetchall()
face_rows = conn.execute(
    "SELECT id, face_centroid, tag FROM image_composites WHERE face_centroid IS NOT NULL AND tag IS NOT NULL"
).fetchall()
conn.close()

delta_vecs = [decode_vector(r[1]) for r in delta_rows]
ref_delta = np.mean(delta_vecs, axis=0)
ref_delta = ref_delta / np.linalg.norm(ref_delta)

face_vecs = [decode_vector(r[1]) for r in face_rows]
ref_face = np.mean(face_vecs, axis=0)
ref_face = ref_face / np.linalg.norm(ref_face)


def get_patches(img_bgr, model):
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = model(pixel_values=tensor)
    # レジスタ付きはCLS + 4レジスタ + 256パッチ = 261トークン
    # レジスタなしはCLS + 256パッチ = 257トークン
    n_tokens = out.last_hidden_state.shape[1]
    n_patches = n_tokens - 1  # CLS除外
    if n_tokens == 261:  # registers model: skip CLS + 4 registers
        patches = out.last_hidden_state[0, 5:, :]
    else:
        patches = out.last_hidden_state[0, 1:, :]
    return patches.cpu().numpy()


def analyze_patches(patches, label):
    """パッチ特徴の統計とノルム分布"""
    norms = np.linalg.norm(patches, axis=1)
    patches_n = patches / np.clip(norms.reshape(-1, 1), 1e-10, None)

    # delta検索
    delta_sims = patches_n @ ref_delta
    # face検索
    high_mask = delta_sims >= (delta_sims.max() * 0.7)
    face_sims = patches_n[high_mask] @ ref_face

    return {
        "label": label,
        "n_patches": len(patches),
        "norm_mean": norms.mean(),
        "norm_std": norms.std(),
        "norm_max": norms.max(),
        "norm_min": norms.min(),
        "delta_max": delta_sims.max(),
        "delta_mean": delta_sims.mean(),
        "face_max": face_sims.max() if len(face_sims) > 0 else 0,
        "high_patches": high_mask.sum(),
    }


# テスト
test_images = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_20260313_*.jpg"))[-5:]
print(f"テスト画像: {len(test_images)} 枚\n")

# ウォームアップ
img0 = cv2.imread(str(test_images[0]))
_ = get_patches(img0, model_noreg)
_ = get_patches(img0, model_reg)

print(f"{'file':<35} {'model':<8} {'nPatch':>6} {'normM':>6} {'normSD':>6} {'normMax':>7} {'S1:dMax':>8} {'S1:dMean':>8} {'highP':>5} {'S2:fMax':>8} {'ms':>5}")
print("-" * 120)

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    fname = img_path.name[:30]

    for model, name in [(model_noreg, "v1"), (model_reg, "v2-reg")]:
        t0 = time.time()
        patches = get_patches(img_bgr, model)
        ms = (time.time() - t0) * 1000
        r = analyze_patches(patches, name)
        print(f"{fname:<35} {name:<8} {r['n_patches']:>6} {r['norm_mean']:>6.2f} {r['norm_std']:>6.3f} {r['norm_max']:>7.2f} {r['delta_max']:>8.4f} {r['delta_mean']:>8.4f} {r['high_patches']:>5} {r['face_max']:>8.4f} {ms:>5.0f}")
    print()
