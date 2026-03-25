"""テスト: 3クロップステッチ + 2段階パッチ検索（離れた画像で検証）"""
import sqlite3
import numpy as np
import cv2
import torch
import time
from pathlib import Path
from server import load_models, state, decode_vector, cos_sim, DB_PATH

load_models()

from PIL import Image as PILImage

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

ref_tag = delta_rows[0][2]


def get_patches(img_bgr: np.ndarray) -> np.ndarray:
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    return out.last_hidden_state[0, 1:, :].cpu().numpy()


def three_crop_batch(img_bgr: np.ndarray, overlap: float = 0.25) -> np.ndarray:
    """3クロップ→バッチ推論→ステッチ"""
    h, w = img_bgr.shape[:2]
    crop_w = w // 2
    step = int(crop_w * (1 - overlap))
    x_starts = [0, step, w - crop_w]

    tensors = []
    for x in x_starts:
        crop = img_bgr[:, x:x + crop_w]
        pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(state.dino_transform(pil))

    batch = torch.stack(tensors)
    if torch.cuda.is_available():
        batch = batch.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=batch)
    all_patches = out.last_hidden_state[:, 1:, :].cpu().numpy()  # [3, 256, 768]

    # ステッチ（overlap_cols=4）
    overlap_cols = 4
    unique_cols = 16 - overlap_cols
    l = all_patches[0].reshape(16, 16, -1)
    c = all_patches[1].reshape(16, 16, -1)
    r = all_patches[2].reshape(16, 16, -1)
    parts = [
        l[:, :unique_cols, :],
        (l[:, unique_cols:, :] + c[:, :overlap_cols, :]) / 2,
        c[:, overlap_cols:unique_cols, :],
        (c[:, unique_cols:, :] + r[:, :overlap_cols, :]) / 2,
        r[:, overlap_cols:, :],
    ]
    stitched = np.concatenate(parts, axis=1)  # [16, 40, 768]
    return stitched


def twostage(patches_2d: np.ndarray, label: str):
    """2段階パッチ検索"""
    rows, cols, dim = patches_2d.shape
    patches_flat = patches_2d.reshape(-1, dim)
    norms = np.linalg.norm(patches_flat, axis=1, keepdims=True)
    patches_n = patches_flat / np.clip(norms, 1e-10, None)

    # S1: delta
    delta_sims = patches_n @ ref_delta
    delta_max = delta_sims.max()

    # S2: face（delta上位パッチから）
    high_mask = delta_sims >= (delta_max * 0.7)
    n_high = high_mask.sum()
    face_sim_val = 0.0
    who = "-"
    if n_high > 0:
        face_sims = patches_n[high_mask] @ ref_face
        face_sim_val = face_sims.max()
        if face_sim_val >= 0.45:
            who = ref_tag

    return delta_max, n_high, face_sim_val, who


# テスト画像
test_images = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_20260313_*.jpg"))[-6:]
print(f"テスト画像: {len(test_images)} 枚\n")

# ウォームアップ
_ = get_patches(cv2.imread(str(test_images[0])))

print(f"{'file':<40} {'mode':<8} {'S1:delta':>9} {'highP':>5} {'S2:face':>9} {'who':>6} {'ms':>6}")
print("-" * 90)

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    fname = img_path.name

    # 1クロップ
    t0 = time.time()
    p1 = get_patches(img_bgr).reshape(16, 16, -1)
    d1, h1, f1, w1 = twostage(p1, "1crop")
    ms1 = (time.time() - t0) * 1000
    print(f"{fname:<40} {'1crop':<8} {d1:>9.4f} {h1:>5} {f1:>9.4f} {w1:>6} {ms1:>6.1f}")

    # 3クロップバッチ
    t0 = time.time()
    p3 = three_crop_batch(img_bgr)
    d3, h3, f3, w3 = twostage(p3, "3crop")
    ms3 = (time.time() - t0) * 1000
    print(f"{'':<40} {'3crop':<8} {d3:>9.4f} {h3:>5} {f3:>9.4f} {w3:>6} {ms3:>6.1f}")
    print()
