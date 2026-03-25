"""ベンチマーク: 3クロップステッチの処理速度"""
import cv2
import numpy as np
import torch
import time
from pathlib import Path
from server import load_models, state

load_models()

from PIL import Image as PILImage


def get_patches(img_bgr: np.ndarray) -> np.ndarray:
    """画像→DINOv2パッチ特徴 [256, 768]"""
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    return out.last_hidden_state[0, 1:, :].cpu().numpy()  # [256, 768]


def three_crop(img_bgr: np.ndarray, overlap: float = 0.25) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """画像を左・中央・右に3分割（overlap付き）"""
    h, w = img_bgr.shape[:2]
    crop_w = w // 2  # 各クロップは幅の半分
    step = int(crop_w * (1 - overlap))

    x_starts = [0, step, w - crop_w]
    crops = []
    for x in x_starts:
        crops.append(img_bgr[:, x:x + crop_w])
    return crops[0], crops[1], crops[2]


def stitch_patches(left_p, center_p, right_p, overlap_cols: int = 4) -> np.ndarray:
    """3クロップのパッチ特徴をステッチ（重複部分は平均）
    各パッチは16x16、overlap_colsは重複するパッチ列数
    """
    # reshape to 16x16x768
    left = left_p.reshape(16, 16, -1)
    center = center_p.reshape(16, 16, -1)
    right = right_p.reshape(16, 16, -1)

    # 重複なしの列数
    unique_cols = 16 - overlap_cols

    # ステッチ: left_unique | (left_overlap + center_start)/2 | center_unique | (center_end + right_start)/2 | right_unique
    parts = [
        left[:, :unique_cols, :],                                          # left unique
        (left[:, unique_cols:, :] + center[:, :overlap_cols, :]) / 2,     # left-center overlap avg
        center[:, overlap_cols:unique_cols, :],                            # center unique
        (center[:, unique_cols:, :] + right[:, :overlap_cols, :]) / 2,    # center-right overlap avg
        right[:, overlap_cols:, :],                                        # right unique
    ]
    stitched = np.concatenate(parts, axis=1)  # [16, total_cols, 768]
    return stitched


# テスト画像
test_images = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_20260313_*.jpg"))[-3:]
print(f"テスト画像: {len(test_images)} 枚\n")

# ウォームアップ
img0 = cv2.imread(str(test_images[0]))
_ = get_patches(img0)

N_RUNS = 5

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue
    h, w = img_bgr.shape[:2]
    print(f"--- {img_path.name} ({w}x{h}) ---")

    # 1. 通常embed (1回)
    times_single = []
    for _ in range(N_RUNS):
        t0 = time.time()
        patches = get_patches(img_bgr)
        times_single.append((time.time() - t0) * 1000)
    avg_single = np.mean(times_single)
    print(f"  1クロップ: {avg_single:.1f}ms (patches: {patches.shape})")

    # 2. 3クロップステッチ
    times_three = []
    for _ in range(N_RUNS):
        t0 = time.time()
        left, center, right = three_crop(img_bgr)
        lp = get_patches(left)
        cp = get_patches(center)
        rp = get_patches(right)
        stitched = stitch_patches(lp, cp, rp, overlap_cols=4)
        times_three.append((time.time() - t0) * 1000)
    avg_three = np.mean(times_three)
    print(f"  3クロップ: {avg_three:.1f}ms (stitched: {stitched.shape})")
    print(f"  倍率: {avg_three / avg_single:.2f}x")

    # 3. バッチ推論（3クロップを1バッチで）
    times_batch = []
    for _ in range(N_RUNS):
        t0 = time.time()
        left, center, right = three_crop(img_bgr)
        tensors = []
        for crop in [left, center, right]:
            pil = PILImage.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            tensors.append(state.dino_transform(pil))
        batch = torch.stack(tensors)
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            out = state.dino_model(pixel_values=batch)
        all_patches = out.last_hidden_state[:, 1:, :].cpu().numpy()  # [3, 256, 768]
        stitched_b = stitch_patches(all_patches[0], all_patches[1], all_patches[2], overlap_cols=4)
        times_batch.append((time.time() - t0) * 1000)
    avg_batch = np.mean(times_batch)
    print(f"  バッチ3:  {avg_batch:.1f}ms (stitched: {stitched_b.shape})")
    print(f"  倍率: {avg_batch / avg_single:.2f}x")
    print()
