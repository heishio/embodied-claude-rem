"""テスト: 2段階パッチ検索
Stage 1: 全体embed → delta_centroid で「人いそう」判定
Stage 2: delta高類似度パッチ → face_centroid で「誰か」特定
"""
import sqlite3
import numpy as np
import cv2
import torch
import time
from pathlib import Path
from server import load_models, state, decode_vector, cos_sim, DB_PATH

load_models()

conn = sqlite3.connect(str(DB_PATH))
# delta_centroid（人物全体）
delta_rows = conn.execute(
    "SELECT id, delta_centroid, tag, member_count FROM image_composites WHERE delta_centroid IS NOT NULL AND tag IS NOT NULL"
).fetchall()
# face_centroid（顔）
face_rows = conn.execute(
    "SELECT id, face_centroid, tag, member_count FROM image_composites WHERE face_centroid IS NOT NULL AND tag IS NOT NULL"
).fetchall()
conn.close()

print("=== 参照composites ===")
print(f"delta: {len(delta_rows)} 件")
for r in delta_rows:
    print(f"  {r[0][:12]}... tag={r[2]} members={r[3]}")
print(f"face: {len(face_rows)} 件")
for r in face_rows:
    print(f"  {r[0][:12]}... tag={r[2]} members={r[3]}")

# 参照ベクトル（tag付きcompositeの平均）
delta_vecs = [decode_vector(r[1]) for r in delta_rows]
ref_delta = np.mean(delta_vecs, axis=0)
ref_delta = ref_delta / np.linalg.norm(ref_delta)

face_vecs = [decode_vector(r[1]) for r in face_rows]
ref_face = np.mean(face_vecs, axis=0)
ref_face = ref_face / np.linalg.norm(ref_face)

from PIL import Image as PILImage

# テスト画像
test_images = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_20260313_*.jpg"))[-6:]
# 人がいない画像も含める
all_captures = sorted(Path("/tmp/wifi-cam-mcp").glob("capture_*.jpg"))
# person_ratio低い画像を探す
conn2 = sqlite3.connect(str(DB_PATH))
empty_rows = conn2.execute(
    "SELECT capture_path FROM image_embeddings WHERE person_ratio < 0.05 ORDER BY timestamp DESC LIMIT 2"
).fetchall()
conn2.close()
for er in empty_rows:
    p = Path(er[0])
    if p.exists() and p not in test_images:
        test_images.insert(0, p)

print(f"\nテスト画像: {len(test_images)} 枚\n")

DELTA_THRESHOLD = 0.35  # Stage 1: 人いそう判定
FACE_SIM_THRESHOLD = 0.45  # Stage 2: 個人特定

print(f"{'file':<40} {'S1:delta':>9} {'judge':>6} {'highP':>5} {'S2:face':>9} {'who':>6} {'ms':>5}")
print("-" * 90)

for img_path in test_images:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        continue

    t0 = time.time()

    # パッチ特徴取得
    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    patches = out.last_hidden_state[0, 1:, :].cpu().numpy()  # [256, 768]

    # L2正規化
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    patches_n = patches / np.clip(norms, 1e-10, None)

    # Stage 1: 各パッチ vs delta_centroid
    delta_sims = patches_n @ ref_delta  # [256]
    delta_max = delta_sims.max()
    person_likely = delta_max >= DELTA_THRESHOLD

    # Stage 2: delta高類似度パッチ → face_centroid
    face_sim_val = 0.0
    n_high = 0
    who = "-"
    if person_likely:
        # delta上位パッチを抽出
        high_mask = delta_sims >= (delta_max * 0.7)  # max の70%以上
        n_high = high_mask.sum()
        high_patches = patches_n[high_mask]

        # 各高パッチ vs face_centroid、最高値を使う
        face_sims = high_patches @ ref_face
        face_sim_val = face_sims.max()

        if face_sim_val >= FACE_SIM_THRESHOLD:
            # 複数人いたら各composite比較するが、今は1人分
            who = delta_rows[0][2] or "?"

    elapsed = (time.time() - t0) * 1000

    fname = img_path.name
    judge = "PERSON" if person_likely else "empty"
    print(f"{fname:<40} {delta_max:>9.4f} {judge:>6} {n_high:>5} {face_sim_val:>9.4f} {who:>6} {elapsed:>5.0f}")
