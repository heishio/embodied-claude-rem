"""Vision Server - マルチモーダル視覚処理の常駐プロセス

DINOv2 ViT-B + MediaPipe をロードし、APIで提供する。
hookからcurlで叩いて使う。
"""

import asyncio
import logging
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("vision-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- 設定 ---
DB_PATH = Path.home() / ".claude" / "memories" / "memory.db"
MODEL_DIR = Path(__file__).parent.parent / "models"
SELFIE_MODEL = str(MODEL_DIR / "selfie_segmenter.tflite")
FACE_MODEL = str(MODEL_DIR / "blaze_face_short_range.tflite")
OD_MODEL = str(MODEL_DIR / "efficientdet_lite0.tflite")
CAPTURE_DIR = Path("/tmp/wifi-cam-mcp")
NEUTRAL_GRAY = np.uint8([128, 128, 128])
TARGET_L = 128
DINO_MODEL_NAME = "facebook/dinov2-with-registers-base"
BACKGROUND_INTERVAL = 10  # 秒
CAMERA_RTSP_URL = None  # 後で設定ファイルから読む
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_ref_lock = threading.Lock()


def validate_image_path(path: str) -> Path:
    """画像パスのバリデーション。不正なパスはHTTPException(400)を投げる。"""
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"ファイルが存在しない: {path}")
    if p.suffix.lower() not in ALLOWED_IMAGE_EXTS:
        raise HTTPException(status_code=400, detail=f"非対応の拡張子: {p.suffix}")
    return p


# --- グローバル状態 ---
class VisionState:
    def __init__(self):
        self.dino_model = None
        self.dino_transform = None
        self.segmenter = None
        self.face_detector = None
        self.object_detector = None
        self.camera_lock = threading.Lock()
        self.mcp_lock = threading.Event()  # set=MCP使用中
        self.mcp_lock.set()  # 初期状態はMCP未使用（set=通過可能）
        self.latest_result: dict | None = None
        self.bg_thread: threading.Thread | None = None
        self.running = False
        self.ref_centroids: dict[str, dict[str, np.ndarray]] = {}  # {tag: {"delta": vec, "face": vec}}
        self.pca_basis: dict | None = None  # {"mean": ndarray(768,), "components": ndarray(n_dims,768), "n_dims": int}

state = VisionState()


# --- モデルロード ---
def load_models():
    logger.info("モデルをロード中...")
    t0 = time.time()

    # DINOv2 ViT-B
    dino = AutoModel.from_pretrained(DINO_MODEL_NAME)
    dino.eval()
    if torch.cuda.is_available():
        dino = dino.cuda()
        logger.info("CUDA使用")
    state.dino_model = dino
    state.dino_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # MediaPipe selfie segmenter
    seg_options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=SELFIE_MODEL),
        output_confidence_masks=True,
    )
    state.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(seg_options)

    # MediaPipe face detector
    face_options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=FACE_MODEL),
    )
    state.face_detector = mp.tasks.vision.FaceDetector.create_from_options(face_options)

    # MediaPipe Object Detector (人物検出用)
    od_options = mp.tasks.vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=OD_MODEL),
        max_results=3,
        score_threshold=0.5,
        category_allowlist=["person"],
    )
    state.object_detector = mp.tasks.vision.ObjectDetector.create_from_options(od_options)

    logger.info(f"ロード完了: {time.time() - t0:.1f}秒")


# --- 画像処理 ---
def normalize_brightness(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """人物領域の平均輝度をTARGET_Lに揃える"""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    person_L = lab[:, :, 0][mask > 0]
    if len(person_L) == 0:
        return img_bgr
    shift = TARGET_L - person_L.mean()
    lab[:, :, 0] = np.clip(lab[:, :, 0] + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def segment_image(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """人物/背景をセグメンテーション。
    Returns: (foreground, background, person_ratio)
    """
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = state.segmenter.segment(mp_image)

    mask = result.confidence_masks[0].numpy_view()
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    binary = (mask > 0.5).astype(np.uint8)
    person_ratio = binary.sum() / binary.size
    mask_3d = np.dstack([binary] * 3)

    # ニュートラルグレー埋め + 輝度正規化
    fill = np.full_like(image_bgr, NEUTRAL_GRAY)
    img_norm = normalize_brightness(image_bgr, binary)
    foreground = np.where(mask_3d, img_norm, fill)
    background = np.where(mask_3d, fill, image_bgr)

    return foreground, background, float(person_ratio)


def detect_face(image_bgr: np.ndarray) -> tuple[np.ndarray | None, float | None]:
    """顔検出+クロップ。Returns: (face_crop_normalized, confidence)"""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = state.face_detector.detect(mp_image)

    if not result.detections:
        return None, None

    det = result.detections[0]
    bb = det.bounding_box
    h, w = image_bgr.shape[:2]
    margin = 0.2
    x1 = max(0, int(bb.origin_x - bb.width * margin))
    y1 = max(0, int(bb.origin_y - bb.height * margin))
    x2 = min(w, int(bb.origin_x + bb.width * (1 + margin)))
    y2 = min(h, int(bb.origin_y + bb.height * (1 + margin)))

    face_crop = image_bgr[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None, None

    # 輝度正規化
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + (TARGET_L - lab[:, :, 0].mean()), 0, 255)
    face_norm = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    return face_norm, float(det.categories[0].score)


def embed_image(img_bgr: np.ndarray) -> np.ndarray:
    """画像をDINOv2 ViT-Bでベクトル化（パッチ平均）。Returns: (768,) float32"""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    n_tokens = out.last_hidden_state.shape[1]
    if n_tokens == 261:  # with-registers: skip CLS + 4 registers
        patches = out.last_hidden_state[0, 5:, :]  # [256, 768]
    else:
        patches = out.last_hidden_state[0, 1:, :]  # skip CLS, [256, 768]
    vec = patches.mean(dim=0).cpu().numpy()
    vec = vec / np.linalg.norm(vec)  # L2正規化
    return vec.astype(np.float32)


def encode_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def get_patches(img_bgr: np.ndarray) -> np.ndarray:
    """画像をDINOv2でパッチベクトル化（平均せず生パッチ）。Returns: [256, 768] float32"""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.dino_transform(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        out = state.dino_model(pixel_values=tensor)
    n_tokens = out.last_hidden_state.shape[1]
    if n_tokens == 261:  # with-registers: skip CLS + 4 registers
        patches = out.last_hidden_state[0, 5:, :]  # [256, 768]
    else:
        patches = out.last_hidden_state[0, 1:, :]  # skip CLS, [256, 768]
    return patches.cpu().numpy()


def three_crop_batch_patches(img_bgr: np.ndarray, overlap: float = 0.25) -> np.ndarray:
    """3クロップ→バッチ推論→ステッチ。Returns: [16, 40, 768] float32"""
    from PIL import Image as PILImage

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

    n_tokens = out.last_hidden_state.shape[1]
    if n_tokens == 261:
        all_patches = out.last_hidden_state[:, 5:, :].cpu().numpy()  # [3, 256, 768]
    else:
        all_patches = out.last_hidden_state[:, 1:, :].cpu().numpy()

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


def _load_ref_centroids():
    """tag付きcompositeのdelta_centroid/face_centroidをキャッシュに読み込み"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """SELECT tag, delta_centroid, face_centroid FROM image_composites
           WHERE tag IS NOT NULL AND delta_centroid IS NOT NULL AND freshness >= 0.01
             AND id LIKE 'img-%'"""
    ).fetchall()
    conn.close()

    centroids: dict[str, dict[str, np.ndarray]] = {}
    for tag, dc_blob, fc_blob in rows:
        dc = decode_vector(dc_blob)
        dc = dc / np.linalg.norm(dc)
        entry = {"delta": dc}
        if fc_blob is not None:
            fc = decode_vector(fc_blob)
            fc = fc / np.linalg.norm(fc)
            entry["face"] = fc
        # 同じtagが複数compositeにあれば平均
        if tag in centroids:
            for key in entry:
                centroids[tag][key] = (centroids[tag].get(key, entry[key]) + entry[key]) / 2
                centroids[tag][key] = centroids[tag][key] / np.linalg.norm(centroids[tag][key])
        else:
            centroids[tag] = entry

    with _ref_lock:
        state.ref_centroids = centroids
    logger.info(f"参照ベクトル読み込み: {list(centroids.keys())} ({len(centroids)}人)")


def detect_person_fast(image_path: str) -> dict:
    """3クロップ+2段階パッチ検索による高速人物検出（DB保存なし）"""
    t0 = time.time()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {"error": f"画像読み込み失敗: {image_path}"}

    with _ref_lock:
        ref_centroids = state.ref_centroids

    if not ref_centroids:
        return {
            "person_ratio": 0.0,
            "match": None,
            "elapsed_ms": round((time.time() - t0) * 1000),
            "timing": {},
            "note": "no_ref_centroids",
        }

    # 3クロップバッチ推論
    t1 = time.time()
    patches_2d = three_crop_batch_patches(image_bgr)  # [16, 40, 768]
    t_crop = time.time()

    # パッチL2正規化
    rows, cols, dim = patches_2d.shape
    patches_flat = patches_2d.reshape(-1, dim)  # [640, 768]
    norms = np.linalg.norm(patches_flat, axis=1, keepdims=True)
    patches_n = patches_flat / np.clip(norms, 1e-10, None)

    # Stage1: 全パッチ vs 各compositeのdelta_centroid
    best_tag = None
    best_delta_max = 0.0
    best_high_patches = None

    for tag, refs in ref_centroids.items():
        delta_sims = patches_n @ refs["delta"]  # [640]
        d_max = float(delta_sims.max())
        if d_max > best_delta_max:
            best_delta_max = d_max
            best_tag = tag
            best_high_patches = patches_n[delta_sims >= (d_max * 0.7)]

    person_ratio = best_delta_max  # delta_maxをperson_ratioとして返す

    if best_delta_max < 0.35:
        t_done = time.time()
        return {
            "person_ratio": round(person_ratio, 3),
            "match": None,
            "elapsed_ms": round((t_done - t0) * 1000),
            "timing": {
                "crop_embed_ms": round((t_crop - t1) * 1000),
                "search_ms": round((t_done - t_crop) * 1000),
            },
        }

    # Stage2: delta高パッチ vs 各compositeのface_centroid
    best_face_tag = None
    best_face_sim = 0.0

    for tag, refs in ref_centroids.items():
        if "face" not in refs:
            continue
        # delta高パッチを再計算（このtagのdelta基準）
        delta_sims = patches_n @ refs["delta"]
        d_max = float(delta_sims.max())
        high_mask = delta_sims >= (d_max * 0.7)
        if high_mask.sum() == 0:
            continue
        face_sims = patches_n[high_mask] @ refs["face"]
        f_max = float(face_sims.max())
        if f_max > best_face_sim:
            best_face_sim = f_max
            best_face_tag = tag

    t_done = time.time()
    match = None
    if best_face_sim >= 0.45 and best_face_tag is not None:
        match = {
            "tag": best_face_tag,
            "delta_sim": round(best_face_sim, 4),
            "type": "composite",
        }

    return {
        "person_ratio": round(person_ratio, 3),
        "match": match,
        "elapsed_ms": round((t_done - t0) * 1000),
        "timing": {
            "crop_embed_ms": round((t_crop - t1) * 1000),
            "search_ms": round((t_done - t_crop) * 1000),
        },
    }


# --- DB操作 ---
def ensure_table():
    """image_embeddingsテーブルがなければ作成"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_embeddings (
            id TEXT PRIMARY KEY,
            capture_path TEXT,
            timestamp TEXT,
            flow_vector BLOB,
            delta_vector BLOB,
            face_vector BLOB,
            camera_position TEXT,
            person_ratio REAL,
            face_confidence REAL,
            freshness REAL DEFAULT 1.0,
            memory_id TEXT,
            tag TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_image_embeddings_timestamp
        ON image_embeddings(timestamp)
    """)
    # 既存テーブルにtagカラムがなければ追加
    cols = {row[1] for row in conn.execute("PRAGMA table_info(image_embeddings)")}
    if "tag" not in cols:
        conn.execute("ALTER TABLE image_embeddings ADD COLUMN tag TEXT")
        logger.info("image_embeddings: tagカラムを追加")
    # image_composites テーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_composites (
            id TEXT PRIMARY KEY,
            delta_centroid BLOB NOT NULL,
            flow_centroid BLOB,
            face_centroid BLOB,
            member_count INTEGER DEFAULT 0,
            freshness REAL DEFAULT 1.0,
            tag TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS composite_members (
            composite_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            added_at TEXT,
            PRIMARY KEY (composite_id, member_id)
        )
    """)
    # PCA基底テーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pca_bases (
            id TEXT PRIMARY KEY,
            mean_vector BLOB NOT NULL,
            components BLOB NOT NULL,
            eigenvalues BLOB NOT NULL,
            n_dims INTEGER NOT NULL,
            n_samples INTEGER NOT NULL,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_embedding(
    capture_path: str,
    flow_vec: np.ndarray,
    delta_vec: np.ndarray,
    face_vec: np.ndarray | None,
    person_ratio: float,
    face_confidence: float | None,
) -> str:
    """ベクトルをDBに保存。Returns: id"""
    emb_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """INSERT INTO image_embeddings
           (id, capture_path, timestamp, flow_vector, delta_vector, face_vector,
            person_ratio, face_confidence, freshness)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1.0)""",
        (
            emb_id,
            capture_path,
            timestamp,
            encode_vector(flow_vec),
            encode_vector(delta_vec),
            encode_vector(face_vec) if face_vec is not None else None,
            person_ratio,
            face_confidence,
        ),
    )
    conn.commit()
    conn.close()
    return emb_id


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


PCA_N_DIMS = 20
PCA_MIN_SAMPLES = 20


def _compute_and_save_pca_basis(n_dims: int = PCA_N_DIMS):
    """タグ付きperson delta_vectorsからPCA基底を計算・保存。"""
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        """SELECT delta_vector FROM image_embeddings
           WHERE tag IS NOT NULL AND delta_vector IS NOT NULL
             AND person_ratio >= 0.1"""
    ).fetchall()

    if len(rows) < PCA_MIN_SAMPLES:
        logger.info(f"PCA: サンプル不足 ({len(rows)} < {PCA_MIN_SAMPLES}), スキップ")
        conn.close()
        return

    vecs = np.stack([np.frombuffer(r[0], dtype=np.float32) for r in rows])
    mean = vecs.mean(axis=0)
    centered = vecs - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 降順ソート
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # 上位n_dims成分
    components = eigvecs[:, :n_dims].T  # (n_dims, 768)
    top_eigvals = eigvals[:n_dims]
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")
    conn.execute(
        """INSERT OR REPLACE INTO pca_bases
           (id, mean_vector, components, eigenvalues, n_dims, n_samples, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            "person_delta",
            encode_vector(mean.astype(np.float32)),
            encode_vector(components.astype(np.float32)),
            encode_vector(top_eigvals.astype(np.float32)),
            n_dims,
            len(rows),
            timestamp,
        ),
    )
    conn.commit()
    conn.close()

    with _ref_lock:
        state.pca_basis = {
            "mean": mean.astype(np.float32),
            "components": components.astype(np.float32),
            "n_dims": n_dims,
        }
    logger.info(
        f"PCA基底更新: n_samples={len(rows)}, n_dims={n_dims}, "
        f"cumvar@{n_dims}={cumvar[n_dims - 1]:.1%}"
    )


def _load_pca_basis():
    """DBからPCA基底を読み込み。"""
    conn = sqlite3.connect(str(DB_PATH))
    # テーブル存在確認
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    if "pca_bases" not in tables:
        conn.close()
        return

    row = conn.execute(
        "SELECT mean_vector, components, eigenvalues, n_dims, n_samples, updated_at FROM pca_bases WHERE id = 'person_delta'"
    ).fetchone()
    conn.close()

    if row is None:
        logger.info("PCA基底: なし")
        return

    mean = np.frombuffer(row[0], dtype=np.float32)
    n_dims = row[3]
    components = np.frombuffer(row[1], dtype=np.float32).reshape(n_dims, -1)

    with _ref_lock:
        state.pca_basis = {
            "mean": mean,
            "components": components,
            "n_dims": n_dims,
        }
    logger.info(f"PCA基底ロード: n_dims={n_dims}, n_samples={row[4]}, updated={row[5]}")


def pca_project(vec: np.ndarray, basis: dict) -> np.ndarray:
    """ベクトルをPCA空間に射影+L2正規化。"""
    centered = vec - basis["mean"]
    projected = basis["components"] @ centered  # (n_dims,)
    norm = np.linalg.norm(projected)
    if norm < 1e-10:
        return projected
    return projected / norm


def pca_cos_sim(a: np.ndarray, b: np.ndarray, basis: dict) -> float:
    """PCA空間でのコサイン類似度。"""
    pa = pca_project(a, basis)
    pb = pca_project(b, basis)
    return float(np.dot(pa, pb))


def search_similar(query_flow: np.ndarray, query_delta: np.ndarray, n: int = 3) -> list[dict]:
    """DB内のimage_embeddings + image_compositesからコサイン類似度で検索。

    compositeは重心類似度とメンバー最近傍類似度の高い方を使う。
    """
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT id, capture_path, timestamp, flow_vector, delta_vector, person_ratio, freshness, memory_id FROM image_embeddings ORDER BY timestamp DESC LIMIT 500"
    ).fetchall()

    results = []
    for row in rows:
        rid, path, ts, fv_blob, dv_blob, pr, fresh, mid = row
        fv = decode_vector(fv_blob)
        dv = decode_vector(dv_blob)
        flow_sim = cos_sim(query_flow, fv)
        delta_sim = cos_sim(query_delta, dv)
        combined = 0.5 * flow_sim + 0.5 * delta_sim
        results.append({
            "id": rid,
            "capture_path": path,
            "timestamp": ts,
            "flow_sim": round(flow_sim, 4),
            "delta_sim": round(delta_sim, 4),
            "combined_sim": round(combined, 4),
            "person_ratio": pr,
            "freshness": fresh,
            "memory_id": mid,
            "type": "embedding",
        })

    # image_composites: 重心 + メンバー最近傍
    composite_rows = conn.execute(
        "SELECT id, delta_centroid, flow_centroid, member_count, freshness, tag FROM image_composites WHERE freshness >= 0.01"
    ).fetchall()

    # composite_membersのベクトルを一括取得
    member_rows = conn.execute(
        """SELECT cm.composite_id, ie.flow_vector, ie.delta_vector
           FROM composite_members cm
           JOIN image_embeddings ie ON ie.id = cm.member_id
           WHERE cm.composite_id LIKE 'img-%' OR cm.composite_id LIKE 'flow-%'"""
    ).fetchall()
    conn.close()

    # composite_id → メンバーベクトルリスト
    member_vecs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for mrow in member_rows:
        cid = mrow[0]
        fv = decode_vector(mrow[1]) if mrow[1] else None
        dv = decode_vector(mrow[2]) if mrow[2] else None
        if fv is not None and dv is not None:
            member_vecs.setdefault(cid, []).append((fv, dv))

    for row in composite_rows:
        cid, dc_blob, fc_blob, member_count, fresh, tag = row
        if dc_blob is None:
            continue

        # 重心との類似度
        dc = decode_vector(dc_blob)
        centroid_delta_sim = cos_sim(query_delta, dc)
        centroid_flow_sim = 0.0
        if fc_blob is not None:
            fc = decode_vector(fc_blob)
            centroid_flow_sim = cos_sim(query_flow, fc)
        centroid_combined = 0.5 * centroid_flow_sim + 0.5 * centroid_delta_sim

        # メンバー最近傍との類似度
        nearest_combined = 0.0
        nearest_flow_sim = 0.0
        nearest_delta_sim = 0.0
        members = member_vecs.get(cid, [])
        for m_fv, m_dv in members:
            mf = cos_sim(query_flow, m_fv)
            md = cos_sim(query_delta, m_dv)
            mc = 0.5 * mf + 0.5 * md
            if mc > nearest_combined:
                nearest_combined = mc
                nearest_flow_sim = mf
                nearest_delta_sim = md

        # 重心と最近傍の高い方を採用
        if nearest_combined > centroid_combined:
            best_flow = nearest_flow_sim
            best_delta = nearest_delta_sim
            best_combined = nearest_combined
            match_type = "nearest"
        else:
            best_flow = centroid_flow_sim
            best_delta = centroid_delta_sim
            best_combined = centroid_combined
            match_type = "centroid"

        results.append({
            "id": cid,
            "composite_id": cid,
            "tag": tag,
            "flow_sim": round(best_flow, 4),
            "delta_sim": round(best_delta, 4),
            "combined_sim": round(best_combined, 4),
            "centroid_sim": round(centroid_combined, 4),
            "nearest_sim": round(nearest_combined, 4),
            "match_type": match_type,
            "member_count": member_count,
            "freshness": fresh,
            "type": "composite",
        })

    results.sort(key=lambda x: x["combined_sim"], reverse=True)
    return results[:n]


# --- フルパイプライン ---
def process_image(image_path: str) -> dict:
    """画像をセグメント→ベクトル化→DB保存→類似検索"""
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {"error": f"画像読み込み失敗: {image_path}"}

    # セグメンテーション
    foreground, background, person_ratio = segment_image(image_bgr)

    # エンベディング
    delta_vec = embed_image(foreground)  # 人物
    flow_vec = embed_image(background)   # 背景

    # 顔検出
    face_crop, face_conf = detect_face(image_bgr)
    face_vec = embed_image(face_crop) if face_crop is not None else None

    emb_id = save_embedding(image_path, flow_vec, delta_vec, face_vec, person_ratio, face_conf)
    similar = search_similar(flow_vec, delta_vec, n=3)

    result = {
        "id": emb_id,
        "capture_path": image_path,
        "person_ratio": round(person_ratio, 3),
        "face_detected": face_conf is not None,
        "face_confidence": round(face_conf, 3) if face_conf else None,
        "similar": similar,
        "vector_dim": len(delta_vec),
    }

    return result


# --- バックグラウンドキャプチャ ---
def background_capture_loop():
    """10秒ごとにカメラからキャプチャしてベクトル化"""
    logger.info("バックグラウンドキャプチャ開始")
    while state.running:
        # MCP使用中なら待つ
        if not state.mcp_lock.wait(timeout=1):
            continue

        try:
            # 最新のキャプチャファイルを使う
            captures = sorted(CAPTURE_DIR.glob("capture_*.jpg"))
            if captures:
                latest = str(captures[-1])
                result = process_image(latest)
                state.latest_result = result
                logger.info(
                    f"BG capture: person={result.get('person_ratio', 0):.0%}, "
                    f"face={result.get('face_detected', False)}, "
                    f"similar={len(result.get('similar', []))}"
                )
        except Exception as e:
            logger.error(f"BG capture error: {e}")

        # 10秒待つ（1秒ごとにrunningチェック）
        for _ in range(BACKGROUND_INTERVAL):
            if not state.running:
                break
            time.sleep(1)

    logger.info("バックグラウンドキャプチャ終了")


# --- FastAPI ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    ensure_table()
    _load_ref_centroids()
    _load_pca_basis()
    state.running = True
    # バックグラウンドキャプチャはPhase3で有効化（現在は能動知覚のみ）
    # state.bg_thread = threading.Thread(target=background_capture_loop, daemon=True)
    # state.bg_thread.start()
    yield
    state.running = False
    if state.bg_thread:
        state.bg_thread.join(timeout=5)
    if state.segmenter:
        state.segmenter.close()
    if state.face_detector:
        state.face_detector.close()
    if state.object_detector:
        state.object_detector.close()
    logger.info("シャットダウン完了")


app = FastAPI(title="Vision Server", lifespan=lifespan)


class EmbedRequest(BaseModel):
    path: str


class DetectRequest(BaseModel):
    path: str


class TagRequest(BaseModel):
    tag: str
    image_id: str | None = None  # Noneなら最新レコードを対象


@app.post("/embed")
async def embed_endpoint(req: EmbedRequest):
    """画像パスを受け取りベクトル化+DB保存"""
    validate_image_path(req.path)
    result = await asyncio.to_thread(process_image, req.path)
    state.latest_result = result
    return result


def _search_delta_match(delta_vec: np.ndarray, conn: sqlite3.Connection) -> dict | None:
    """tag付きのimage_embeddings/image_compositesからdelta類似検索。
    PCA基底がある場合はPCA simで再ランキング。"""

    with _ref_lock:
        basis = state.pca_basis

    candidates = []

    rows = conn.execute(
        """SELECT id, tag, delta_vector FROM image_embeddings
           WHERE tag IS NOT NULL AND delta_vector IS NOT NULL
             AND person_ratio >= 0.1
           ORDER BY timestamp DESC LIMIT 200"""
    ).fetchall()
    for row in rows:
        dv = decode_vector(row[2])
        sim = cos_sim(delta_vec, dv)
        if sim >= 0.60:
            candidates.append({"tag": row[1], "raw_sim": sim, "type": "embedding", "id": row[0], "dv": dv})

    comp_rows = conn.execute(
        """SELECT id, tag, delta_centroid FROM image_composites
           WHERE tag IS NOT NULL AND delta_centroid IS NOT NULL
             AND freshness >= 0.01"""
    ).fetchall()
    for row in comp_rows:
        dc = decode_vector(row[2])
        sim = cos_sim(delta_vec, dc)
        if sim >= 0.60:
            candidates.append({"tag": row[1], "raw_sim": sim, "type": "composite", "id": row[0], "dv": dc})

    if not candidates:
        return None

    if basis is not None:
        # PCA simで再ランキング
        for c in candidates:
            c["pca_sim"] = pca_cos_sim(delta_vec, c["dv"], basis)
        candidates.sort(key=lambda x: x["pca_sim"], reverse=True)
        best = candidates[0]
        if best["pca_sim"] >= 0.0 and best["raw_sim"] >= 0.60:
            return {
                "tag": best["tag"],
                "delta_sim": round(best["raw_sim"], 4),
                "pca_sim": round(best["pca_sim"], 4),
                "type": best["type"],
            }
    else:
        # PCA基底なし: 従来通り raw sim >= 0.75
        candidates.sort(key=lambda x: x["raw_sim"], reverse=True)
        best = candidates[0]
        if best["raw_sim"] >= 0.75:
            return {
                "tag": best["tag"],
                "delta_sim": round(best["raw_sim"], 4),
                "type": best["type"],
            }

    return None


def _best_delta_sim(delta_vec: np.ndarray, conn: sqlite3.Connection) -> float:
    """デバッグ用: 閾値以下でも最高delta類似度を返す"""
    best = 0.0
    rows = conn.execute(
        """SELECT delta_vector FROM image_embeddings
           WHERE tag IS NOT NULL AND delta_vector IS NOT NULL AND person_ratio >= 0.1
           ORDER BY timestamp DESC LIMIT 200"""
    ).fetchall()
    for row in rows:
        sim = cos_sim(delta_vec, decode_vector(row[0]))
        if sim > best:
            best = sim
    comp_rows = conn.execute(
        """SELECT delta_centroid FROM image_composites
           WHERE tag IS NOT NULL AND delta_centroid IS NOT NULL AND freshness >= 0.01"""
    ).fetchall()
    for row in comp_rows:
        sim = cos_sim(delta_vec, decode_vector(row[0]))
        if sim > best:
            best = sim
    return round(best, 4)


def _crop_and_segment_person(image_bgr: np.ndarray, bb) -> np.ndarray:
    """バウンディングボックスでクロップ→segment_imageで人物抽出→グレー埋め。"""
    h, w = image_bgr.shape[:2]
    x1 = max(0, bb.origin_x)
    y1 = max(0, bb.origin_y)
    x2 = min(w, bb.origin_x + bb.width)
    y2 = min(h, bb.origin_y + bb.height)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full_like(image_bgr, NEUTRAL_GRAY)

    # クロップ領域にsegment_imageをかけて背景除去
    foreground, _bg, _ratio = segment_image(crop)

    # 元画像サイズでグレー埋め + クロップ領域に貼り付け
    filled = np.full_like(image_bgr, NEUTRAL_GRAY)
    filled[y1:y2, x1:x2] = foreground
    return filled


def detect_person_legacy(image_path: str) -> dict:
    """【旧】人物検出+delta類似検索のみ（DB保存しない）。フォールバック用。
    1人: segment_imageで従来通り処理（後方互換）
    2-3人: Object Detectorでクロップ→各人物をembed+match
    """
    t0 = time.time()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {"error": f"画像読み込み失敗: {image_path}"}

    # Object Detectorで人物検出
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    od_result = state.object_detector.detect(mp_image)
    person_detections = od_result.detections
    t_od = time.time()

    if len(person_detections) <= 1:
        # 0-1人: 従来のsegment_image処理（精度維持）
        foreground, _background, person_ratio = segment_image(image_bgr)
        t_seg = time.time()

        if person_ratio < 0.1:
            return {
                "person_ratio": round(person_ratio, 3),
                "match": None,
                "elapsed_ms": round((time.time() - t0) * 1000),
            }

        delta_vec = embed_image(foreground)
        t_embed = time.time()

        conn = sqlite3.connect(str(DB_PATH))
        match = _search_delta_match(delta_vec, conn)
        conn.close()
        t_search = time.time()

        return {
            "person_ratio": round(person_ratio, 3),
            "match": match,
            "elapsed_ms": round((t_search - t0) * 1000),
            "timing": {
                "od_ms": round((t_od - t0) * 1000),
                "segment_ms": round((t_seg - t_od) * 1000),
                "embed_ms": round((t_embed - t_seg) * 1000),
                "search_ms": round((t_search - t_embed) * 1000),
            },
        }

    # 2-3人: 各人物をクロップ→embed→match
    conn = sqlite3.connect(str(DB_PATH))
    persons = []
    for i, det in enumerate(person_detections):
        bb = det.bounding_box
        filled = _crop_and_segment_person(image_bgr, bb)
        delta_vec = embed_image(filled)
        match = _search_delta_match(delta_vec, conn)
        persons.append({
            "id": f"P{i + 1}",
            "match": match,
            "score": round(det.categories[0].score, 3),
        })
    conn.close()
    t_done = time.time()

    return {
        "person_count": len(persons),
        "persons": persons,
        "elapsed_ms": round((t_done - t0) * 1000),
        "timing": {
            "od_ms": round((t_od - t0) * 1000),
            "embed_match_ms": round((t_done - t_od) * 1000),
        },
    }


@app.post("/detect")
async def detect_endpoint(req: DetectRequest):
    """人物検出+2段階パッチ検索（DB保存なし、半受動視覚用）"""
    validate_image_path(req.path)
    result = await asyncio.to_thread(detect_person_fast, req.path)
    return result


@app.post("/detect-legacy")
async def detect_legacy_endpoint(req: DetectRequest):
    """旧パイプライン（フォールバック用）"""
    validate_image_path(req.path)
    result = await asyncio.to_thread(detect_person_legacy, req.path)
    return result


@app.get("/reload-refs")
async def reload_refs_endpoint():
    """参照ベクトルとPCA基底を再読み込み（consolidate後に呼ぶ）"""
    await asyncio.to_thread(_load_ref_centroids)
    await asyncio.to_thread(_load_pca_basis)
    return {
        "reloaded": True,
        "tags": list(state.ref_centroids.keys()),
        "count": len(state.ref_centroids),
        "pca_loaded": state.pca_basis is not None,
    }


SIMILARITY_TAG_THRESHOLD = 0.80  # delta類似度がこれ以上なら自動tag伝播


@app.post("/tag")
async def tag_endpoint(req: TagRequest):
    """image_embeddingsにタグを書き込み、delta類似度が高いtagなしembeddingにも自動伝播する。"""
    def do_tag():
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row

        # 1. image_embeddingsにtag書き込み
        if req.image_id:
            target_id = req.image_id
            conn.execute(
                "UPDATE image_embeddings SET tag = ? WHERE id = ?",
                (req.tag, target_id),
            )
        else:
            row = conn.execute(
                "SELECT id FROM image_embeddings ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row is None:
                conn.close()
                return 0, 0, 0
            target_id = row["id"]
            conn.execute(
                "UPDATE image_embeddings SET tag = ? WHERE id = ?",
                (req.tag, target_id),
            )

        conn.commit()
        embedding_affected = conn.execute("SELECT changes()").fetchone()[0]

        # 2. delta類似度でtagなしembeddingに自動伝播
        similar_tagged = 0
        target_row = conn.execute(
            "SELECT delta_vector FROM image_embeddings WHERE id = ?",
            (target_id,),
        ).fetchone()

        if target_row and target_row["delta_vector"]:
            target_delta = decode_vector(bytes(target_row["delta_vector"]))
            target_norm = np.linalg.norm(target_delta)

            if target_norm > 1e-10:
                # tagなし + person_ratio >= 0.1 のembeddingを検索
                candidates = conn.execute(
                    """SELECT id, delta_vector FROM image_embeddings
                       WHERE tag IS NULL AND delta_vector IS NOT NULL
                         AND person_ratio >= 0.1 AND id != ?""",
                    (target_id,),
                ).fetchall()

                for cand in candidates:
                    cand_delta = decode_vector(bytes(cand["delta_vector"]))
                    cand_norm = np.linalg.norm(cand_delta)
                    if cand_norm < 1e-10:
                        continue
                    sim = float(np.dot(target_delta, cand_delta) / (target_norm * cand_norm))
                    if sim >= SIMILARITY_TAG_THRESHOLD:
                        conn.execute(
                            "UPDATE image_embeddings SET tag = ? WHERE id = ?",
                            (req.tag, cand["id"]),
                        )
                        similar_tagged += 1

                if similar_tagged > 0:
                    conn.commit()

        # 3. composite_membersからtag付きembeddingが属するimage_compositeにもtag伝播
        composite_affected = 0
        tagged_ids = conn.execute(
            "SELECT id FROM image_embeddings WHERE tag = ?",
            (req.tag,),
        ).fetchall()

        for trow in tagged_ids:
            composite_rows = conn.execute(
                """SELECT composite_id FROM composite_members
                   WHERE member_id = ? AND (composite_id LIKE 'img-%' OR composite_id LIKE 'flow-%')""",
                (trow["id"],),
            ).fetchall()
            for crow in composite_rows:
                cid = crow["composite_id"]
                conn.execute(
                    "UPDATE image_composites SET tag = ? WHERE id = ? AND tag IS NULL",
                    (req.tag, cid),
                )
                composite_affected += conn.execute("SELECT changes()").fetchone()[0]

        conn.commit()
        conn.close()
        return embedding_affected, similar_tagged, composite_affected

    embedding_affected, similar_tagged, composite_affected = await asyncio.to_thread(do_tag)
    # PCA基底を再計算（タグ付きデータが変わった）
    await asyncio.to_thread(_compute_and_save_pca_basis)
    return {
        "tagged": embedding_affected > 0,
        "tag": req.tag,
        "image_id": req.image_id,
        "similar_tagged": similar_tagged,
        "composites_tagged": composite_affected,
        "pca_updated": state.pca_basis is not None,
    }


@app.get("/latest")
async def latest_endpoint():
    """直近のベクトル検索結果を返す"""
    if state.latest_result is None:
        return {"status": "no_data"}
    return state.latest_result


@app.get("/status")
async def status_endpoint():
    """サーバー状態"""
    return {
        "running": state.running,
        "model": DINO_MODEL_NAME,
        "vector_dim": 768,
        "cuda": torch.cuda.is_available(),
        "has_latest": state.latest_result is not None,
        "bg_alive": state.bg_thread.is_alive() if state.bg_thread else False,
    }


@app.get("/pca-status")
async def pca_status_endpoint():
    """PCA基底の状態を返す（デバッグ用）"""
    def do_fetch():
        conn = sqlite3.connect(str(DB_PATH))
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if "pca_bases" not in tables:
            conn.close()
            return None
        row = conn.execute(
            "SELECT n_dims, n_samples, updated_at, eigenvalues FROM pca_bases WHERE id = 'person_delta'"
        ).fetchone()
        conn.close()
        if row is None:
            return None
        eigvals = np.frombuffer(row[3], dtype=np.float32)
        return {
            "n_dims": row[0],
            "n_samples": row[1],
            "updated_at": row[2],
            "top_eigenvalues": [round(float(v), 6) for v in eigvals[:5]],
        }

    result = await asyncio.to_thread(do_fetch)
    if result is None:
        return {"status": "no_pca_basis"}
    return {"status": "active", **result, "in_memory": state.pca_basis is not None}


@app.get("/composites")
async def composites_endpoint():
    """画像compositeの一覧を返す（デバッグ用）"""
    def do_fetch():
        conn = sqlite3.connect(str(DB_PATH))
        rows = conn.execute(
            """SELECT id, member_count, freshness, tag, created_at, updated_at
               FROM image_composites ORDER BY updated_at DESC"""
        ).fetchall()
        conn.close()
        return [
            {
                "id": r[0],
                "member_count": r[1],
                "freshness": r[2],
                "tag": r[3],
                "created_at": r[4],
                "updated_at": r[5],
            }
            for r in rows
        ]

    results = await asyncio.to_thread(do_fetch)
    return {"composites": results, "count": len(results)}


@app.post("/lock")
async def lock_endpoint():
    """MCP see時にカメラロックを取得（バックグラウンド一時停止）"""
    state.mcp_lock.clear()  # バックグラウンドを停止
    return {"locked": True}


@app.post("/unlock")
async def unlock_endpoint():
    """MCP see完了後にカメラロックを解放"""
    state.mcp_lock.set()  # バックグラウンドを再開
    return {"locked": False}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")
