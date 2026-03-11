"""Vision Server - マルチモーダル視覚処理の常駐プロセス

MobileCLIP2-S0 + MediaPipe をロードし、APIで提供する。
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
import open_clip
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger("vision-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --- 設定 ---
DB_PATH = Path.home() / ".claude" / "memories" / "memory.db"
MODEL_DIR = Path(__file__).parent.parent / "models"
SELFIE_MODEL = str(MODEL_DIR / "selfie_segmenter.tflite")
FACE_MODEL = str(MODEL_DIR / "blaze_face_short_range.tflite")
CAPTURE_DIR = Path("/tmp/wifi-cam-mcp")
NEUTRAL_GRAY = np.uint8([128, 128, 128])
TARGET_L = 128
BACKGROUND_INTERVAL = 10  # 秒
CAMERA_RTSP_URL = None  # 後で設定ファイルから読む


# --- グローバル状態 ---
class VisionState:
    def __init__(self):
        self.clip_model = None
        self.clip_preprocess = None
        self.segmenter = None
        self.face_detector = None
        self.camera_lock = threading.Lock()
        self.mcp_lock = threading.Event()  # set=MCP使用中
        self.mcp_lock.set()  # 初期状態はMCP未使用（set=通過可能）
        self.latest_result: dict | None = None
        self.bg_thread: threading.Thread | None = None
        self.running = False

state = VisionState()


# --- モデルロード ---
def load_models():
    logger.info("モデルをロード中...")
    t0 = time.time()

    # MobileCLIP2-S0
    model, _, preprocess = open_clip.create_model_and_transforms(
        "MobileCLIP2-S0", pretrained="dfndr2b"
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("CUDA使用")
    state.clip_model = model
    state.clip_preprocess = preprocess

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
    """画像をMobileCLIP2-S0でベクトル化。Returns: (512,) float32"""
    from PIL import Image as PILImage

    pil = PILImage.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    tensor = state.clip_preprocess(pil).unsqueeze(0)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    with torch.no_grad():
        vec = state.clip_model.encode_image(tensor).cpu().numpy()[0]
    return vec.astype(np.float32)


def encode_vector(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def decode_vector(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


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

    # DB保存
    emb_id = save_embedding(image_path, flow_vec, delta_vec, face_vec, person_ratio, face_conf)

    # 類似検索
    similar = search_similar(flow_vec, delta_vec, n=3)

    result = {
        "id": emb_id,
        "capture_path": image_path,
        "person_ratio": round(person_ratio, 3),
        "face_detected": face_conf is not None,
        "face_confidence": round(face_conf, 3) if face_conf else None,
        "similar": similar,
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
    result = await asyncio.to_thread(process_image, req.path)
    state.latest_result = result
    return result


def detect_person(image_path: str) -> dict:
    """人物検出+delta類似検索のみ（DB保存しない）"""
    t0 = time.time()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return {"error": f"画像読み込み失敗: {image_path}"}

    # セグメンテーション
    foreground, _background, person_ratio = segment_image(image_bgr)
    t_seg = time.time()

    if person_ratio < 0.1:
        return {
            "person_ratio": round(person_ratio, 3),
            "match": None,
            "elapsed_ms": round((time.time() - t0) * 1000),
        }

    # deltaベクトルのみ計算
    delta_vec = embed_image(foreground)
    t_embed = time.time()

    # tag付きのimage_embeddingsとimage_compositesからdelta類似検索
    conn = sqlite3.connect(str(DB_PATH))
    best_match = None
    best_sim = 0.0

    # image_embeddingsからtag付きを検索
    rows = conn.execute(
        """SELECT id, tag, delta_vector FROM image_embeddings
           WHERE tag IS NOT NULL AND delta_vector IS NOT NULL
             AND person_ratio >= 0.1
           ORDER BY timestamp DESC LIMIT 200"""
    ).fetchall()
    for row in rows:
        dv = decode_vector(row[2])
        sim = cos_sim(delta_vec, dv)
        if sim > best_sim:
            best_sim = sim
            best_match = {"tag": row[1], "sim": sim, "type": "embedding", "id": row[0]}

    # image_compositesからtag付きを検索
    comp_rows = conn.execute(
        """SELECT id, tag, delta_centroid FROM image_composites
           WHERE tag IS NOT NULL AND delta_centroid IS NOT NULL
             AND freshness >= 0.01"""
    ).fetchall()
    for row in comp_rows:
        dc = decode_vector(row[2])
        sim = cos_sim(delta_vec, dc)
        if sim > best_sim:
            best_sim = sim
            best_match = {"tag": row[1], "sim": sim, "type": "composite", "id": row[0]}

    conn.close()
    t_search = time.time()

    result = {
        "person_ratio": round(person_ratio, 3),
        "match": None,
        "elapsed_ms": round((t_search - t0) * 1000),
        "timing": {
            "segment_ms": round((t_seg - t0) * 1000),
            "embed_ms": round((t_embed - t_seg) * 1000),
            "search_ms": round((t_search - t_embed) * 1000),
        },
    }

    if best_match and best_match["sim"] >= 0.70:
        result["match"] = {
            "tag": best_match["tag"],
            "delta_sim": round(best_match["sim"], 4),
            "type": best_match["type"],
        }

    return result


@app.post("/detect")
async def detect_endpoint(req: DetectRequest):
    """人物検出+delta類似検索（DB保存なし、半受動視覚用）"""
    result = await asyncio.to_thread(detect_person, req.path)
    return result


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
    return {
        "tagged": embedding_affected > 0,
        "tag": req.tag,
        "image_id": req.image_id,
        "similar_tagged": similar_tagged,
        "composites_tagged": composite_affected,
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
        "cuda": torch.cuda.is_available(),
        "has_latest": state.latest_result is not None,
        "bg_alive": state.bg_thread.is_alive() if state.bg_thread else False,
    }


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
