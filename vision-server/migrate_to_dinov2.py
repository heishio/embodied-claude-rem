"""DB移行スクリプト: 512d (MobileCLIP2-S0) → 768d (DINOv2 ViT-B)

既存のimage_embeddingsを再エンベッドし、image_compositesをリセットする。
vision-serverが起動していなくても実行可能（直接DINOv2をロードする）。

Usage:
    python migrate_to_dinov2.py              # 通常実行
    python migrate_to_dinov2.py --dry-run    # 変更しない確認モード
    python migrate_to_dinov2.py --yes        # 画像なしレコード削除を自動承認
"""

import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

# server.pyと同じディレクトリにいることを前提にインポート
sys.path.insert(0, str(Path(__file__).parent))
from server import (
    load_models,
    segment_image,
    detect_face,
    embed_image,
    encode_vector,
    state,
)

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="DB移行: 512d → 768d (DINOv2 ViT-B)")
    parser.add_argument(
        "--yes", action="store_true", help="画像なしレコードを確認なしで削除"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="実際には変更しない（確認モード）"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(Path.home() / ".claude" / "memories" / "memory.db"),
        help="DBパス（デフォルト: ~/.claude/memories/memory.db）",
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://127.0.0.1:8100",
        help="vision-serverのURL（現在は未使用、直接モデルロード）",
    )
    return parser.parse_args()


def get_all_embeddings(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, capture_path, flow_vector, delta_vector, face_vector, person_ratio
           FROM image_embeddings"""
    ).fetchall()
    return [dict(r) for r in rows]


def re_embed_row(row: dict) -> dict | None:
    """1レコードを再エンベッド。画像がなければNoneを返す。"""
    capture_path = row["capture_path"]
    if not capture_path or not os.path.exists(capture_path):
        return None

    image_bgr = cv2.imread(capture_path)
    if image_bgr is None:
        return None

    # セグメンテーション
    foreground, background, person_ratio = segment_image(image_bgr)

    # エンベディング
    delta_vec = embed_image(foreground)
    flow_vec = embed_image(background)

    # 顔検出
    face_crop, face_conf = detect_face(image_bgr)
    face_vec = embed_image(face_crop) if face_crop is not None else None

    return {
        "flow_vector": encode_vector(flow_vec),
        "delta_vector": encode_vector(delta_vec),
        "face_vector": encode_vector(face_vec) if face_vec is not None else None,
        "person_ratio": person_ratio,
    }


def main():
    args = parse_args()
    db_path = args.db
    dry_run = args.dry_run

    if dry_run:
        print("=== DRY RUN モード（変更なし） ===\n")

    # DB接続
    if not os.path.exists(db_path):
        print(f"ERROR: DBが見つかりません: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(db_path)

    # 既存レコード取得
    rows = get_all_embeddings(conn)
    total = len(rows)
    print(f"image_embeddings: {total} レコード")

    if total == 0:
        print("移行対象なし。")
        conn.close()
        return

    # 既存ベクトルの次元数チェック
    sample = rows[0]
    if sample["delta_vector"]:
        import numpy as np

        dim = len(np.frombuffer(sample["delta_vector"], dtype=np.float32))
        print(f"現在のベクトル次元: {dim}d")
        if dim == 768 and not args.yes:
            print("既に768dです。移行不要かもしれません。")
            resp = input("続行しますか？ (y/N): ").strip().lower()
            if resp != "y":
                conn.close()
                return
        elif dim == 768:
            print("既に768dですが、--yes指定のため続行します。")

    # モデルロード
    print("\nDINOv2モデルをロード中...")
    t0 = time.time()
    load_models()
    print(f"ロード完了: {time.time() - t0:.1f}秒\n")

    # 再エンベッド
    re_embedded = 0
    missing = []

    for i, row in enumerate(rows):
        rid = row["id"]
        path = row["capture_path"]
        print(f"  [{i + 1}/{total}] {path or '(パスなし)'}", end=" ")

        result = re_embed_row(row)
        if result is None:
            print("→ 画像なし（削除候補）")
            missing.append(row)
            continue

        if not dry_run:
            conn.execute(
                """UPDATE image_embeddings
                   SET flow_vector = ?, delta_vector = ?, face_vector = ?, person_ratio = ?
                   WHERE id = ?""",
                (
                    result["flow_vector"],
                    result["delta_vector"],
                    result["face_vector"],
                    result["person_ratio"],
                    rid,
                ),
            )
        re_embedded += 1
        dim = len(result["delta_vector"]) // 4  # float32 = 4 bytes
        print(f"→ 再エンベッド完了 ({dim}d)")

    if not dry_run:
        conn.commit()

    # 画像なしレコードの処理
    print(f"\n--- 再エンベッド: {re_embedded}/{total} 件 ---")
    print(f"--- 画像なし: {len(missing)} 件 ---")

    if missing:
        print("\n画像なしレコード:")
        for m in missing:
            print(f"  {m['id'][:8]}... | {m['capture_path'] or '(なし)'}")

        if not dry_run:
            if args.yes:
                do_delete = True
            else:
                resp = input("\nこれらのレコードを削除しますか？ (y/N): ").strip().lower()
                do_delete = resp == "y"

            if do_delete:
                for m in missing:
                    conn.execute(
                        "DELETE FROM image_embeddings WHERE id = ?", (m["id"],)
                    )
                conn.commit()
                print(f"{len(missing)} 件を削除しました。")
            else:
                print("削除をスキップしました。")

    # image_composites + composite_members をリセット
    print("\nimage_composites / composite_members (img-*/flow-*) をリセット中...")

    if not dry_run:
        deleted_members = conn.execute(
            "DELETE FROM composite_members WHERE composite_id LIKE 'img-%' OR composite_id LIKE 'flow-%'"
        ).rowcount
        deleted_composites = conn.execute("DELETE FROM image_composites").rowcount
        conn.commit()
        print(f"  composite_members: {deleted_members} 件削除")
        print(f"  image_composites: {deleted_composites} 件削除")
    else:
        count_members = conn.execute(
            "SELECT COUNT(*) FROM composite_members WHERE composite_id LIKE 'img-%' OR composite_id LIKE 'flow-%'"
        ).fetchone()[0]
        count_composites = conn.execute(
            "SELECT COUNT(*) FROM image_composites"
        ).fetchone()[0]
        print(f"  composite_members: {count_members} 件（削除予定）")
        print(f"  image_composites: {count_composites} 件（削除予定）")

    conn.close()

    print("\n=== 完了 ===")
    print(f"再エンベッド: {re_embedded} 件")
    print(f"画像なし: {len(missing)} 件")
    if dry_run:
        print("\n※ DRY RUN のため、実際の変更は行われていません。")
    else:
        print("\n次回の consolidate_memories で image_composites が768dベクトルから再構築されます。")


if __name__ == "__main__":
    main()
