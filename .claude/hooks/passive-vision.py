"""
UserPromptSubmit hook: 半受動視覚
go2rtcからスナップショットを取得し、vision-server /detect で人物検出。
結果を [passive-vision] タグでstdoutに出力。
"""
import json
import os
import sys
import tempfile
import time

# Windows環境でのUTF-8出力
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
import urllib.request
import urllib.error

GO2RTC_SNAPSHOT = "http://localhost:1984/api/frame.jpeg?src=tapo_cam"
VISION_SERVER = "http://127.0.0.1:8100"
CAPTURE_DIR = os.path.join(tempfile.gettempdir(), "passive-vision")


def main():
    t0 = time.time()

    # vision-serverが生きてるか確認（高速フェイル）
    try:
        urllib.request.urlopen(f"{VISION_SERVER}/status", timeout=1)
    except (urllib.error.URLError, OSError):
        return

    # go2rtcからスナップショット取得
    try:
        resp = urllib.request.urlopen(GO2RTC_SNAPSHOT, timeout=3)
        jpeg_data = resp.read()
    except (urllib.error.URLError, OSError):
        return

    if len(jpeg_data) < 1000:
        return

    # 一時ファイルに保存（上書き）
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    snap_path = os.path.join(CAPTURE_DIR, "snapshot.jpg")
    with open(snap_path, "wb") as f:
        f.write(jpeg_data)

    # Pythonからvision-serverに渡すパスはスラッシュ形式に
    snap_path_posix = snap_path.replace("\\", "/")

    # vision-server /detect に送る
    try:
        payload = json.dumps({"path": snap_path_posix}).encode("utf-8")
        req = urllib.request.Request(
            f"{VISION_SERVER}/detect",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req, timeout=4)
        result = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return

    total_ms = round((time.time() - t0) * 1000)

    # 結果をstdoutに出力
    person_ratio = result.get("person_ratio", 0)
    match = result.get("match")

    if match:
        tag = match["tag"]
        sim = match["delta_sim"]
        print(f"[passive-vision] person_ratio={person_ratio} match={tag}({sim}) elapsed={total_ms}ms")
    elif person_ratio >= 0.1:
        print(f"[passive-vision] person_ratio={person_ratio} match=unknown elapsed={total_ms}ms")
    else:
        print(f"[passive-vision] person_ratio={person_ratio} elapsed={total_ms}ms")


if __name__ == "__main__":
    main()
