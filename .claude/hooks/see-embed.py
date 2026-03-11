"""
PostToolUse hook for mcp__wifi-cam__see / mcp__usb-webcam__see
tool_responseのtextからタイムスタンプを取り出し、
vision-server /embed にパスを送る。
vision-serverが落ちていたら黙って終了（seeは影響を受けない）。
"""
import sys
import json
import re
import urllib.request
import urllib.error

VISION_SERVER = "http://127.0.0.1:8100"
CAPTURE_DIR = "C:/tmp/wifi-cam-mcp"


def main():
    try:
        data = json.loads(sys.stdin.read())
    except Exception:
        return

    tool_response = data.get("tool_response", [])
    if not isinstance(tool_response, list):
        return

    # textブロックからタイムスタンプを抽出
    timestamp = None
    for block in tool_response:
        if isinstance(block, dict) and block.get("type") == "text":
            m = re.search(r"(\d{8}_\d{6})", block.get("text", ""))
            if m:
                timestamp = m.group(1)
                break

    if not timestamp:
        return

    capture_path = f"{CAPTURE_DIR}/capture_{timestamp}.jpg"

    # vision-server /embed に送る
    try:
        payload = json.dumps({"path": capture_path}).encode("utf-8")
        req = urllib.request.Request(
            f"{VISION_SERVER}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=30)
    except (urllib.error.URLError, OSError):
        # vision-serverが落ちていても無視
        pass


if __name__ == "__main__":
    main()
