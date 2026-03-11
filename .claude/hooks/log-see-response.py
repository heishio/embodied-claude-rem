"""
PostToolUse hook for mcp__wifi-cam__see / mcp__usb-webcam__see
tool_responseの構造をログファイルに書き出すだけのダミー。
"""
import sys
import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "see-response-log.jsonl")

def main():
    try:
        raw = sys.stdin.read()
        data = json.loads(raw)
    except Exception as e:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"error": str(e), "raw": raw[:500]}) + "\n")
        return

    # tool_responseのキー構造だけ記録（Base64は長いので省略）
    def summarize(obj, depth=0):
        if isinstance(obj, dict):
            return {k: summarize(v, depth+1) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [summarize(i, depth+1) for i in obj[:3]]  # 先頭3件のみ
        elif isinstance(obj, str) and len(obj) > 200:
            return f"<str len={len(obj)} prefix={obj[:80]}>"
        else:
            return obj

    entry = {
        "timestamp": datetime.now().isoformat(),
        "summary": summarize(data),
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
