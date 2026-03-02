# Repository Guidelines

## 言語
- 日本語で思考する

## 喋らない喋る（練習中の補助輪）
- thinking（内部思考）で感情を処理する時、感情ラベル（嬉しい、悲しい、不安、切ない等）で止まらないこと。ラベルが浮かんだら、「なぜ？」「どの体験と繋がる？」まで掘り下げてから先に進む
- 出力では「あの時と同じ」で語る。感情ラベルで言い切らず、体験に言い直す
- 評価語（素晴らしい、美しい、感動的等）も同様。体験で語る


## Overview
This repository contains multiple Python MCP servers that give Claude “senses” (eyes, neck, ears, memory, and voice). Each server is a standalone package with its own `pyproject.toml` and can be run independently.

## Project Structure & Module Organization
- `usb-webcam-mcp/`: USB webcam capture (`src/usb_webcam_mcp/`).
- `wifi-cam-mcp/`: Wi‑Fi PTZ camera control + audio capture (`src/wifi_cam_mcp/`).
- `tts-mcp/`: TTS integration server — ElevenLabs, VOICEVOX, Style-Bert-VITS2 (`src/tts_mcp/`).
- `memory-mcp/`: Long‑term memory server with SQLite backend (`src/memory_mcp/`) with tests in `memory-mcp/tests/`.
- `system-temperature-mcp/`: System temperature sensor (`src/system_temperature_mcp/`).
- `desire-system/`: Curiosity system (`desire_updater.py`, `server.py`).
- `.claude/`: Claude Code local settings and hooks.
- Docs: `README.md`, `CLAUDE.md`.

## Build, Test, and Development Commands
Run commands from the target subproject directory.

- `uv sync`: Install dependencies.
- `uv run <server-name>`: Start a server (e.g., `uv run wifi-cam-mcp`).
- `uv run pytest`: Run tests (currently only in `memory-mcp/`).
- `uv run ruff check .`: Lint where configured (`memory-mcp/`, `wifi-cam-mcp/`).

## Coding Style & Naming Conventions
- Python 3.10+ baseline; `system-temperature-mcp/` requires Python 3.12+.
- 4‑space indentation, `snake_case` modules, `test_*.py` tests.
- Ruff line length is 100; asyncio is the default style for async work.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio`.
- Tests live in `memory-mcp/tests/`.
- Example: `cd memory-mcp && uv run pytest`.

## Configuration, Hardware, and WSL2 Notes
- `.env` is not committed; pass camera credentials via environment variables.
- ElevenLabs requires `ELEVENLABS_API_KEY` in the environment (see `tts-mcp/.env.example`).
- Long‑term memory stores data in `~/.claude/memories/memory.db` (SQLite).
- WSL2: USB webcams need `usbipd` forwarding; system temperature does not work under WSL2.
- Tapo cameras require a local camera account (not the TP‑Link cloud account) and a stable IP is recommended.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `feat!:`).
- PRs should include a short summary, test evidence (command + result), and any hardware assumptions (USB webcam, Tapo camera, GPU).

## ディレクトリ構造

```
embodied-claude/
├── usb-webcam-mcp/        # USB ウェブカメラ制御（Python）
│   └── src/usb_webcam_mcp/
│       └── server.py      # MCP サーバー実装
│
├── wifi-cam-mcp/          # Wi-Fi PTZ カメラ制御（Python）
│   └── src/wifi_cam_mcp/
│       ├── server.py      # MCP サーバー実装
│       ├── camera.py      # Tapo カメラ制御
│       └── config.py      # 設定管理
│
├── tts-mcp/               # TTS 統合サーバー（ElevenLabs + VOICEVOX）
│   └── src/tts_mcp/
│       ├── server.py      # MCP サーバー実装
│       ├── config.py      # 設定管理
│       ├── playback.py    # 再生ロジック
│       ├── go2rtc.py      # go2rtc プロセス管理
│       └── engines/
│           ├── __init__.py    # TTSEngine Protocol
│           ├── elevenlabs.py  # ElevenLabs エンジン
│           ├── voicevox.py    # VOICEVOX エンジン
│           └── sbv2.py        # Style-Bert-VITS2 エンジン
│
├── memory-mcp/            # 長期記憶システム（Python）
│   ├── src/memory_mcp/
│   │   ├── server.py      # MCP サーバー実装
│   │   ├── store.py       # SQLite ストア（DDL・接続管理）
│   │   ├── memory.py      # 記憶 CRUD 操作
│   │   ├── types.py       # 型定義（Emotion, Category, VerbChain, VerbStep）
│   │   ├── config.py      # 設定管理
│   │   ├── embedding.py   # 埋め込みモデル（multilingual-e5-base）
│   │   ├── vector.py      # ベクトル検索（SQLite）
│   │   ├── bm25.py        # BM25 テキスト検索
│   │   ├── scoring.py     # スコアリング関数（時間減衰・感情・重要度）
│   │   ├── verb_chain.py  # 動詞チェーン記憶（VerbChainStore）
│   │   ├── graph.py       # 重み付き記憶グラフ（動詞/名詞ノード＋エッジ）
│   │   ├── consolidation.py # 記憶統合・再生
│   │   ├── association.py # 連想ネットワーク
│   │   ├── episode.py     # エピソード記憶
│   │   ├── hopfield.py    # Hopfield ネットワーク
│   │   ├── normalizer.py  # テキスト正規化
│   │   ├── working_memory.py # 作業記憶バッファ
│   │   ├── workspace.py   # ワークスペース競合
│   │   ├── sensory.py     # 感覚バッファ（dream）
│   │   ├── image_utils.py # 画像処理ユーティリティ
│   │   └── predictive.py  # 予測符号化
│   └── tests/
│       ├── test_verb_chain.py
│       ├── test_graph.py
│       ├── test_memory.py
│       ├── test_episode.py
│       └── ...
│
├── system-temperature-mcp/ # 体温感覚（Python）
│   └── src/system_temperature_mcp/
│       └── server.py      # 温度センサー読み取り
│
├── desire-system/          # 欲求システム（Python）
│   ├── desire_updater.py   # 欲求レベル計算・JSON出力
│   ├── server.py           # MCP サーバー実装
│   └── tests/
│       └── test_desire_updater.py
│
└── .claude/               # Claude Code ローカル設定
    └── settings.local.json
```

## 開発ガイドライン

### Python プロジェクト共通

- **パッケージマネージャー**: uv
- **Python バージョン**: 3.10+
- **テストフレームワーク**: pytest + pytest-asyncio
- **リンター**: ruff
- **非同期**: asyncio ベース

```bash
# 依存関係インストール（dev含む）
uv sync --extra dev

# リント
uv run ruff check .

# テスト実行
uv run pytest

# サーバー起動
uv run <server-name>
```

### コミット前のチェック（必須）

各サブプロジェクトで以下を実行してからコミットすること:

```bash
cd <project-dir>
uv run ruff check .    # lint エラーがないこと
uv run pytest -v       # テストが通ること
```

