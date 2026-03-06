# heishio-integrated ブランチの差分解説

heishio/main からの全差分（コミット済み 7 件 + 未コミット変更）

---

## コミット一覧

| # | コミット | 内容 |
|---|---------|------|
| 1 | `7bd2038` | mcpBehavior.toml + jurigged live reload + tilt direction fix |
| 2 | `7a0d6be` | wifi-cam: pan/tilt/see/camera_info レスポンスに position を含める |
| 3 | `19b7961` | autonomous-action.sh — desire system, schedule config, continuation chain |
| 4 | `e82a3db` | hearing MCP + mcpBehavior.toml + interoception hook |
| 5 | `3223bfc` | chiVe .kv ファイルの読み込みを KeyedVectors.load() に修正 |
| 6 | `4c0b739` | macOS 対応 — .cmd フックを .sh に統一、venv python を使用 |
| 7 | `bf7ea2b` | recall-lite.py が MEMORY_DB_PATH 環境変数を参照するように修正 |
| - | (未コミット) | 起動高速化、recall index 行列演算化、save_batch、diary description 強化、backfill-keywords |

---

## 1. mcpBehavior.toml + jurigged live reload (`7bd2038`)

### 変更ファイル

- `mcpBehavior.toml` (新規)
- `wifi-cam-mcp/src/wifi_cam_mcp/camera.py`
- `wifi-cam-mcp/src/wifi_cam_mcp/server.py`

### 内容

- プロジェクトルートに `mcpBehavior.toml` を追加。wifi-cam の `mount_mode` や tilt 方向の設定をランタイムで変更可能にする
- camera.py の `_move_impl` / `get_hw_position` が `mcpBehavior.toml` から `mount_mode` を読むように変更
- tilt 方向の修正（ceiling/normal モード対応）

### 意図

カメラの取り付けモードやチルト方向を MCP サーバー再起動なしで切り替え可能にする。

---

## 2. wifi-cam: position レスポンス追加 (`7a0d6be`)

### 変更ファイル

- `wifi-cam-mcp/src/wifi_cam_mcp/server.py`

### 内容

- `see`, `look_left/right/up/down`, `camera_info` のレスポンスに現在の pan/tilt position を含めるように変更

### 意図

カメラ操作後の位置を毎回確認する追加呼び出しを不要にする。

---

## 3. autonomous-action.sh — desire system (`19b7961`)

### 変更ファイル

- `autonomous-action.sample.sh` (新規)
- `desires.sample.conf` (新規)
- `schedule.sample.conf` (新規)
- `scripts/desire-tick.ts` (新規)
- `scripts/interoception.ts` (新規)
- `test-autonomous.sh` (新規)
- `.claude/commands/awake.md`, `.claude/commands/sleep.md` (新規)
- `.claude/hooks/continue-check.sh` (新規)
- `.gitignore`

### 内容

- cron / launchd から定期実行される `autonomous-action.sh` のサンプルスクリプト
- `schedule.conf` による実行確率・時間帯制御
- `desires.conf` による好奇心の種の管理
- `desire-tick.ts` が好奇心レベルを計算し、閾値を超えたら行動
- `continue-check.sh` による session 継続判定
- `/sleep`, `/awake` コマンドで活動頻度を変更

### 意図

Claude が定期的に自発的に起動し、好奇心に基づいて行動する仕組み。schedule.conf で時間帯や確率を制御し、desires.conf で「調べたいこと」を管理する。

---

## 4. hearing MCP + interoception hook (`e82a3db`)

### 変更ファイル

- `hearing/` ディレクトリ一式 (新規)
- `.claude/hooks/hearing-daemon.py`, `hearing-hook.sh`, `hearing-stop-hook.sh` (新規)
- `.claude/hooks/interoception.sh` (大幅変更)
- `.claude/hooks/post-compact-recovery.sh` (新規)
- `.claude/hooks/statusline.ts` (新規)
- `.claude/settings.json`

### 内容

- hearing MCP サーバー: Wi-Fi カメラの RTSP 音声を常時監視し、mlx-whisper で文字起こし
- VAD（音声区間検出）+ ノイズフィルタ + 発話バッファリング
- hearing-hook.sh: hook 経由でバッチ注入（`[hearing]` タグ）
- hearing-stop-hook.sh: Stop hook でイベントドリブン通知
- interoception.sh の大幅拡張: session 判定、曜日、arousal、thermal、mem_free、uptime、heartbeats、context 使用率
- post-compact-recovery.sh: コンテキスト圧縮後の状態復元
- statusline.ts: ターミナルステータスバー表示

### 意図

Claude に「聞く」能力を追加。常時マイク監視 → 文字起こし → hook 注入で、ユーザーの音声発話を会話に取り込む。interoception の拡張は「身体感覚」の充実。

---

## 5. chiVe .kv 読み込み修正 (`3223bfc`)

### 変更ファイル

- `memory-mcp/src/memory_mcp/chive.py`

### 内容

- chiVe の `.kv` ファイル読み込みを `KeyedVectors.load_word2vec_format()` → `KeyedVectors.load()` に変更

### 意図

gensim の `.kv` 形式（ネイティブバイナリ）を正しく読み込む。`.kv` は word2vec テキスト形式ではないため。

---

## 6. macOS 対応 — .cmd → .sh 統一 (`4c0b739`)

### 変更ファイル

- `.claude/hooks/*.sh` (複数)
- `.claude/hooks/run-keyword-buffer.sh`
- `.claude/hooks/run-recall-lite.sh` (新規)
- `.claude/hooks/run-uncertainty-check.sh` (新規)
- `.claude/settings.json`

### 内容

- Windows 用 `.cmd` フックを `.sh` に統一
- Python 実行パスを `memory-mcp/.venv/bin/python3` に変更（venv 内の python を直接使用）
- settings.json のフックコマンドを `.sh` に書き換え

### 意図

macOS / Linux 環境での動作統一。元の heishio/main は Windows 前提の `.cmd` フックだった。

---

## 7. recall-lite.py 環境変数対応 (`bf7ea2b`)

### 変更ファイル

- `.claude/hooks/recall-lite.py`

### 内容

- DB パスを `MEMORY_DB_PATH` 環境変数から取得するように変更

### 意図

DB の配置先が環境によって異なる場合に対応。

---

## 8. 未コミット変更（ステージ済み）

### memory-mcp/src/memory_mcp/server.py

- **chiVe 遅延読み込み**: 起動時に `chive._load()` を呼ばず、最初の `recall` 時に recall index を構築
- **diary description 強化**: 「レポートではなく体験を書く」ルールを description に埋め込み
- **crystallize の save_batch 化**: ループ `save()` → 一括 `save_batch()` に変更
- **起動ログのタイムスタンプ化**: `_elapsed()` で各ステップの経過時間を表示

解決する問題: 起動が遅い（chiVe 読み込み）、diary が事実の羅列になりがち

### memory-mcp/src/memory_mcp/store.py

- **recall index のバッチ行列演算化**: cosine_similarity ループ → numpy 行列乗算 `@` に変更
- **`np.argpartition` で top-K 抽出**: full sort より高速
- **per-target delta mask**: delta ベクトルの有無をターゲットごとに追跡。delta なしターゲットは `combined = flow_sim`（旧挙動維持）

解決する問題: 記憶増加に伴う recall index 構築の遅延、行列演算化による意図しないスコア変更の防止

### memory-mcp/src/memory_mcp/verb_chain.py

- **meta テーブル永続化の廃止**: 起動時に毎回 DB から構築する方式に簡素化
- **`_persist_index()` 削除**
- **`_decay_freshness()` 共通化**: `save()` と `save_batch()` の重複ロジックをヘルパーに抽出
- **`save_batch()` 新設**: crystallize 用一括保存（freshness decay 1回、1トランザクション）

解決する問題: 永続化インデックスの不整合リスク、crystallize の大量保存時の低速、decay ロジック重複

### wifi-cam-mcp/src/wifi_cam_mcp/_behavior.py (新規)

- `mcpBehavior.toml` を読む薄いラッパー。毎回ファイルを読む（キャッシュなし）のでライブリロード対応
- camera.py が import して `mount_mode` を取得

### .claude/hooks/backfill-keywords-batch.py + backfill-keywords.sh (新規)

- 過去のトランスクリプトから `sensory_buffer.jsonl` にキーワードを一括追記するユーティリティ
- `ccconv raws --format=talk` → sudachipy で名詞・動詞抽出
- `backfill-keywords.sh` の `trap` 内 `$TMPFILE` クオート修正済み
