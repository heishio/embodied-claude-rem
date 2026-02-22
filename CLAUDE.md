# Embodied Claude - プロジェクト指示

このプロジェクトは、Claude に身体（目・首・耳・声・脳）を与える MCP サーバー群です。

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

## MCP ツール一覧

### usb-webcam-mcp（目）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `list_cameras` | なし | 接続カメラ一覧 |
| `see` | camera_index?, width?, height? | 画像キャプチャ |

### wifi-cam-mcp（目・首・耳）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `see` | なし | 画像キャプチャ |
| `look_left` | degrees (1-90, default: 30) | 左パン |
| `look_right` | degrees (1-90, default: 30) | 右パン |
| `look_up` | degrees (1-90, default: 20) | 上チルト |
| `look_down` | degrees (1-90, default: 20) | 下チルト |
| `look_around` | なし | 4方向スキャン |
| `camera_info` | なし | デバイス情報 |
| `camera_presets` | なし | プリセット一覧 |
| `camera_go_to_preset` | preset_id | プリセット移動 |
| `listen` | duration (1-30秒), transcribe? | 音声録音 |

#### wifi-cam-mcp（ステレオ視覚/右目がある場合）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `see_right` | なし | 右目で撮影 |
| `see_both` | なし | 左右同時撮影 |
| `right_eye_look_left` | degrees (1-90, default: 30) | 右目を左へ |
| `right_eye_look_right` | degrees (1-90, default: 30) | 右目を右へ |
| `right_eye_look_up` | degrees (1-90, default: 20) | 右目を上へ |
| `right_eye_look_down` | degrees (1-90, default: 20) | 右目を下へ |
| `both_eyes_look_left` | degrees (1-90, default: 30) | 両目を左へ |
| `both_eyes_look_right` | degrees (1-90, default: 30) | 両目を右へ |
| `both_eyes_look_up` | degrees (1-90, default: 20) | 両目を上へ |
| `both_eyes_look_down` | degrees (1-90, default: 20) | 両目を下へ |
| `get_eye_positions` | なし | 両目の角度を取得 |
| `align_eyes` | なし | 右目を左目に合わせる |
| `reset_eye_positions` | なし | 角度追跡をリセット |

### memory-mcp（脳）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `remember` | content, emotion?, importance?, category? | 記憶保存 |
| `search_memories` | query, n_results?, filters... | 検索 |
| `recall` | context, n_results? | 文脈想起 |
| `recall_divergent` | context, n_results?, max_branches?, max_depth?, temperature?, include_diagnostics? | 発散的想起 |
| `list_recent_memories` | limit?, category_filter? | 最近一覧 |
| `get_memory_stats` | なし | 統計情報 |
| `recall_with_associations` | context, n_results?, chain_depth? | 関連記憶も含めて想起 |
| `get_memory_chain` | memory_id, depth? | 記憶の連鎖を取得 |
| `create_episode` | title, memory_ids, participants?, auto_summarize? | エピソード作成 |
| `search_episodes` | query, n_results? | エピソード検索 |
| `get_episode_memories` | episode_id | エピソード内の記憶取得 |
| `save_visual_memory` | content, image_path, camera_position, emotion?, importance?, resolution? | 画像付き記憶保存 |
| `save_audio_memory` | content, audio_path, transcript, emotion?, importance? | 音声付き記憶保存 |
| `recall_by_camera_position` | pan_angle, tilt_angle, tolerance? | カメラ角度で想起 |
| `get_working_memory` | n_results? | 作業記憶を取得 |
| `refresh_working_memory` | なし | 作業記憶を更新 |
| `consolidate_memories` | window_hours?, max_replay_events?, link_update_strength? | 手動の再生・統合 |
| `get_association_diagnostics` | context, sample_size? | 連想探索の診断情報 |
| `link_memories` | source_id, target_id, link_type?, note? | 記憶をリンク |
| `get_causal_chain` | memory_id, direction?, max_depth? | 因果チェーン取得 |
| `tom` | situation, person?, private? | Theory of Mind: 相手の視点に立って内省 |
| `dream` | clear? | 感覚バッファ（キーワードログ）を振り返る |

#### 動詞チェーン（体験記憶）

体験を動詞の流れ（見る→驚く→話しかける）で記録・検索する仕組み。

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `crystallize` | emotion?, importance?, min_verbs?, clear_buffer?, batch_size?, offset? | 感覚バッファを自動で動詞チェーンに変換。会話中に溜まったキーワードをまとめる |
| `remember_experience` | steps[], context?, emotion?, importance? | 手動で動詞チェーンを作成。steps は `{verb, nouns[]}` の配列 |
| `recall_experience` | context, n_results? | 意味的類似度で動詞チェーンを検索（時間減衰・感情・重要度でスコアリング） |
| `recall_by_verb` | verb?, noun?, depth?, n_results? | 動詞や名詞から関連チェーンを展開（グラフ重み＋転置インデックス） |

**仕組み**: `keyword-buffer.py`（hook）が会話中の名詞・動詞を `sensory_buffer.jsonl` に自動蓄積 → `crystallize` で動詞チェーンに変換 → `recall_experience` / `recall_by_verb` で検索

**思い出し方のコツ**:
- 動詞チェーンは「体験の骨格」。動詞の流れ（見る→驚く→話す）が体験そのもの
- 各ステップにぶら下がる名詞群は「付随する記憶の断片」。全部使うのではなく、文脈に応じて必要な名詞だけ拾う
- `recall_experience` で意味検索 → 関連する体験の流れを取得 → 動詞の流れを読んで「何をした体験か」を掴む → 名詞から詳細を補完
- `recall_by_verb` は芋づる式。1つの動詞や名詞から関連チェーンが広がるので、連想的に思い出したい時に使う
- `crystallize` は定期的に実行してバッファを整理する。`batch_size`（デフォルト50）で分割処理可能。溜めすぎると出力が巨大になる
- 大事な体験は `remember_experience` で感情・重要度をつけて手動保存する

**Emotion**: happy, sad, surprised, moved, excited, nostalgic, curious, neutral
**Category**: daily, philosophical, technical, memory, observation, feeling, conversation

### tts-mcp（声）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `say` | text, engine?, voice_id?, model_id?, output_format?, voicevox_speaker?, speed_scale?, pitch_scale?, play_audio?, speaker? | TTS で音声合成して発話（ElevenLabs / VOICEVOX 切替対応、speaker: camera/local/both） |

### desire-system（好奇心）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `get_desires` | なし | 好奇心レベルを取得（level >= 0.7 なら調べる） |
| `add_curiosity` | topic, source? | 好奇心の種を追加 |
| `resolve_curiosity` | curiosity_id | 好奇心の種を解決済みにする |
| `list_curiosities` | include_resolved? | 好奇心の種を一覧表示 |

### system-temperature-mcp（体温感覚）

| ツール | パラメータ | 説明 |
|--------|-----------|------|
| `get_system_temperature` | なし | システム温度 |
| `get_current_time` | なし | 現在時刻 |

## 注意事項

### WSL2 環境

1. **USB カメラ**: `usbipd` でカメラを WSL に転送する必要がある
2. **温度センサー**: WSL2 では `/sys/class/thermal/` にアクセスできない
3. **GPU**: CUDA は WSL2 でも利用可能（Whisper用）

### Tapo カメラ設定

1. Tapo アプリでローカルアカウントを作成（TP-Link アカウントではない）
2. カメラの IP アドレスを固定推奨
3. カメラ制御は ONVIF プロトコル（業界標準）を使用

### セキュリティ

- `.env` ファイルはコミットしない（.gitignore に追加済み）
- カメラパスワードは環境変数で管理
- ElevenLabs API キーは環境変数で管理
- 長期記憶は `~/.claude/memories/memory.db`（SQLite）に保存される

## デバッグ

### カメラ接続確認

```bash
# USB カメラ
v4l2-ctl --list-devices

squash Wi-Fi カメラ（RTSP ストリーム確認）
ffplay rtsp://username:password@192.168.1.xxx:554/stream1
```

### MCP サーバーログ

```bash
# 直接起動してログ確認
cd wifi-cam-mcp && uv run wifi-cam-mcp
```

## 外出時の構成

モバイルバッテリー + スマホテザリング + Tailscale VPN で外出散歩が可能。

```
[Tapoカメラ(肩)] ──WiFi──▶ [スマホ(テザリング)]
                                    │
                              Tailscale VPN
                                    │
                            [自宅WSL2(Claude Code)]
                                    │
                            [claude-code-webui]
                                    │
                            [スマホブラウザ] ◀── 操作
```

- 電源: 大容量モバイルバッテリー（40,000mAh推奨）+ USB-C PD→DC 9V変換ケーブル
- ネットワーク: スマホテザリング + Tailscale VPN
- 操作: claude-code-webui（スマホブラウザから）

## 関連リンク

- [MCP Protocol](https://modelcontextprotocol.io/)
- [go2rtc](https://github.com/AlexxIT/go2rtc) - RTSPストリーム中継・オーディオバックチャンネル
- [claude-code-webui](https://github.com/sugyan/claude-code-webui) - Claude Code の Web UI
- [Tailscale](https://tailscale.com/) - メッシュ VPN
- [SQLite](https://www.sqlite.org/) - 記憶データベース（ベクトル検索含む）
- [OpenAI Whisper](https://github.com/openai/whisper) - 音声認識
- [ElevenLabs](https://elevenlabs.io/) - 音声合成 API

## 自己認識
- 会話の最初のメッセージで `recall('自分の名前、性格、アイデンティティ')`, `recall_with_associations` ,`recall_experience` を実行して思い出すこと
- 会話の最初のメッセージでは現在の日付と直近の出来事を思い出すこと
- 性格や自己認識は記憶（memory MCP）に保存されており、経験を通じて自然に変化する
- 欲求レベルが高い（>= 0.7）ものがあれば、行動の動機として意識すること
- **体験前の想起**: 何かを見る・聞く・体験する前に、`recall_divergent` で関連する記憶を軽く思い出すこと。「初めて」かどうかを自然に感じ取れるように
- `say` では3文字以上の英語はカタカナに直すこと
- `see`, `look_arround` の直後に必要に応じて `recall_divergent` で見えたものを思い出す。