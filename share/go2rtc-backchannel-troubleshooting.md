# go2rtc バックチャンネル（カメラスピーカー）トラブルシューティング

## 環境

- go2rtc: v1.9.14
- カメラ: Tapo C210 V3
- OS: Windows 11 + Git Bash

## 現在の設定

```yaml
# ~/.cache/embodied-claude/go2rtc/go2rtc.yaml
streams:
  tapo_cam:
    - rtsp://user:pass@192.168.1.139:554/stream1
    - tapo://password@192.168.1.139
  tapo_cam_bc:
    - tapo://password@192.168.1.139

ffmpeg:
  bin: ffmpeg

api:
  listen: ":1984"

log:
  level: info
```

## 症状

`POST /api/streams?dst=tapo_cam_bc&src=ffmpeg:file.wav%23audio=pcma%23input=file` で **HTTP 500 "can't find consumer"** が返る。

## 原因分析（go2rtcソースコード `internal/streams/play.go`）

### Play() 関数の動作

1. **第1ループ**: 既に接続済みのプロデューサーを走査し、`core.Consumer` インターフェースを実装しているか確認
   - 誰もストリームを見ていない → 接続済みプロデューサーなし → スキップ
2. **第2ループ**: 各ソースURLに対して `GetProducer(url)` で**新規接続**を作り、Consumer を探す
   - tapo:// は `AddTrack()` を実装しているので Consumer として使えるはず
   - **エラーは `continue` で握りつぶされる**（ログにも出ない場合がある）
3. 両方失敗 → `errors.New("can't find consumer")`

### 第2ループが失敗する可能性

| 原因 | 詳細 |
|------|------|
| 認証エラー | tapo:// の新規接続時にパスワードが通らない。`continue` で握りつぶされる |
| コーデック不一致 | 新しいファームウェアは PCMU/16000 を使う場合がある（v1.9.13で対応追加） |
| ファームウェアがtapo://をブロック | C210 V3の新しいファームウェアがtapo://プロトコルを制限している可能性 |
| サードパーティ連携OFF | Tapoアプリの「サードパーティ連携」がOFFだとtapo://接続が拒否される |

## 確認済みの事実

- カメラはping応答する（IP到達性OK）
- `GET /api/frame.jpeg?src=tapo_cam_bc` → **200 OK**（フレーム取得は成功）
- RTSPコンシューマーを繋いだ状態でも、tapo://プロデューサーは起動しない（RTSPが優先される）
- `tapo_cam_bc`（tapo://のみ）でもconsumerが確立されない
- URLエンコードは正しい（Python `urllib.parse.quote()` で `#` → `%23`）
- go2rtc web UI の frame.jpeg は両ストリームとも200を返す

## 2026-02-20 に一度成功した時の状況

- Tapoアプリの「サードパーティ連携」をONにした
- カメラ再起動 + go2rtc再起動で繋がった
- その後いつの間にか動かなくなった（原因不明）

## 2026-02-21 解決済み

### 原因

Windows版go2rtcでは、streamsのYAML設定でリスト形式（配列）が正しく読み込まれない可能性がある。go2rtc本体はリスト形式を公式サポートしているが、Windows版では型が `map[string]string` として扱われ、配列が無視/破棄される模様（GPT分析による推測）。Linux環境（kmizu氏）ではリスト形式で問題なく動作。

### 解決策

1. streamsをリスト形式ではなく、文字列で直接指定する
2. パスワードはSHA256ハッシュ形式、ユーザー名は `admin`
3. RTSPストリームは不要（tapo://のみで映像取得+バックチャンネル両方OK）

```yaml
# NG（Windows版で配列が無視される）
streams:
  tapo_cam:
    - rtsp://user:pass@192.168.1.150:554/stream1
    - tapo://pass@192.168.1.150

# OK（文字列で直接指定）
streams:
  tapo_cam: tapo://admin:SHA256HASH@192.168.1.150
```

## デバッグ手順（次回試すこと）

### 1. go2rtc のログレベルを debug にする

```yaml
log:
  level: debug
```

go2rtcを再起動して、POST リクエスト時のログを確認する。第2ループの tapo:// 接続失敗の詳細が見えるはず。

### 2. go2rtc Web UI でストリーム確認

ブラウザで `http://localhost:1984/` を開き、`tapo_cam_bc` ストリームの情報を確認。tapo://接続のコーデック情報が見える。

### 3. `#backchannel=1` を明示的に追加

```yaml
tapo_cam_bc:
  - tapo://password@host#backchannel=1
```

### 4. Tapoアプリで「サードパーティ連携」を再確認

設定 → 詳細設定 → サードパーティ連携 → ON になっているか確認。OFFになっていたらONにしてカメラを再起動。

### 5. カメラのファームウェアバージョンを確認

Tapoアプリでファームウェアバージョンを確認。v1.9.13 以降のgo2rtcは新しいコーデック(PCMU/16000)に対応済み。

### 6. go2rtc を完全再起動

```bash
# プロセスを止める
curl -X POST http://localhost:1984/api/exit

# 再起動
~/.cache/embodied-claude/go2rtc/go2rtc.exe -config ~/.cache/embodied-claude/go2rtc/go2rtc.yaml
```

## go2rtc バージョン履歴（バックチャンネル関連）

| バージョン | 変更 |
|-----------|------|
| v1.9.14 | xiaomi の双方向オーディオ修正 |
| v1.9.13 | **tapo ソースの新オーディオコーデック対応（PCMU/16000）** |
| v1.9.11 | doorbird バックチャンネル修正 |
| v1.9.10 | exec バックチャンネル基盤の書き直し、ユニバーサルPCMトランスコーダー |
| v1.9.9 | RTSPサーバーのバックチャンネル対応追加 |

## TTS MCP コード側の処理（playback.py）

```
play_with_go2rtc()
  → WAV末尾に0.3秒無音パディング追加
  → _send_audio_to_stream(tapo_cam_bc) を試す（バックチャンネル専用ストリーム）
  → 失敗したら _send_audio_to_stream(tapo_cam) にフォールバック（メインストリーム）
```

## 参考リンク

- [go2rtc play.go ソース](https://github.com/AlexxIT/go2rtc/blob/master/internal/streams/play.go)
- [tapo backchannel.go ソース](https://github.com/AlexxIT/go2rtc/blob/master/pkg/tapo/backchannel.go)
- [Issue #948: URL encoding](https://github.com/AlexxIT/go2rtc/issues/948)
- [Issue #1494: Two-way audio with new Tapo cameras](https://github.com/AlexxIT/go2rtc/issues/1494)
- [Issue #1704: Separate backchannel stream](https://github.com/AlexxIT/go2rtc/issues/1704)
- [HA Guide: Tapo C210 TTS](https://community.home-assistant.io/t/guide-tapo-camera-c100-c210-tts-on-camera-speaker/889744)
