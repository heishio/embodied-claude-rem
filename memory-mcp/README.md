# memory-mcp

embodied-claude の長期記憶システム。SQLite + numpy ベクトルで構築。[概念設計はこちら](DESIGN.md)。

## 全体構成

```
会話 → keyword-buffer.py(hook) → sensory_buffer.jsonl → crystallize → 動詞チェーン
                                                                         ↓
会話 → diary(MCP tool) → normalize → chiVe embedding → SQLite保存 → recall_index更新
                                                        ↑
                                              consolidate(定期) → 共活性化強化
                                                                → 合成記憶生成
                                                                → バウンダリー更新
                                                                → 交差検出
                                                                → freshness減衰
```

## ストレージ (SQLite)

| テーブル | 役割 |
|---------|------|
| `memories` | 記憶本体（content, emotion, importance, freshness, level 等） |
| `embeddings` | chiVe 2ベクトル（flow_vector + delta_vector, 各300次元 float32 BLOB） |
| `verb_chains` | 動詞チェーン（体験記憶） |
| `graph_nodes` / `graph_edges` | 動詞・名詞の共起グラフ（重み付き） |
| `coactivation` | 記憶ペアの共活性化重み（0.0〜1.0） |
| `recall_index` | 単語→記憶の類似度テーブル（語彙ごとtop-20） |
| `composite_members` | 合成記憶のメンバー関係（二重所属対応） |
| `composite_axes` | 合成記憶の主成分軸ベクトル（異方的距離計算用） |
| `boundary_layers` | 合成記憶のエッジ分類（ノイズ層） |
| `template_biases` | 動詞チェーンテンプレートのバイアス蓄積 |
| `intersections` | 合成記憶間の交差情報（parallel / transversal） |
| `episodes` | 記憶のエピソード的グルーピング |
| `categories` | グラフノードの階層的カテゴリ |

DBファイル: `~/.claude/memories/memory.db`

## エンベディング（2ベクトルアーキテクチャ）

- **モデル**: [chiVe](https://github.com/WorksApplications/chiVe)（日本語word2vec, 300次元）
- **2ベクトル構造**: 各記憶は2つのベクトルを持つ
  - `flow_vector`（300次元）: 動詞バイグラム中点の平均。「何をしたか」
  - `delta_vector`（300次元）: 名詞平均 − 動詞平均。「何に対してか」
- **flow_vectorの計算**:
  1. 各動詞のchiVeベクトルを取得
  2. 汎用動詞（する/ある/いる/言う/くれる/みる等）が左側のバイグラムをスキップ（補助動詞的用法は前の動詞に吸収されるが、次への橋渡しはしない）
  3. 残ったバイグラムの中点を計算
  4. ブックエンド: (先頭動詞+末尾動詞)/2 をバイグラム数だけ繰り返し追加（チェーンの始まりと終わりの方向性を保持）
  5. 全中点の平均 → L2正規化
- **delta_vectorの計算**: 名詞群のchiVe平均 − 動詞群のchiVe平均 → L2正規化
- **テキスト→ベクトル**: sudachipyで動詞・名詞を抽出 → chiVeで各単語をベクトル化 → 上記の計算
- **OOV処理**: normalized_form → SplitMode.A分割 → skip の順でフォールバック
- 後方互換: legacy `vector` カラムに flow+delta の concat（600次元）を保存
- **性能**: E5比でflow stdが16倍改善（0.009→0.142）、軸間相関0.43（独立に近い）

## 記憶の保存フロー

```
content
  → normalize_japanese()     # NFKC正規化, ヴ→バ, 小書きカナ統一等
  → get_reading()            # sudachipyで読み取得
  → chiVe encode_text()      # sudachiで動詞・名詞抽出 → 2ベクトル生成
  → SQLite INSERT            # memories + embeddings (flow_vector, delta_vector)
  → BM25 dirty flag          # 次回検索時に再構築
  → freshness -0.003         # 既存記憶のfreshness微減
  → recall_index更新          # 新記憶に対する語彙類似度計算
```

## 検索の仕組み（ハイブリッド）

### 1. ベクトル検索（2軸セマンティック）
クエリを chiVe で2ベクトルにエンコードし、2軸の重み付きコサイン類似度で検索。

```
similarity = 0.6 × cos(query_flow, mem_flow) + 0.4 × cos(query_delta, mem_delta)
```

flow軸（行動）とdelta軸（対象）を独立に比較するため、「同じことを違うものにした」「違うことを同じものにした」を区別できる。

### 2. BM25（キーワード）
英語: 単語分割、日本語: 文字バイグラム（"打ち合わせ"→["打ち","ち合","合わ","わせ"]）。
形態素解析なしで部分文字列マッチを実現。

### 3. スコアリング
```
final_score = semantic_distance × 1.0
            + (1.0 - time_decay) × 0.3    # 古いほどペナルティ
            - importance_boost × 0.2       # 重要な記憶は優先
```

time_decay: `2^(-経過日数 / 30)` （半減期30日）

## Freshness（鮮度）

記憶の主観的な「最近さ」を表すスカラー値。

- **範囲**: 1.0（直近）〜 0.01（フロア、消えない）
- **保存時微減**: -0.003（consolidate以降の記憶のみ）
- **consolidate時減衰**: ×0.92（全記憶一律）
- **用途**: recall系ツールで `freshness_min` / `freshness_max` フィルタ

## 動詞チェーン（体験記憶）

体験を「動詞の流れ」で記録する仕組み。

```
例: 見る(コウタ, 空) → 気になる(色) → 調べる(wiki) → 理解する(天気)
```

### 蓄積
`keyword-buffer.py`（hook）が会話中の名詞・動詞をsudachipyで抽出し、`sensory_buffer.jsonl` に蓄積。

### 結晶化（crystallize）
バッファを読み込み、共有名詞でグルーピングして動詞チェーンに変換。

### 汎用動詞フィルタ

flow_vector計算時に、汎用動詞（補助動詞的に使われる動詞）を特別扱いする。

```
_GENERIC_VERBS = {為る, 有る, 居る, 成る, 出来る, 言う, 呉れる, 遣る, 来る, 見る, 行く, 貰う, 置く}
```

日本語の「食べてみる」「やってくれる」等では、「みる」「くれる」は前の動詞の修飾であり、次の動詞への橋渡しはしない。そのため、汎用動詞が左側に来るバイグラムをスキップする。

```
例: 食べる → みる → 驚く
  バイグラム: (食べる, みる) → スキップ（みるが汎用動詞）
             (みる, 驚く) → 採用（みるは右側なのでOK）
  結果: 「食べる」は前に吸収、「驚く」との関係が残る
```

これにより、flow stdが改善（汎用動詞がflow類似度を潰す問題を解消）。

### ブックエンド

チェーンの「始まり→終わり」の方向性を保持するため、先頭動詞と末尾動詞の中点をバイグラム数だけ繰り返し追加する。

```
例: 探す → 見つける → 理解する
  バイグラム中点: mid(探す, 見つける), mid(見つける, 理解する)
  ブックエンド:   mid(探す, 理解する), mid(探す, 理解する)   ← 2回（バイグラム数分）
  flow = mean(4つの中点) → L2正規化
```

バイグラム平均だけだと「AがBに変わる」と「BがAに変わる」の区別が弱くなるが、ブックエンドにより始点と終点の関係が強調される。

### 4象限分析（Gentner）

flow軸とdelta軸の独立性により、記憶ペアの関係を4象限で分類できる：

| 象限 | flow | delta | 意味 | 例 |
|------|------|-------|------|-----|
| Literal | 高 | 高 | 同じことを同じ対象に | 同一体験 |
| Analogy | 高 | 低 | 同じことを違う対象に | 構造的類似 |
| Surface | 低 | 高 | 違うことを同じ対象に | 表面的類似 |
| Anomaly | 低 | 低 | 全く違う | 無関係 |

実測値（verb chains 1,957件）: Literal 33.1%, **Analogy 16.9%**, **Surface 16.9%**, Anomaly 33.1%。

### グラフ（補助的）
動詞・名詞をノードとした重み付きグラフ。`recall_by_verb` の芋づる式展開で使用。

| エッジ種 | 意味 | 初期重み |
|---------|------|---------|
| vv | 動詞→動詞（連続動作） | 0.3 |
| vn | 動詞→名詞（動作と対象） | 0.2 |
| nn | 名詞↔名詞（同ステップ共起） | 0.1 |

recall時に同じ重みで強化、consolidate時に×0.95で減衰、<0.01で刈り取り。

## 発散的想起（recall_divergent）

Global Workspace Theory（GWT）に基づく多段階検索。

```
クエリ
  → ベクトル検索（シード取得）
  → Association.spread()     # coactivation + linksでグラフ展開
  → 合成記憶のメンバー展開    # level>=1の構成記憶を含める
  → Workspace競合            # 多基準スコアリングで勝者選択
  → 結果返却 + activation記録
```

### Workspace スコア
```
utility = 0.45×relevance + 0.20×novelty + 0.20×prediction_error
        + 0.10×boundary_score
        - 0.25×redundancy_penalty
```
temperature（デフォルト0.7）で確率的に選択。低い=決定的、高い=探索的。

## コンソリデーション（統合）

定期的に実行する記憶の整理プロセス。

### Phase 1: 共活性化リプレイ
直近24時間の隣接記憶ペアの共活性化重みを+0.2。prediction_errorを10%減少。

### Phase 2: 合成記憶生成（多段）

類似した記憶を Union-Find でグループ化し、グループの代表ベクトルを生成する。

```
L0: 個別の記憶
 ↓ Union-Find（閾値 0.75、グループサイズ 2〜8）
L1: 類似グループの合成記憶
 ↓ 孤立記憶の救出（最寄り composite に吸収）
 ↓ クラスタ重なり検出（二重所属メンバーの追加）
 ↓ Union-Find（閾値 0.55）
L2: グループのグループ（より粗い抽象化）
```

- 代表ベクトルは importance による重み付き平均（L2正規化）
- 同じメンバー構成の合成記憶は重複生成しない
- 孤立記憶は最寄り composite に contribution_weight=0.3 で吸収
- クラスタ間の二重所属は contribution_weight=0.5 で追加

### Phase 3: バウンダリー層（ノイズ層）

合成記憶の内部で各メンバーを edge（外縁）/ core（中心）に分類する。

1. **Layer 0**: ノイズなしで分類（重心からの距離ベース + 主成分軸による異方的距離）
2. **Layer 1〜n**: 動詞チェーンのテンプレートベクトルを使ったノイズを付加して再分類
3. **バイアス蓄積**: Layer 0 と Layer k のハミング距離（分類の揺れ）に応じてテンプレートバイアスを蓄積
4. **バイアス減衰**: consolidateのたびに×0.95で減衰、閾値以下は刈り取り

よく効くテンプレート（よくある体験パターン）のバイアスが育ち、そのパターン方向への連想が広がりやすくなる。

### Phase 4: 交差検出

合成記憶間の関係を主成分軸の方向で分類する。

| 種類 | 条件 | 意味 |
|------|------|------|
| parallel | \|cos\| ≥ 0.8 | 同次元の重なり（似た文脈） |
| transversal | \|cos\| ≤ 0.3 かつ共有メンバーあり | 横断的交差（異なる文脈が交わる） |

transversal 交差のメンバーは recall_divergent で intersection boost を受け、「全然違う文脈なのに同じ記憶が浮かぶ」連想の飛躍を可能にする。

### Phase 5: Freshness減衰
全記憶の freshness を ×0.92。

## recall-lite（自動想起フック）

会話のたびに裏で動く軽量検索フック。ユーザーの発話から名詞・動詞を抽出し、recall_index（事前計算テーブル）を使って関連する記憶を即座に検索する。結果は `[memory-hint]` としてコンテキストに注入される。

```
ユーザー発話 → sudachi形態素解析 → 全名詞(重複排除) + 動詞(末尾2つ)
  → recall_index検索（事前計算済み類似度テーブル、各単語top-10）
  → (target_type, target_id) でスコアをアグリゲート
  → 複数単語ヒット時にボーナス（×1.0 + 0.2×(n_hits-1)）
  → [memory-hint] [梅,見る] (8件, 例: ...) として注入
```

keyword-buffer が「溜める」のに対して、recall-lite は「今すぐ調べてヒントを返す」。recall_indexが未構築の場合はLIKE検索にフォールバック。

## その他のコンポーネント

| モジュール | 役割 |
|-----------|------|
| `hopfield.py` | Modern Hopfield Network。パターン補完のフォールバック（600次元concat, β=4.0） |
| `working_memory.py` | セッション中のインメモリバッファ（容量20, 再起動で消える） |
| `sensory.py` | 画像・音声の記憶への紐付け（base64エンコード, カメラ位置情報） |
| `normalizer.py` | 日本語テキスト正規化（NFKC, ヴ→バ, 小書きカナ等。純Python） |
| `predictive.py` | 予測誤差・新規性スコア。Jaccard重複ベースの文脈関連度 |
| `episode.py` | 記憶のエピソード的グルーピング（narrative grouping） |

## 依存ライブラリ

- `gensim` — chiVe word2vecモデルの読み込み
- `rank-bm25` — BM25Plusスコアリング
- `numpy` — ベクトル演算
- `sudachipy` — 日本語形態素解析（動詞・名詞抽出, reading取得）
- `mcp` — Model Context Protocol サーバー
- `sqlite3` — ストレージ（標準ライブラリ）


## License

MIT