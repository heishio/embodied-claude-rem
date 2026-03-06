#!/usr/bin/env python
"""recall-lite.py - 軽い記憶検索フック。全名詞+動詞2つでメモリDBに問い合わせ、
スコアをアグリゲートして文脈に近い記憶を返す。

recall_index テーブル（ベクトル類似度の事前計算インデックス）があればそれを使い、
なければ従来の LIKE 検索にフォールバックする。
"""
import json
import os
import re
import sqlite3
import sys


# ── 入力 ──

text = ""
try:
    data = json.load(sys.stdin)
    text = data.get("prompt", "")
except Exception:
    sys.exit(0)

if not text or len(text) < 2:
    sys.exit(0)

# autonomous-action のプロンプトはスキップ
if os.environ.get("CLAUDE_AUTONOMOUS"):
    sys.exit(0)
if "自律行動タイム" in text:
    sys.exit(0)

# サロゲート文字を除去
text = text.encode("utf-8", errors="ignore").decode("utf-8")

# <system-reminder>タグを除去
text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)

try:
    from sudachipy import Dictionary

    tokenizer = Dictionary().create()
except ImportError:
    sys.exit(0)

try:
    tokens = tokenizer.tokenize(text)
except Exception:
    sys.exit(0)

# ── 単語抽出 ──

VERB_STOPLIST = {"為る", "有る", "居る", "成る", "出来る"}
NOUN_STOPLIST = {"こと", "もの", "ため", "よう", "ところ", "はず", "わけ", "つもり", "ほう"}

all_nouns = []
all_verbs = []
for t in tokens:
    pos = t.part_of_speech()
    if pos[0] == "名詞":
        surface = t.surface()
        if surface not in NOUN_STOPLIST and len(surface) >= 2:
            all_nouns.append(surface)
    elif pos[0] == "動詞":
        lemma = t.normalized_form()
        if lemma not in VERB_STOPLIST:
            all_verbs.append(lemma)

# 名詞: 全部（重複排除、順序保持）
nouns = list(dict.fromkeys(all_nouns))
# 動詞: 末尾2つ（重複排除）
unique_verbs = list(dict.fromkeys(all_verbs))
verbs = unique_verbs[-2:] if len(unique_verbs) >= 2 else unique_verbs

query_words = [(w, "noun") for w in nouns] + [(w, "verb") for w in verbs]

if not query_words:
    sys.exit(0)

# ── DB接続 ──

db_path = os.environ.get("MEMORY_DB_PATH", os.path.join(os.path.expanduser("~"), ".claude", "memories", "memory.db"))
if not os.path.exists(db_path):
    sys.exit(0)

try:
    conn = sqlite3.connect(db_path, timeout=3)
except Exception:
    sys.exit(0)

has_recall_index = False
try:
    conn.execute("SELECT 1 FROM recall_index LIMIT 1")
    has_recall_index = True
except Exception:
    pass

PREVIEW_LEN = 30
MAX_RESULTS = 8


def _like_fallback(conn, nouns, verbs):
    """recall_index が使えない場合の LIKE フォールバック。"""
    hints = []
    for noun in nouns[:2]:
        row = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content LIKE ?",
            (f"%{noun}%",),
        ).fetchone()
        cnt = row[0] if row else 0
        if cnt > 0:
            rows = conn.execute(
                "SELECT content FROM memories WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 3",
                (f"%{noun}%",),
            ).fetchall()
            samples = [r[0][:PREVIEW_LEN] for r in rows if r[0]]
            sample_str = " / ".join(samples)
            if sample_str:
                hints.append(f"noun={noun} ({cnt}件, 例: {sample_str})")
            else:
                hints.append(f"noun={noun} ({cnt}件)")
    for verb in verbs:
        row = conn.execute(
            "SELECT COUNT(*) FROM verb_chains WHERE all_verbs LIKE ?",
            (f"%{verb}%",),
        ).fetchone()
        cnt = row[0] if row else 0
        if cnt > 0:
            hints.append(f"verb={verb} ({cnt}件)")
    if hints:
        print(f"[memory-hint] {', '.join(hints)}")


try:
    if has_recall_index:
        # ── アグリゲート方式 ──
        # 全単語について recall_index を引き、(target_type, target_id) でスコア合算
        scores = {}       # (type, id) → total_similarity
        previews = {}     # (type, id) → content_preview
        hit_words = {}    # (type, id) → set of words that matched
        flow_sims = {}    # (type, id) → [flow_sim values]
        delta_sims = {}   # (type, id) → [delta_sim values]

        # Check if flow_sim/delta_sim columns exist
        has_fd = False
        try:
            conn.execute("SELECT flow_sim, delta_sim FROM recall_index LIMIT 1")
            has_fd = True
        except Exception:
            pass

        for word, wtype in query_words:
            if has_fd:
                rows = conn.execute(
                    "SELECT target_type, target_id, similarity, content_preview, "
                    "flow_sim, delta_sim "
                    "FROM recall_index WHERE word = ? "
                    "ORDER BY similarity DESC LIMIT 10",
                    (word,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT target_type, target_id, similarity, content_preview "
                    "FROM recall_index WHERE word = ? "
                    "ORDER BY similarity DESC LIMIT 10",
                    (word,),
                ).fetchall()

            for row in rows:
                target_type, target_id, similarity, preview = row[0], row[1], row[2], row[3]
                f_sim = row[4] if has_fd and len(row) > 4 else None
                d_sim = row[5] if has_fd and len(row) > 5 else None
                key = (target_type, target_id)
                scores[key] = scores.get(key, 0.0) + similarity
                if key not in previews and preview:
                    previews[key] = preview
                if key not in hit_words:
                    hit_words[key] = set()
                hit_words[key].add(word)
                # Accumulate flow/delta sims for averaging
                if f_sim is not None:
                    if key not in flow_sims:
                        flow_sims[key] = []
                    flow_sims[key].append(f_sim)
                if d_sim is not None:
                    if key not in delta_sims:
                        delta_sims[key] = []
                    delta_sims[key].append(d_sim)

        if scores:
            # 複数単語がヒットした記憶にボーナス
            for key in scores:
                n_hits = len(hit_words[key])
                if n_hits > 1:
                    scores[key] *= (1.0 + 0.2 * (n_hits - 1))

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:MAX_RESULTS]
            total = len(scores)
            samples = []
            for key, score in ranked[:4]:
                p = previews.get(key, "")
                if p:
                    samples.append(p[:PREVIEW_LEN])

            # Compute flow/delta averages over top hits
            fd_tag = ""
            if flow_sims and delta_sims:
                top_keys = [k for k, _ in ranked]
                f_vals = [v for k in top_keys for v in flow_sims.get(k, [])]
                d_vals = [v for k in top_keys for v in delta_sims.get(k, [])]
                if f_vals and d_vals:
                    f_avg = sum(f_vals) / len(f_vals)
                    d_avg = sum(d_vals) / len(d_vals)
                    if abs(f_avg - d_avg) > 0.15:
                        fd_tag = f" f{f_avg:.1f}/d{d_avg:.1f}"

            sample_str = " / ".join(samples)
            words_str = ",".join(w for w, _ in query_words)
            if sample_str:
                print(f"[memory-hint] [{words_str}] ({total}件, 例: {sample_str}){fd_tag}")
            else:
                print(f"[memory-hint] [{words_str}] ({total}件){fd_tag}")
        else:
            # recall_index にヒットなし → LIKE フォールバック
            _like_fallback(conn, nouns, verbs)

    else:
        # recall_index なし → LIKE フォールバック
        _like_fallback(conn, nouns, verbs)

except Exception:
    pass
finally:
    conn.close()
