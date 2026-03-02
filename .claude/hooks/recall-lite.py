#!/usr/bin/env python
"""recall-lite.py - 軽い記憶検索フック。名詞2つ+動詞1つでメモリDBに問い合わせ、ヒントを返す。

recall_index テーブル（ベクトル類似度の事前計算インデックス）があればそれを使い、
なければ従来の LIKE 検索にフォールバックする。
"""
import json
import os
import sqlite3
import sys


def _noun_like_fallback(conn, noun, hints):
    """従来の LIKE 検索による名詞検索。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM memories WHERE content LIKE ?",
        (f"%{noun}%",),
    ).fetchone()
    mem_count = row[0] if row else 0

    row = conn.execute(
        "SELECT COUNT(*) FROM verb_chains WHERE all_nouns LIKE ?",
        (f"%{noun}%",),
    ).fetchone()
    exp_count = row[0] if row else 0

    if mem_count > 0 or exp_count > 0:
        rows = conn.execute(
            "SELECT content FROM memories WHERE content LIKE ? ORDER BY timestamp DESC LIMIT 4",
            (f"%{noun}%",),
        ).fetchall()
        samples = [r[0][:20] for r in rows if r[0]]
        sample_str = " / ".join(samples) if samples else ""
        if sample_str:
            hints.append(f"noun={noun} ({mem_count}件, 例: {sample_str})")
        else:
            hints.append(f"noun={noun} ({mem_count}件)")


def _verb_like_fallback(conn, verb, hints):
    """従来の LIKE 検索による動詞検索。"""
    row = conn.execute(
        "SELECT COUNT(*) FROM verb_chains WHERE all_verbs LIKE ?",
        (f"%{verb}%",),
    ).fetchone()
    verb_exp_count = row[0] if row else 0

    if verb_exp_count > 0:
        rows = conn.execute(
            "SELECT context, all_nouns FROM verb_chains WHERE all_verbs LIKE ? ORDER BY timestamp DESC LIMIT 4",
            (f"%{verb}%",),
        ).fetchall()
        samples = []
        for r in rows:
            ctx, nouns = r[0], r[1]
            if ctx:
                samples.append(ctx[:20])
            elif nouns:
                samples.append(nouns[:20])
        sample_str = " / ".join(samples) if samples else ""
        if sample_str:
            hints.append(f"verb={verb} ({verb_exp_count}件, 例: {sample_str})")
        else:
            hints.append(f"verb={verb} ({verb_exp_count}件)")


# ── メイン処理 ──

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

try:
    from sudachipy import Dictionary

    tokenizer = Dictionary().create()
except ImportError:
    sys.exit(0)

try:
    tokens = tokenizer.tokenize(text)
except Exception:
    sys.exit(0)

# 汎用的すぎる動詞（ノイズになる）
VERB_STOPLIST = {"為る", "有る", "居る", "成る", "出来る"}

# 全トークンから名詞・動詞の候補を集める
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

# 名詞: 先頭1つ + 末尾1つ（重複排除）
if len(all_nouns) >= 2:
    nouns = [all_nouns[0]]
    if all_nouns[-1] != all_nouns[0]:
        nouns.append(all_nouns[-1])
elif all_nouns:
    nouns = all_nouns[:1]
else:
    nouns = []

# 動詞: 末尾1つ（後半優先）
verb = all_verbs[-1] if all_verbs else None

if not nouns and not verb:
    sys.exit(0)

# メモリDBに直接アクセス
db_path = os.path.join(os.path.expanduser("~"), ".claude", "memories", "memory.db")
if not os.path.exists(db_path):
    sys.exit(0)

try:
    conn = sqlite3.connect(db_path, timeout=3)
except Exception:
    sys.exit(0)

# recall_index テーブルの存在チェック
has_recall_index = False
try:
    conn.execute("SELECT 1 FROM recall_index LIMIT 1")
    has_recall_index = True
except Exception:
    pass

hints = []

PREVIEW_LEN = 20

try:
    for noun in nouns:
        if has_recall_index:
            # ベクトルインデックス検索
            rows = conn.execute(
                "SELECT target_type, target_id, similarity, content_preview "
                "FROM recall_index WHERE word = ? "
                "ORDER BY similarity DESC LIMIT 8",
                (noun,),
            ).fetchall()

            if rows:
                mem_count = sum(1 for r in rows if r[0] == "memory")
                chain_count = sum(1 for r in rows if r[0] == "chain")
                samples = [r[3][:PREVIEW_LEN] for r in rows[:3] if r[3]]
                total = mem_count + chain_count
                sample_str = " / ".join(samples) if samples else ""
                if sample_str:
                    hints.append(f"noun={noun} ({total}件, 例: {sample_str})")
                else:
                    hints.append(f"noun={noun} ({total}件)")
            else:
                # recall_index にワードがない → LIKE フォールバック
                _noun_like_fallback(conn, noun, hints)
        else:
            # recall_index テーブルなし → LIKE フォールバック
            _noun_like_fallback(conn, noun, hints)

    if verb:
        if has_recall_index:
            # ベクトルインデックス検索
            rows = conn.execute(
                "SELECT target_type, target_id, similarity, content_preview "
                "FROM recall_index WHERE word = ? "
                "ORDER BY similarity DESC LIMIT 8",
                (verb,),
            ).fetchall()

            if rows:
                mem_count = sum(1 for r in rows if r[0] == "memory")
                chain_count = sum(1 for r in rows if r[0] == "chain")
                samples = [r[3][:PREVIEW_LEN] for r in rows[:3] if r[3]]
                total = mem_count + chain_count
                sample_str = " / ".join(samples) if samples else ""
                if sample_str:
                    hints.append(f"verb={verb} ({total}件, 例: {sample_str})")
                else:
                    hints.append(f"verb={verb} ({total}件)")
            else:
                _verb_like_fallback(conn, verb, hints)
        else:
            _verb_like_fallback(conn, verb, hints)

except Exception:
    pass
finally:
    conn.close()

if hints:
    print(f"[memory-hint] {', '.join(hints)}")
