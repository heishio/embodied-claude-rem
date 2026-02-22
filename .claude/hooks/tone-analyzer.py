"""tone-analyzer.py - 軽量テキストトーン分析（キーワードベース）
stdin からテキストを受け取り、推定トーンを1語で stdout に出力する。
MCP呼び出しなし、記憶検索なし、純粋にパターンマッチのみ。
"""
import sys
import re

text = sys.stdin.read().strip()
if not text:
    sys.exit(0)

# トーンパターン（優先度順、最初にマッチしたものを採用）
patterns = [
    # 愛情・親密
    ("affectionate", [
        r"大好き", r"好き", r"愛してる", r"ありがとう", r"嬉しい",
        r"love", r"thx", r"thanks",
    ]),
    # ふざけ・遊び
    ("playful", [
        r"[wW]$", r"ｗ$", r"笑", r"草", r"lol", r"lmao",
        r"ﾜﾛﾀ", r"ワロタ", r"ちょ[wｗ]",
        r"茶化", r"冗談", r"ネタ",
    ]),
    # 不満・苛立ち
    ("frustrated", [
        r"うまくいかない", r"動かない", r"壊れ", r"バグ", r"エラー",
        r"だめ", r"ダメ", r"無理", r"最悪",
        r"broken", r"bug", r"error", r"fail",
    ]),
    # 疲労・眠気
    ("tired", [
        r"眠", r"疲れ", r"だるい", r"しんどい", r"寝",
        r"tired", r"sleepy", r"exhausted",
        r"zzz", r"Zzz",
    ]),
    # 好奇心・興味
    ("curious", [
        r"なんで", r"どうして", r"なぜ", r"どうやって",
        r"気になる", r"知りたい", r"教えて",
        r"how", r"why", r"what if",
    ]),
    # 依頼・お願い
    ("requesting", [
        r"して[ほくね]", r"お願い", r"頼[むみ]",
        r"できる？", r"できない？", r"ください",
        r"please", r"can you", r"could you",
    ]),
    # 満足・承認
    ("satisfied", [
        r"いいね", r"いい感じ", r"よさそう", r"完璧", r"さすが",
        r"やったね", r"ナイス", r"グッド",
        r"nice", r"great", r"perfect", r"awesome", r"good",
    ]),
    # 不安・心配
    ("anxious", [
        r"大丈夫", r"心配", r"不安", r"怖い",
        r"worried", r"afraid", r"nervous",
    ]),
    # 軽い挨拶
    ("greeting", [
        r"おはよう", r"こんにちは", r"こんばんは", r"おやすみ",
        r"ただいま", r"おかえり",
        r"^hi$", r"^hey$", r"^hello",
    ]),
]

detected = "neutral"
for tone, regexes in patterns:
    for pat in regexes:
        if re.search(pat, text, re.IGNORECASE):
            detected = tone
            break
    if detected != "neutral":
        break

print(detected)
