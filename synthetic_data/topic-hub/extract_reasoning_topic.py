import json
import sys

# python scripts/extract_reasoning_topic.py > data/reasoning_topic.jsonl


# https://cir.nii.ac.jp/crid/1520853834410310272
# https://keiho.repo.nii.ac.jp/record/329/files/daigakuronshu_110_01.pdf

reasoning_topics_1 = [
    "順序推理",
    "順序の情報を不等号を用いて表し、順序を推理",
    "対戦推理",
    "試合の勝敗から順位を推理",
    "リーグ戦方式の試合の勝敗から対応表(勝敗表)を使って順位を推理",
    "席順推理",
    "座席の配置を推理",
    "区画の配置を推理",
    "建物の配置を推理",
    "真偽推理",
    "うそつきが誰なのかを仮定し矛盾を発見",
    "背理法を使った真偽の推理",
    "時間推理",
    "時間の前後関係を用いて推理",
    "時系列に関する推理",
    "時間のズレに関する推理",
    "位置推理",
    "平面的位置関係の推理",
    "空間的位置関係の推理",
    "集合とベン図",
    "集合とベン図を使った真偽判定",
    "命題推理",
    "命題と対偶の真偽関係",
    "命題と逆の真偽関係",
    "命題と裏の真偽関係",
    "命題の結びと交わり",
    "ド・モルガンの法則",
    "2つの命題を連結して1つの命題を作る",
    "三段論法",
    "規則推理",
    "単純な暗号の問題",
    "数字や記号と文字の間の規則性を考え暗号を解く",
    "対応推理",
    "2つ以上の項目の対応関係を推理",
    "対応表に条件を整理して対応関係を推理",
    "手順推理",
    "操作の手順を追いながら推理",
    "天秤ばかりで選び出す問題",
]


# https://www.lec-jp.com/koumuin/about/handansuiri.html

reasoning_topics_2 = [
    # "論理",
    "真偽",
    "対応関係",
    "試合",
    "数量推理",
    "順序関係",
    "位置関係",
    "暗号",
    "操作手順",
    # "推理",
]


# by hand

reasoning_topics_3 = [
    "家族間の続柄",
    "親族間の続柄",
    "日常生活",
    "日常生活のトラブル",
]


print(f"length of reasoning_topics_1: {len(reasoning_topics_1)}", file=sys.stderr)
print(f"length of reasoning_topics_2: {len(reasoning_topics_2)}", file=sys.stderr)
print(f"length of reasoning_topics_3: {len(reasoning_topics_3)}", file=sys.stderr)

# Merge the lists
reasoning_topics = list(
    set(reasoning_topics_1)
    | set(reasoning_topics_2)
    | set(reasoning_topics_3)
)
print(f"length of reasoning_topics: {len(reasoning_topics)}", file=sys.stderr)

# Sort the merged list
reasoning_topics = sorted(reasoning_topics)

# Save the merged list to a file
for topic in reasoning_topics:
    print(json.dumps({"topic": topic}, ensure_ascii=False))

# print(reasoning_topics)
