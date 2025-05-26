import argparse
import json
from logging import DEBUG, StreamHandler, getLogger, ERROR, INFO  # noqa: F401

import numpy as np

# Set logging level
logger = getLogger(__name__)
# logger.setLevel(DEBUG)
logger.setLevel(INFO)
handler = StreamHandler()
# handler.setLevel(DEBUG)
handler.setLevel(INFO)
logger.addHandler(handler)


# QUESTION_ROLE = "問題:\n"
# ANSWER_ROLE = "解答:\n"
# SEPARATOR = "\n\n"
QUESTION_ROLE = "user: "
ANSWER_ROLE = "assistant: "
SEPARATOR = "\n\n"


ATTRIBUTES = {
    "len": "要素数",
    "sum": "合計",
    "mean": "平均値",
    "median": "中央値",
    "max": "最大値",
    "min": "最小値",
    "sorted_asc": "昇順",
    "sorted_desc": "降順",
}


def generate_random_array():
    a_len = np.random.randint(5, 10)
    a_low = np.random.choice([0, 0, 0, -10, -100])
    a_high = np.random.choice([10, 10, 10, 100]) if a_low == 0 else np.abs(a_low)
    a = np.random.randint(a_low, a_high, a_len)
    logger.debug(f"a_len: {a_len}, a_low: {a_low}, a_high: {a_high}, a: {a}")
    return a


def get_attributes(a):
    a_len = len(a)
    a_sum = np.sum(a)
    a_mean = np.mean(a)
    a_median = np.median(a)
    a_max = np.max(a)
    a_min = np.min(a)
    a_sorted_asc = np.sort(a)
    a_sorted_desc = np.sort(a)[::-1]
    logger.debug(
        f"a_len: {a_len}, a_sum: {a_sum}, a_mean: {a_mean}, a_median: {a_median}, a_max: {a_max}, a_min: {a_min}, a_sorted_asc: {a_sorted_asc}, a_sorted_desc: {a_sorted_desc}"  # noqa: E501
    )
    return {
        "len": a_len,
        "sum": a_sum,
        "mean": a_mean,
        "median": a_median,
        "max": a_max,
        "min": a_min,
        "sorted_asc": a_sorted_asc,
        "sorted_desc": a_sorted_desc,
    }


def generate_qa_sort(sequence, sequence_sorted, sort_order):
    # Input this prompt to Nemotron-4-340B-Instruct and get 10 templates.
    # Then, tweak the templates to make them more natural by hand.
    #
    # 小学校の算数のドリルを作ろうとしています。任意の数列をソートする問題について、ドリルの文章のテンプレートを生成して、その数値部分を変数で置き換えてください。テンプレートは Python の文字列形式で出力してください。解説は不要で、テンプレートだけ10個出力してください。  # noqa: E501
    template_question_sort = [
        "次の数列を{sort_order}に並び替えなさい: {sequence}",
        "以下の数を{sort_order}に並べ替えてください: {sequence}",
        "この数列を{sort_order}にソートしなさい: {sequence}",
        "次の数を{sort_order}に並べ替えて数列を作りなさい: {sequence}",
        "以下の数列を{sort_order}に並び替えてください: {sequence}",
        "次の数を{sort_order}に並べ替えて数列を作り、答えを書きなさい: {sequence}",
        "以下の数を{sort_order}に並べ替えて数列を作り、その数列を書きなさい: {sequence}",
        "次の数列を{sort_order}に並び替えて答えを書きなさい: {sequence}",
        "以下の数列を{sort_order}にソートし、その結果を書きなさい: {sequence}",
        "次の数を{sort_order}に並べ替えて数列を作り、その数列を答えなさい: {sequence}",
    ]
    # prompt:
    #
    # 次の数列を{昇順/降順}に並び替えなさい: {数列}
    #
    # 上記の問題に対する解答のテンプレートを出力してください。解説は不要で、テンプレートだけ10個出力してください。
    template_answer_sort = [
        "{sort_order}に並び替えた数列は {sorted_sequence} となります。",
        "{sort_order}に並べ替えると、 {sorted_sequence} となります。",
        "数列 {sequence} を{sort_order}に並べ替えると、 {sorted_sequence} になります。",
        "{sort_order}に並べ替えた数列 {sequence} は、 {sorted_sequence} です。",
        "数列 {sequence} を{sort_order}に並べ替えた結果は、 {sorted_sequence} です。",
        "数列 {sequence} を {sort_order}に並べ替えると、 {sorted_sequence} が得られます。",
        "{sort_order}に並べ替えた数列 {sequence} は以下の通りです: {sorted_sequence}",
        "数列 {sequence} を{sort_order}に並べ替えると、 {sorted_sequence} となります。",
        "数列 {sequence} を{sort_order}に並べ替えた数列は {sorted_sequence} です。",
        "{sort_order}に並べ替えた数列 {sequence} は {sorted_sequence} となります。",
    ]
    logger.debug(f"sort_order: {sort_order}, sequence: {sequence}, sequence_sorted: {sequence_sorted}")
    question = np.random.choice(template_question_sort).format(sequence=sequence, sort_order=sort_order)
    answer = np.random.choice(template_answer_sort).format(sequence=sequence, sorted_sequence=sequence_sorted, sort_order=sort_order)  # noqa: E501
    logger.debug(f"question: {question}")
    logger.debug(f"answer: {answer}")
    return json.dumps({"text": f"{QUESTION_ROLE}{question}{SEPARATOR}{ANSWER_ROLE}{answer}"}, ensure_ascii=False)


def generate_qa_attr(sequence, attr_label, attr_value):
    # prompt:
    #
    # 小学校の算数のドリルを作ろうとしています。任意の数列の平均値を求める問題について、ドリルの文章のテンプレートを生成して、その数値部分を変数で置き換えてください。テンプレートは Python の文字列形式で出力してください。解説は不要で、テンプレートだけ10個出力してください。 # noqa: E501
    # 小学校の算数のドリルを作ろうとしています。任意の数列の最大値を求める問題について、ドリルの文章のテンプレートを生成して、その数値部分を変数で置き換えてください。テンプレートは Python の文字列形式で出力してください。解説は不要で、テンプレートだけ10個出力してください。 # noqa: E501
    template_question_attr = [
        "数列 {sequence} の{attr_label}を求めましょう。",
        "次の数列の{attr_label}を計算してください: {sequence}",
        "数列 {sequence} の{attr_label}はいくらになるでしょうか？",
        "数列 {sequence} の{attr_label}を答えましょう。",
        "次の数列の{attr_label}を求めてください: {sequence}",
        "数列 {sequence} の{attr_label}を計算しましょう。",
        "数列 {sequence} の{attr_label}を調べてください。",
        "数列 {sequence} の{attr_label}を算出しましょう。",
        "数列 {sequence} の{attr_label}を導き出しましょう。",
        "数列 {sequence} の{attr_label}を計算する問題です。",
        "数列 {sequence} の{attr_label}を求めよ。",
        "数列 {sequence} の{attr_label}を見つけよ。",
        "次の数列 {sequence} の{attr_label}を求め、その値を答えよ。",
        "数列 {sequence} の{attr_label}を答えよ。",
        "数列 {sequence} の{attr_label}を計算せよ。",
    ]
    # I could not find a suitable prompt to generate this templates with Nemotron-4-340B-Instruct.
    # These templates are generated by hand.
    template_answer_attr = [
        "{attr_label}: {attr_value}",
        "{attr_label}は {attr_value} です。",
        "{attr_label}は {attr_value} となります。",
        "数列 {sequence} の{attr_label}: {attr_value}",
        "数列 {sequence} の{attr_label}は {attr_value} です。",
        "数列 {sequence} の{attr_label}は {attr_value} となります。",
        "{sequence} の{attr_label}: {attr_value}",
        "{sequence} の{attr_label}は {attr_value} です。",
        "{sequence} の{attr_label}は {attr_value} となります。",
    ]
    attr_value = f"{attr_value:.2f}" if attr_value != int(attr_value) else str(int(attr_value))
    question = np.random.choice(template_question_attr).format(sequence=sequence, attr_label=attr_label)
    answer = np.random.choice(template_answer_attr).format(sequence=sequence, attr_label=attr_label, attr_value=attr_value)  # noqa: E501
    logger.debug(f"question: {question}")
    logger.debug(f"answer: {answer}")
    return json.dumps({"text": f"{QUESTION_ROLE}{question}{SEPARATOR}{ANSWER_ROLE}{answer}"}, ensure_ascii=False)


parser = argparse.ArgumentParser(description="Generate synth dataset.")
parser.add_argument("--num_samples", type=int, help="Number of samples.", default=1_000_000_000)
parser.add_argument("--outputs_jsonl", type=str, help="Output JSONL file.", default="data/outputs.jsonl")
parser.add_argument("--seed", type=int, help="Random seed.", default=None)
parser.add_argument("--log_interval", type=int, help="Log interval.", default=10_000)
args = parser.parse_args()

num_samples = args.num_samples
outputs_jsonl = args.outputs_jsonl
seed = args.seed
log_interval = args.log_interval

if seed >= 0:
    np.random.seed(seed)

with open(outputs_jsonl, "a") as f:
    for i in range(num_samples):
        arr = generate_random_array()
        logger.debug(f"a: {arr}")

        attr = get_attributes(arr)
        logger.debug(f"a_attr: {attr}")

        attr_key = np.random.choice(list(ATTRIBUTES.keys()))
        logger.debug(f"attr_key: {attr_key}")

        if attr_key in ["sorted_asc", "sorted_desc"]:
            sort_order = ATTRIBUTES[attr_key]
            sequence = ", ".join(map(str, arr))
            sequence_sorted = ", ".join(map(str, attr[attr_key]))
            qa = generate_qa_sort(sequence, sequence_sorted, sort_order)
        else:
            attr_label = ATTRIBUTES[attr_key]
            sequence = ", ".join(map(str, arr))
            attr_value = attr[attr_key]
            qa = generate_qa_attr(sequence, attr_label, attr_value)
        if i % log_interval == 0:
            logger.info(f"i: {i}, qa: {qa}")
        f.write(qa + "\n")
