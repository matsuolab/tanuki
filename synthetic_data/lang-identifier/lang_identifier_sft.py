# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-FileCopyrightText: 2024 Yuta Oriike
# SPDX-License-Identifier: Apache-2.0


# Language identification by fastText
#
# - Download `model.bin` from huggingface.
#
# ```
# pip install huggingface_hub[hf_transfer]
# HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
#   facebook/fasttext-language-identification model.bin  \
#   --repo-type=model --local-dir="." --local-dir-use-symlinks=False
# ```
#
# - Run the script.
#
# ```
# python scripts/lang_identifier.py \
#   --input_jsonl "data/input.jsonl" \
#   --output_jsonl "data/output.jsonl" \
#   --fasttext_path "model.bin"
# ```
#
# - The output jsonl file will contain the score column.
#
# ```
# {"messages": [{"role": "user", "content": "こんにちは。"}], "fasttext_jp_score": 0.9999}
# ```

import argparse
import json

import fasttext
from datasets import load_dataset
from tqdm import tqdm


def extract_text(data, target_columns, messages_indices, separator):
    """Extract text from data.
    If the target column is "messages", it extracts the content of the specified messages_indices.
    Otherwise, it extracts the column text as it is.
    Texts are joined by the separator.
    """
    texts = []
    for column in target_columns:
        if column == "messages":
            texts.extend([data[column][messages_index]["content"] for messages_index in messages_indices])
        else:
            texts.append(data[column])
    return separator.join(texts)


def get_lang_score(ft_model, text, lang_label="__label__jpn_Jpan", top_k=30):
    text = text.replace("\n", " ")  # TODO: are there any other special characters that need to be removed?
    scores = [score for label, score in zip(*ft_model.predict(text, k=top_k)) if label == lang_label]
    return scores[0] if len(scores) > 0 else 0.0


parser = argparse.ArgumentParser(description="Language identification by fastText.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--trash_jsonl", type=str, help="Output jsonl file for trash data (score < threshold).", default="trash.jsonl")  # noqa: E501
parser.add_argument("--output_score_column", type=str, help="Output score column name.", default="fasttext_jp_score")
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--fasttext_path", type=str, help="fastText model path.", default="model.bin")
parser.add_argument("--fasttext_label", type=str, help="fastText language label. e.g. __label__jpn_Jpan", default="__label__jpn_Jpan")  # noqa: E501
parser.add_argument("--target_columns", type=str, help="Target dataset columns to score, separated by comma. e.g. instruction,input,output", default="messages")  # noqa: E501
parser.add_argument("--messages_indices", type=str, help="Messages indices to score, separated by comma. e.g. 0,1", default="0")  # noqa: E501
parser.add_argument("--separator", type=str, help="Separator to join texts.", default=" ")
parser.add_argument("--sort", type=bool, help="Sort by score.", default=True)
parser.add_argument("--reverse", type=bool, help="Reverse sort order.", default=True)
parser.add_argument("--threshold", type=float, help="Threshold to filter scores.", default=0.0)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
trash_jsonl = args.trash_jsonl
output_score_column = args.output_score_column
dataset_path = args.dataset_path
dataset_name = args.dataset_name
dataset_split = args.dataset_split
cache_dir = args.cache_dir
fasttext_path = args.fasttext_path
fasttext_label = args.fasttext_label
target_columns = args.target_columns.split(",")
messages_indices = list(map(int, args.messages_indices.split(",")))
separator = args.separator.replace("\\n", "\n")
sort = args.sort
reverse = args.reverse
threshold = args.threshold

if input_jsonl is not None:
    dataset = load_dataset("json", data_files=input_jsonl, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
elif dataset_path is not None:
    dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
else:
    raise ValueError("You must specify either input_jsonl or dataset_path.")

ft_model = fasttext.load_model(fasttext_path)

all_scores = []
for data in tqdm(dataset):
    text = extract_text(data, target_columns, messages_indices, separator)
    score = get_lang_score(ft_model, text, lang_label=fasttext_label, top_k=30)
    score = float(f"{score:.4f}")  # TODO: more digits?
    all_scores.append(score)

dataset = dataset.add_column(output_score_column, all_scores)

if sort:
    dataset = dataset.sort(output_score_column, reverse=reverse)

if threshold > 0.0:
    with open(output_jsonl, "w") as f_output, open(trash_jsonl, "w") as f_trash:
        for data in tqdm(dataset):
            f = f_trash if data[output_score_column] < threshold else f_output
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
else:
    with open(output_jsonl, "w") as f:
        for data in tqdm(dataset):
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
