# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

import argparse
from datasets import load_dataset
from tqdm import tqdm
import re
import json


parser = argparse.ArgumentParser(description="Filter by regular expressions.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
dataset_path = args.dataset_path
dataset_name = args.dataset_name
dataset_split = args.dataset_split
cache_dir = args.cache_dir


if input_jsonl is not None:
    dataset = load_dataset("json", data_files=input_jsonl, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
elif dataset_path is not None:
    dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
else:
    raise ValueError("You must specify either input_jsonl or dataset_path.")

with open(args.output_jsonl, "w") as f:
    for data in tqdm(dataset):
        messages = []
        for message in data["messages"]:
            content = message["content"]
            # delete frequent expressions
            content = re.sub(r"もちろん、喜んでお手伝いします！", "", content, flags=re.DOTALL)
            content = re.sub(r"もちろん、日本語で回答いたします。", "", content, flags=re.DOTALL)
            content = re.sub(r"もちろん、日本語で回答します。", "", content, flags=re.DOTALL)
            content = re.sub(r"もちろん、お答えします。", "", content, flags=re.DOTALL)
            content = re.sub(r"もちろん、説明します。", "", content, flags=re.DOTALL)
            content = re.sub(r"もちろんです[、。]", "", content, flags=re.DOTALL)
            content = re.sub(r"日本語で回答しました。", "", content, flags=re.DOTALL)
            content = re.sub(r"日本語で回答しています。", "", content, flags=re.DOTALL)
            content = re.sub(r"（日本語で回答しました。?）", "", content, flags=re.DOTALL)
            content = re.sub(r"（回答は日本語で記載しています。）", "", content, flags=re.DOTALL)
            content = re.sub(r"（回答は日本語で記[述入]してください）", "", content, flags=re.DOTALL)
            content = re.sub(r"日本語で回答しなさい。", "", content, flags=re.DOTALL)
            content = re.sub(r"ただし、問題文は日本語で書かれています。", "", content, flags=re.DOTALL)
            content = re.sub(r"（解答は書きません）", "", content, flags=re.DOTALL)
            content = re.sub(r"（注意：解答は含まれていません。）", "", content, flags=re.DOTALL)
            content = re.sub(r"（解答はここには書きません）", "", content, flags=re.DOTALL)
            content = re.sub(r"（解答は含めないでください）", "", content, flags=re.DOTALL)
            content = re.sub(r"（解答はここには記載しません）", "", content, flags=re.DOTALL)
            content = re.sub(r"（この問題の解答は含まれていません）", "", content, flags=re.DOTALL)
            content = re.sub(r"Sure, I'd be happy to help you with that[\!\.]", "", content, flags=re.DOTALL)
            content = re.sub(r"Sure, I'd be happy to help you with that calculation\.", "", content, flags=re.DOTALL)
            content = re.sub(r"Sure, I'd be happy to help you with that follow-up question\.", "", content, flags=re.DOTALL)  # noqa: E501
            content = re.sub(r"Sure, I'd be happy to help with your question!", "", content, flags=re.DOTALL)
            content = re.sub(r"Sure, I'd be happy to help!", "", content, flags=re.DOTALL)
            content = re.sub(r"Sure, I can help you with that[\!\.]", "", content, flags=re.DOTALL)
            # delete translation
            content = re.sub(r"\n\n[Tt]ranslation:\n\n.+", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"\n\n\s?Topic description translation[：:].+", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"\n\n\s?Note: The translation.+", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"\n\n[\(（]?([Tt]he |[Cc]orrect |[Pp]ossible )?[Aa]nswer.+", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"\n\n[\(（]?[Nn]ote: [Tt]he( [Cc]orrect) [Aa]nswer.+", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"\n\n[\(（]?解答[：:].+", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"\n\n[\(（](この問題の)?解答は、.+", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"\n\n[\(（]?[Ee]xplanation[：:].+", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"[\(（]\s?[Tt]ranslation( for reference| for your reference)?[：:].+[\)）]?", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]\s?([Tt]ranslating|[Tt]ranslation).+[\)）]?", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]\s?([Tt]raduction|[Tt]raducción|[Tt]raduzione)\s?( al español)?[：:].+[\)）]?", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]\s?[Tt]ranslate(d to English|s to)?[：:].+?[\)）]", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]\s?[Aa]nswer in English[：:].+?[\)）]", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]Note[：:] (The|This)( original)? (translations?|problems?|questions?|answers?).+[\)）]?", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（][Tt]ranslating the( above)? (problem|answer) into English.+$", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（]English translation.+?[\)）]", "", content, flags=re.DOTALL)  # noqa: E501  # cspell: disable-line
            content = re.sub(r"[\(（](英語翻|ヒントの|翻)?訳[：:].+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（]英語訳[：:].+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（]回答を英語に翻訳すると[：:].+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（]日本語で回答[：:].+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（]回答は日本語で.+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（]注：日本語で.+?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（](Note|注意|注)[：:]\s?(この)?(問題|解答|回答)は、?日本語.+?[\)）]", "", content, flags=re.DOTALL)
            message["content"] = content.strip()
            messages.append(message)
        data["messages"] = messages
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
