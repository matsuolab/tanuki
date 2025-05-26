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
            content = re.sub(r"(はい、)?了解しました。", "", content, flags=re.DOTALL)
            content = re.sub(r"(はい、)?もちろんです。", "", content, flags=re.DOTALL)
            content = re.sub(r"それでは、以下のような質問を設定してみます。\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"それでは、以下のような質問を追加で行ってください。\n\n---\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"それでは、以下のような追加の質問を作成いたします。\n\n---\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下のよう(に|な)質問を行うと良いでしょう[：。]\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下に追加の質問を提示します。\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下に追加の質問を示します。\n\n---\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下の追加質問をご覧ください：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下に質問を作成いたします。\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下の追加質問を作成します：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下のような(追加の|追加)?質問はいかがでしょうか[：？]\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下のような追加の質問を考えま(す|した)：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下の追加質問を検討してください：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"以下の変更を加えた追加の質問を考えてみました：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"(では、)?以下に簡潔な追加(の)?質問を(作成|示)します[：。]\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"(では、)?以下に具体的な問題を記述した追加の質問を作成します。\n\n---\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"(では、)?追加の質問として以下のようなものはどうでしょうか[：？]\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"この問題に関連して、次のような追加質問を作ることができます：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"追加の質問を作成しました：\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"はい、理解しました。追加の質問を検討いたします。\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"はい、理解しました。この場合について説明します。\n\n", "", content, flags=re.DOTALL)
            content = re.sub(r"^「(.+)」$", "\\1", content, flags=re.DOTALL)
            message["content"] = content.strip()
            messages.append(message)
        data["messages"] = messages
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
