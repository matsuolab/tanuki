# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Convert SFT dataset to pre-training dataset.


import argparse
import json

from datasets import load_dataset
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Scoring by Ask-LLM.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output1_jsonl", type=str, help="Output jsonl file 1.", default="output1.jsonl")
parser.add_argument("--output2_jsonl", type=str, help="Output jsonl file2 .", default="output2.jsonl")
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output1_jsonl = args.output1_jsonl
output2_jsonl = args.output2_jsonl
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

task1 = ["math", "arithmetic", "basic math", "basic arithmetic", "数学", "算数"]

task2 = [
    "logical reasoning",
    "reasoning",
    "reasoning quiz",
    "reasoning game",
    "logic quiz",
    "logic game",
    "論理推論",
    "論理クイズ",
    "論理ゲーム",
    "推理",
    "推理クイズ",
    "推理ゲーム",
]

with open(output1_jsonl, "w", encoding="utf-8") as f1, open(output2_jsonl, "w", encoding="utf-8") as f2:
    for data in tqdm(dataset):
        if data["task"] in task1:
            f1.write(json.dumps(data, ensure_ascii=False) + "\n")
        elif data["task"] in task2:
            f2.write(json.dumps(data, ensure_ascii=False) + "\n")
        else:
            raise ValueError(f"Unknown task: {data['task']}")
