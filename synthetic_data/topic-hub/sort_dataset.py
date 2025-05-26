# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0


import argparse
import json

from datasets import load_dataset
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Sorting dataset.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default="input.jsonl")
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--column", type=str, help="Sort column name.", default="score")
parser.add_argument("--reverse", type=bool, help="Reverse sort order.", default=True)
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
column = args.column
reverse = args.reverse
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


dataset = dataset.sort(column, reverse=reverse)

with open(output_jsonl, "w") as f:
    for data in tqdm(dataset):
        f.write(json.dumps(data, ensure_ascii=False) + "\n")
