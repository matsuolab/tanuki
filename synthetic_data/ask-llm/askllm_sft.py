# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Scoring SFT dataset by Ask-LLM.
#
# This script is a simplified script to run Ask-LLM for SFT dataset so that it can be easily modified.
# SFT datasets often contain complicated data structures, so I kept the code simple to make it easy to
# adapt to such structures. Instead, I removed the batch processing code, so this script is not suitable
# for large datasets.


import argparse
import json
# from logging import ERROR, StreamHandler, getLogger

import wandb
from datasets import load_dataset
from nano_askllm import AskLLM  # type: ignore
from transformers import AutoModelForCausalLM, AutoTokenizer
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


# Set logging level to ERROR.
# logger = getLogger("nano_askllm.askllm")
# logger.setLevel(ERROR)
# handler = StreamHandler()
# handler.setLevel(ERROR)
# logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Scoring by Ask-LLM.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--output_score_column", type=str, help="Output score column name.", default="askllm_score")
parser.add_argument("--max_tokens", type=int, help="Max tokens for Ask-LLM.", default=2048)
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--target_columns", type=str, help="Target dataset columns to score, separated by comma. e.g. instruction,input,output", default="messages")  # noqa: E501
parser.add_argument("--messages_indices", type=str, help="Messages indices to score, separated by comma. e.g. 0,1", default="0")  # noqa: E501
parser.add_argument("--separator", type=str, help="Separator to join texts.", default=" ")
parser.add_argument("--sort", type=bool, help="Sort by score.", default=True)
parser.add_argument("--reverse", type=bool, help="Reverse sort order.", default=True)
parser.add_argument("--model_id", type=str, help="Model ID.", default="cyberagent/calm3-22b-chat")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--log_interval", type=int, help="Log interval.", default=1000)
parser.add_argument("--wandb_project", type=str, help="WandB project name.", default=None)
parser.add_argument("--wandb_entity", type=str, help="WandB entity name.", default=None)
parser.add_argument("--wandb_name", type=str, help="WandB experiment name name.", default=None)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
output_score_column = args.output_score_column
max_tokens = args.max_tokens
dataset_path = args.dataset_path
dataset_name = args.dataset_name
dataset_split = args.dataset_split
target_columns = args.target_columns.split(",")
messages_indices = list(map(int, args.messages_indices.split(",")))
separator = args.separator.replace("\\n", "\n")
sort = args.sort
reverse = args.reverse
model_id = args.model_id
cache_dir = args.cache_dir
log_interval = args.log_interval
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
wandb_name = args.wandb_name

is_wandb = wandb_project is not None and wandb_entity is not None and wandb_name is not None

if is_wandb:
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_name)

tokenizer = AutoTokenizer.from_pretrained(model_id, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)
# tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto", cache_dir=cache_dir, attn_implementation="flash_attention_2",
)
# model.generation_config.pad_token_id = model.generation_config.eos_token_id


if input_jsonl is not None:
    dataset = load_dataset("json", data_files=input_jsonl, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
elif dataset_path is not None:
    dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
else:
    raise ValueError("You must specify either input_jsonl or dataset_path.")


prompt_template_prefix = "###\n"

prompt_template_postfix = """
###

Is it possible to solve this problem with the information provided?

OPTIONS: yes / no
ANSWER:"""  # noqa: E501

# yes_tokens = ["yes", "Yes"]         # for Rakuten/RakutenAI-7B-instruct
yes_tokens = ["yes", "Yes", "YES"]  # for microsoft/Phi-3-medium-128k-instruct
# yes_tokens = [" yes", " Yes"]         # for cyberagent/calm3-22b-chat

llm = AskLLM(
    tokenizer,
    model,
    prompt_template_prefix=prompt_template_prefix,
    prompt_template_postfix=prompt_template_postfix,
    yes_tokens=yes_tokens,
    max_tokens=max_tokens,
)

all_scores = []
for i in tqdm(range(len(dataset))):
    text = extract_text(dataset[i], target_columns, messages_indices, separator)
    scores = llm.ask([text])
    score = scores.tolist()[0]
    score = float(f"{score:.4f}")  # TODO: more digits?
    all_scores.append(score)
    if i % log_interval == 0:
        # print(f"step: {i}, score: {score}, text: {text}")
        if is_wandb:
            wandb.log({"score": score, "text": text}, step=i)
    del scores

dataset = dataset.add_column(output_score_column, all_scores)

if sort:
    dataset = dataset.sort(output_score_column, reverse=reverse)

with open(output_jsonl, "w") as f:
    for data in tqdm(dataset):
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


if is_wandb:
    wandb.finish()
