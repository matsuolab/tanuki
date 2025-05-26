# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from logging import DEBUG, StreamHandler, getLogger

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore
from vllm.model_executor.utils import set_random_seed  # type: ignore

# Set logging level to DEBUG.
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="Filter by regular expressions.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--model", type=str, help="Model name or path.", default="cyberagent/calm3-22b-chat")
parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path.", default="cyberagent/calm3-22b-chat")
parser.add_argument("--batch_size", type=int, help="Batch size.", default=128)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
dataset_path = args.dataset_path
dataset_name = args.dataset_name
dataset_split = args.dataset_split
cache_dir = args.cache_dir
model = args.model
tokenizer = args.tokenizer
batch_size = args.batch_size

if input_jsonl is not None:
    dataset = load_dataset("json", data_files=input_jsonl, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
elif dataset_path is not None:
    dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
else:
    raise ValueError("You must specify either input_jsonl or dataset_path.")

set_random_seed(1)

vllm = LLM(model, tokenizer=tokenizer)
tokenizer = vllm.get_tokenizer()

end = len(dataset)
start = 0

with open(output_jsonl, "w", encoding="utf-8") as f:
    for i in tqdm(range(start, end, batch_size)):
        batch_start = i
        batch_end = min(i + batch_size, end)
        data = dataset[batch_start:batch_end]
        # data is column-oriented data, so convert it to row-oriented data
        batch = pd.DataFrame(data).to_dict(orient="records")
        if len(batch) == 0:
            break
        questions = [b["messages"][0]["content"] for b in batch]
        messages_list = [[{"role": "user", "content": question}] for question in questions]
        inputs = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        # logger.debug(f"inputs: {inputs}")
        stop = ["<EOD>", "</s>"]
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=512, seed=1, repetition_penalty=1.1, stop=stop, n=2, best_of=4
        )
        outputs = vllm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        logger.debug(f"len(outputs): {len(outputs)}")
        logger.debug(f"len(outputs[0].outputs): {len(outputs[0].outputs)}")
        # logger.debug(f"outputs: {outputs}")
        assert len(outputs) == len(batch)
        for j, output in enumerate(outputs):
            assert len(output.outputs) == 2
            answer_1 = output.outputs[0].text
            answer_2 = output.outputs[1].text
            f.write(
                json.dumps({"question": questions[j], "answer_1": answer_1, "answer_2": answer_2}, ensure_ascii=False)
                + "\n"
            )
        f.flush()
