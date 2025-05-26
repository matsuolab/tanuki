# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Generate multi-turn instruction dataset using topics.

# Get an API key from NVIDIA page.
# Press "Get API Key" button on the following page.
# https://build.nvidia.com/nvidia/nemotron-4-340b-instruct

# Set the API key as an environment variable.
# export NVIDIA_API_KEY="nvapi-..."


import argparse
import json
import os
import random
import re
from logging import DEBUG, StreamHandler, getLogger
from time import time

from openai import OpenAI

# Set logging level to DEBUG.
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


PROMPT_TEMPLATE = """Create a {task} problem related to the following {topic}:

{item}

Note:

1. The {task} problem should be simple and involve basic {task} skills and knowledge. All average grade school students can solve it correctly.
2. The {task} problem should include equations rather than text sentences.
3. The {task} problem should be short and clear.
4. Your response should always start with "Problem:". Your response should not include a solution to the created {task} problem.
5. 日本語で回答しなさい。
"""  # noqa: E501


class LLM:
    def __init__(self, base_url, api_key, model, temperature=0.7, max_tokens=512, seed=None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def __call__(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return {
            "content": completion.choices[0].message.content,
            "estimated_cost": completion.usage.estimated_cost if hasattr(completion.usage, "estimated_cost") else 0.0,
        }


def generate_question_1(llm, task, topic, item):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": PROMPT_TEMPLATE.format(task=task.strip(), topic=topic.strip(), item=item.strip())},
    ]
    result = llm(messages)
    content = result["content"]
    content = content.strip()
    content = re.sub(r"^Problem[：:]", "", content, flags=re.DOTALL)
    content = re.sub(r"^問題[：:]", "", content, flags=re.DOTALL)
    content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
    content = re.sub(r"[\(（]翻?訳[：:].*?[\)）]", "", content, flags=re.DOTALL)
    content = re.sub(r"[\(（][Aa]nswer:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
    content = re.sub(r"[\(（]答え.*?[\)）]", "", content, flags=re.DOTALL)
    content = re.sub(r"\n\n答え[：:].*?", "", content, flags=re.DOTALL)
    content = content.strip()
    return {"content": content, "estimated_cost": result["estimated_cost"]}


def generate_answer_1(llm, question_1):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": f"{question_1}\n\n日本語で回答しなさい。"},
    ]
    result = llm(messages)
    content = result["content"]
    content = content.strip()
    content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
    content = re.sub(r"日本語で答えます。", "", content, flags=re.DOTALL)
    content = content.strip()
    return {"content": content, "estimated_cost": result["estimated_cost"]}


def generate_question_2(llm, question_1, answer_1):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": question_1},
        {"role": "assistant", "content": answer_1},
        {
            "role": "user",
            "content": "前述の問題をより理解するために、簡潔な追加の質問を一つ作りなさい。問題の一部を変更したり、条件を追加しても良いです。追加の質問だけを書き、決して答えを含めないでください。",
        },  # noqa: E501
    ]
    result = llm(messages)
    content = result["content"]
    content = content.strip()
    content = re.sub(r"^追加の質問[：:]", "", content, flags=re.DOTALL)
    content = re.sub(r"^質問[：:]", "", content, flags=re.DOTALL)
    content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
    content = re.sub(r"^「(.+)」$", "\\1", content, flags=re.DOTALL)
    content = content.strip()
    return {"content": content, "estimated_cost": result["estimated_cost"]}


def generate_answer_2(llm, question_1, answer_1, question_2):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": question_1},
        {"role": "assistant", "content": answer_1},
        {"role": "user", "content": question_2},
    ]
    result = llm(messages)
    content = result["content"]
    content = content.strip()
    content = re.sub(r"[\(（][Tt]ranslation:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
    content = content.strip()
    return {"content": content, "estimated_cost": result["estimated_cost"]}


def generate_instruction_data(llm, task, topic, item):
    estimated_cost = 0.0
    logger.debug(f"- タスク\n```{task}```")
    logger.debug(f"- {topic}\n```{item}```")
    q1 = generate_question_1(llm, task, topic, item)
    question_1 = q1["content"]
    estimated_cost += q1["estimated_cost"]
    logger.debug(f"- 問題1\n```{question_1}```")
    a1 = generate_answer_1(llm, question_1)
    answer_1 = a1["content"]
    estimated_cost += a1["estimated_cost"]
    logger.debug(f"- 解答1\n```{answer_1}```")
    # q2 = generate_question_2(llm, question_1, answer_1)
    # question_2 = q2["content"]
    # estimated_cost += q2["estimated_cost"]
    # logger.debug(f"- 問題2\n```{question_2}```")
    # a2 = generate_answer_2(llm, question_1, answer_1, question_2)
    # answer_2 = a2["content"]
    # estimated_cost += a2["estimated_cost"]
    # logger.debug(f"- 解答2\n```{answer_2}```")
    return {
        "messages": [
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            # {"role": "user", "content": question_2},
            # {"role": "assistant", "content": answer_2},
        ],
        "task": task,
        topic: item,
        "estimated_cost": float(f"{estimated_cost:.7f}"),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synth dataset.")
    parser.add_argument("--task_jsonl", type=str, help="Task JSONL file.", default="data/math_task.jsonl")
    parser.add_argument("--topic_jsonl", type=str, help="Topic JSONL file.", default="data/math_topic.jsonl")
    parser.add_argument("--outputs_jsonl", type=str, help="Output JSONL file.", default="data/outputs.jsonl")
    parser.add_argument("--base_url", type=str, help="API base URL.", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("NVIDIA_API_KEY"))
    parser.add_argument("--model", type=str, help="Model name.", default="nvidia/nemotron-4-340b-instruct")
    # parser.add_argument("--base_url", type=str, help="API base URL.", default="https://api.deepinfra.com/v1/openai")
    # parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("DEEPINFRA_API_KEY"))
    # parser.add_argument("--model", type=str, help="Model name.", default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--num_samples", type=int, help="Number of samples.", default=1)
    parser.add_argument("--temperature", type=float, help="Temperature for inference.", default=0.7)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for inference.", default=512)
    parser.add_argument("--seed", type=int, help="Random seed.", default=None)

    args = parser.parse_args()

    task_jsonl = args.task_jsonl
    topic_jsonl = args.topic_jsonl
    outputs_jsonl = args.outputs_jsonl
    base_url = args.base_url
    api_key = args.api_key
    model = args.model
    num_samples = args.num_samples
    temperature = args.temperature
    max_tokens = args.max_tokens
    seed = args.seed

    # base_url = "https://integrate.api.nvidia.com/v1"
    # api_key = os.getenv("NVIDIA_API_KEY")
    # model = "nvidia/nemotron-4-340b-instruct"

    # base_url = "https://api.deepinfra.com/v1/openai"
    # api_key = os.getenv("DEEPINFRA_API_KEY")
    # model = "meta-llama/Meta-Llama-3.1-405B-Instruct"

    if seed < 0:
        seed = None
    random.seed(seed)

    llm = LLM(base_url, api_key, model, temperature=temperature, max_tokens=max_tokens, seed=seed)

    with open(task_jsonl, "r") as f:
        tasks = [json.loads(line)["task"] for line in f]

    with open(topic_jsonl, "r") as f:
        topics = [json.loads(line) for line in f]

    start = time()
    estimated_cost = 0.0

    for i in range(num_samples):
        task = random.choice(tasks)
        topic = random.choice(topics)
        key = list(topic.keys())[0]
        item = topic[key]
        result = generate_instruction_data(llm, task, key, item)
        estimated_cost += result["estimated_cost"]
        with open(outputs_jsonl, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        elapsed = time() - start
        logger.debug(f"Elapsed time (sec): {elapsed:.2f}")
        logger.debug(f"Estimated cost (USD): {estimated_cost:.7f}")


if __name__ == "__main__":
    main()
