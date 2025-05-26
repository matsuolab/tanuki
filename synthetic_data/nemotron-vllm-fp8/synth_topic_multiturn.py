# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Generate multi-turn instruction dataset using topics.

# Get an API key from NVIDIA page.
# Press "Get API Key" button on the following page.
# https://build.nvidia.com/nvidia/nemotron-4-340b-instruct

# Set the API key as an environment variable.
# export NVIDIA_API_KEY="nvapi-..."


from abc import ABC, abstractmethod
import argparse
import json
# import os
import random
import re
from logging import DEBUG, StreamHandler, getLogger
from time import time

from openai import OpenAI
from vllm import LLM, SamplingParams  # type: ignore
from vllm.model_executor.utils import set_random_seed  # type: ignore

# Set logging level to DEBUG.
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


class LanguageModel(ABC):
    Q1_PROMPT_TEMPLATE = """Create a {task} problem related to the following {topic}:

{item}

Note:

1. The {task} problem should be simple and involve basic {task} skills and knowledge. Any average grade school student can solve it correctly.
2. You should make full use of the {topic} description to create the {task} problem to ensure that the {task} problem is unique and specific to the {topic}.
3. Your response should always start with "問題:". Your response should not include a solution to the created {task} problem.
4. 簡潔に日本語で回答してください。
"""  # noqa: E501

    A1_PROMPT_TEMPLATE = "{question_1}\n\n簡潔に日本語で回答してください。"

    Q2_PROMPT_TEMPLATE = "前述の問題をより理解するために、簡潔な追加の質問を一つ作りなさい。問題の一部を変更したり、条件を追加しても良いです。追加の質問だけを書き、決して答えを含めないでください。"  # noqa: E501

    A2_PROMPT_TEMPLATE = "{question_2}\n\n簡潔に日本語で回答してください。"

    def __init__(self, model_name_or_path, temperature=0.7, max_tokens=512, seed=None):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        logger.debug(f"model_name_or_path: {model_name_or_path}, temperature: {temperature}, max_tokens: {max_tokens}, seed: {seed}")  # noqa: E501

    @abstractmethod
    def __call__(self, messages_list):
        pass

    def get_question_1_messages(self, task, topic, item):
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": self.Q1_PROMPT_TEMPLATE.format(task=task.strip(), topic=topic.strip(), item=item.strip())},  # noqa: E501
        ]

    def get_answer_1_messages(self, question_1):
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": self.A1_PROMPT_TEMPLATE.format(question_1=question_1.strip())},
        ]

    def get_question_2_messages(self, question_1, answer_1):
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": self.Q2_PROMPT_TEMPLATE},
        ]

    def get_answer_2_messages(self, question_1, answer_1, question_2):
        return [
            {"role": "system", "content": ""},
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": self.A2_PROMPT_TEMPLATE.format(question_2=question_2)},
        ]


class NemotronVLLM(LanguageModel):
    # default apply_chat_template looks broken, so I implemented it manually.
    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        if tokenize:
            raise NotImplementedError("tokenize=True is not supported.")
        prompts = []
        for m in messages:
            if m["role"] == "system":
                prompts.append(f"<extra_id_0>System\n{m['content']}\n")
            elif m["role"] == "user":
                prompts.append(f"<extra_id_1>User\n{m['content']}\n")
            elif m["role"] == "assistant":
                prompts.append(f"<extra_id_1>Assistant\n{m['content']}\n")
            else:
                raise ValueError(f"Unknown role: {m['role']}")
        if add_generation_prompt:
            if messages[-1]["role"] == "system":
                prompts.append("<extra_id_1>User\n")
            elif messages[-1]["role"] == "user":
                prompts.append("<extra_id_1>Assistant\n")
            elif messages[-1]["role"] == "assistant":
                prompts.append("<extra_id_1>User\n")
            else:
                raise ValueError(f"Unknown role: {messages[-1]['role']}")
        return "".join(prompts)

    def __init__(self, model_name_or_path, temperature=0.7, max_tokens=512, seed=None):
        super().__init__(model_name_or_path, temperature=temperature, max_tokens=max_tokens, seed=seed)
        # TODO: test enforce_eager=False and see the performance and stability.
        self.vllm = LLM(model_name_or_path, tensor_parallel_size=8, distributed_executor_backend="ray", enforce_eager=True)  # noqa: E501
        stop = ["<|endoftext|>", "<extra_id_0>", "<extra_id_1>", "\x11"]
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed, stop=stop)
        logger.debug(f"sampling_params: {self.sampling_params}")

    def __call__(self, messages_list):
        prompts = [self.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]  # noqa: E501
        outputs = self.vllm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        logger.debug(f"len(outputs): {len(outputs)}")
        contents = [o.outputs[0].text for o in outputs]
        return {"contents": contents, "estimated_cost": 0.0}


class Calm3VLLM(LanguageModel):
    CALM3_Q1_PROMPT_TEMPLATE = """Create a {task} problem related to the following {topic}:

{item}

Note:

1. The {task} problem should be simple and involve basic {task} skills and knowledge. Any average grade school student can solve it correctly.
2. You should make full use of the {topic} description to create the {task} problem to ensure that the {task} problem is unique and specific to the {topic}.
3. Your response should always start with "問題:". Your response should not include a solution to the created {task} problem.
4. 簡潔に日本語で回答してください。
"""  # noqa: E501

    CALM3_A1_PROMPT_TEMPLATE = "{question_1}\n\n簡潔に日本語で回答してください。"

    CALM3_Q2_PROMPT_TEMPLATE = "前述の問題をより理解するために、簡潔な追加の質問を一つ作ってください。問題の一部を変更したり、条件を追加しても良いです。追加の質問だけを書き、決して答えを含めないでください。"  # noqa: E501

    CALM3_A2_PROMPT_TEMPLATE = "{question_2}\n\n簡潔に日本語で回答してください。"

    def __init__(self, model_name_or_path, temperature=0.7, max_tokens=512, seed=None):
        super().__init__(model_name_or_path, temperature=temperature, max_tokens=max_tokens, seed=seed)
        # TODO: test enforce_eager=True and see the performance and stability.
        self.vllm = LLM(model_name_or_path)
        stop = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, seed=seed, stop=stop, repetition_penalty=1.1)  # noqa: E501
        logger.debug(f"sampling_params: {self.sampling_params}")

    def __call__(self, messages_list):
        tokenizer = self.vllm.get_tokenizer()
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_list]  # noqa: E501
        # logger.debug(f"prompts[0]: {prompts[0]}")
        outputs = self.vllm.generate(prompts, sampling_params=self.sampling_params, use_tqdm=False)
        # logger.debug(f"outputs[0].outputs[0].text: {outputs[0].outputs[0].text}")
        contents = [o.outputs[0].text for o in outputs]
        return {"contents": contents, "estimated_cost": 0.0}

    def get_question_1_messages(self, task, topic, item):
        return [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": self.CALM3_Q1_PROMPT_TEMPLATE.format(task=task.strip(), topic=topic.strip(), item=item.strip())},  # noqa: E501
        ]

    def get_answer_1_messages(self, question_1):
        return [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": self.CALM3_A1_PROMPT_TEMPLATE.format(question_1=question_1.strip())},
        ]

    def get_question_2_messages(self, question_1, answer_1):
        return [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": self.CALM3_Q2_PROMPT_TEMPLATE},
        ]

    def get_answer_2_messages(self, question_1, answer_1, question_2):
        return [
            {"role": "system", "content": "あなたは親切なAIアシスタントです。日本語で回答してください。"},
            {"role": "user", "content": question_1},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": self.CALM3_A2_PROMPT_TEMPLATE.format(question_2=question_2)},
        ]


class OpenAIAPI(LanguageModel):
    def __init__(self, model_name_or_path, base_url=None, api_key=None, temperature=0.7, max_tokens=512, seed=None):
        super().__init__(model_name_or_path, temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.base_url = base_url
        self.api_key = api_key
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        logger.debug(f"base_url: {base_url}, api_key: {api_key[:10]}")

    def __call__(self, messages_list):
        contents = []
        estimated_cost = 0.0
        for messages in messages_list:
            completion = self.client.chat.completions.create(
                model=self.model_name_or_path,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                seed=self.seed,
            )
            contents.append(completion.choices[0].message.content)
            estimated_cost += completion.usage.estimated_cost if hasattr(completion.usage, "estimated_cost") else 0.0
        return {"contents": contents, "estimated_cost": estimated_cost}


def generate_question_1(llm, tasks, topics, items):
    response = llm([llm.get_question_1_messages(task, topic, item) for task, topic, item in zip(tasks, topics, items)])  # noqa: E501
    contents = []
    for content in response["contents"]:
        content = content.strip()
        content = re.sub(r"^Problem[：:]", "", content, flags=re.DOTALL)
        content = re.sub(r"^問題[：:]", "", content, flags=re.DOTALL)
        content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
        content = re.sub(r"[\(（]翻?訳[：:].*?[\)）]", "", content, flags=re.DOTALL)
        content = re.sub(r"[\(（][Aa]nswer:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
        content = re.sub(r"[\(（]答え.*?[\)）]", "", content, flags=re.DOTALL)
        content = re.sub(r"\n\n答え[：:].*?", "", content, flags=re.DOTALL)
        content = content.strip()
        contents.append(content)
    return {"contents": contents, "estimated_cost": response["estimated_cost"]}


def generate_answer_1(llm, questions_1):
    response = llm([llm.get_answer_1_messages(q1) for q1 in questions_1])
    contents = []
    for content in response["contents"]:
        content = content.strip()
        content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
        content = content.strip()
        contents.append(content)
    return {"contents": contents, "estimated_cost": response["estimated_cost"]}


def generate_question_2(llm, questions_1, answers_1):
    response = llm([llm.get_question_2_messages(q1, a1) for q1, a1 in zip(questions_1, answers_1)])
    contents = []
    for content in response["contents"]:
        content = content.strip()
        content = re.sub(r"^追加の質問[：:]", "", content, flags=re.DOTALL)
        content = re.sub(r"^質問[：:]", "", content, flags=re.DOTALL)
        content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
        content = re.sub(r"^「(.+)」$", "\\1", content, flags=re.DOTALL)
        content = content.strip()
        contents.append(content)
    return {"contents": contents, "estimated_cost": response["estimated_cost"]}


def generate_answer_2(llm, questions_1, answers_1, questions_2):
    response = llm([llm.get_answer_2_messages(q1, a1, q2) for q1, a1, q2 in zip(questions_1, answers_1, questions_2)])
    contents = []
    for content in response["contents"]:
        content = content.strip()
        content = re.sub(r"[\(（][Tt]ranslation:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
        content = content.strip()
        contents.append(content)
    return {"contents": contents, "estimated_cost": response["estimated_cost"]}


def main():
    parser = argparse.ArgumentParser(description="Generate synth dataset.")
    parser.add_argument("--task_jsonl", type=str, help="Task JSONL file.", default="data/math_task.jsonl")
    parser.add_argument("--topic_jsonl", type=str, help="Topic JSONL file.", default="data/math_topic.jsonl")
    parser.add_argument("--outputs_jsonl", type=str, help="Output JSONL file.", default="data/outputs.jsonl")
    parser.add_argument("--base_url", type=str, help="API base URL.", default="")
    parser.add_argument("--api_key", type=str, help="API key.", default="")
    parser.add_argument("--model", type=str, help="Model name.", default="mgoin/Nemotron-4-340B-Instruct-hf-FP8")
    # parser.add_argument("--base_url", type=str, help="API base URL.", default="https://integrate.api.nvidia.com/v1")
    # parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("NVIDIA_API_KEY"))
    # parser.add_argument("--model", type=str, help="Model name.", default="nvidia/nemotron-4-340b-instruct")
    # parser.add_argument("--base_url", type=str, help="API base URL.", default="https://api.deepinfra.com/v1/openai")
    # parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("DEEPINFRA_API_KEY"))
    # parser.add_argument("--model", type=str, help="Model name.", default="meta-llama/Meta-Llama-3.1-405B-Instruct")
    parser.add_argument("--num_samples", type=int, help="Number of samples.", default=1)
    parser.add_argument("--batch_size", type=int, help="Size of batch.", default=128)
    parser.add_argument("--temperature", type=float, help="Temperature for inference.", default=0.7)
    parser.add_argument("--max_tokens", type=int, help="Max tokens for inference.", default=512)
    parser.add_argument("--seed", type=int, help="Random seed.", default=1)

    args = parser.parse_args()

    task_jsonl = args.task_jsonl
    topic_jsonl = args.topic_jsonl
    outputs_jsonl = args.outputs_jsonl
    base_url = args.base_url if args.base_url.strip() != "" else None
    api_key = args.api_key if args.api_key.strip() != "" else None
    model = args.model
    num_samples = args.num_samples
    batch_size = args.batch_size
    temperature = args.temperature
    max_tokens = args.max_tokens
    seed = 0 if args.seed < 0 else args.seed  # TODO: generate random seed

    set_random_seed(seed)
    logger.debug(f"seed: {seed}")

    if base_url is None and api_key is None:
        if "nemotron" in model.lower():
            llm = NemotronVLLM(model, temperature=temperature, max_tokens=max_tokens, seed=seed)
        elif "calm3" in model.lower():
            llm = Calm3VLLM(model, temperature=temperature, max_tokens=max_tokens, seed=seed)
        else:
            raise ValueError(f"Unknown model: {model}")
    else:
        llm = OpenAIAPI(model, base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens, seed=seed)  # noqa: E501

    with open(task_jsonl, "r") as f:
        TASK_LIST = [json.loads(line)["task"] for line in f]

    with open(topic_jsonl, "r") as f:
        TOPIC_LIST = [json.loads(line) for line in f]

    loop_start = time()
    estimated_cost = 0.0

    logger.debug(f"batch_size: {batch_size}")
    for i in range(num_samples):
        set_random_seed(seed=(seed+i))
        logger.debug(f"seed: {(seed+i)}")
        tasks = []
        topics = []
        items = []
        for j in range(batch_size):
            task = random.choice(TASK_LIST)
            tasks.append(task)
            topic = random.choice(TOPIC_LIST)
            key = list(topic.keys())[0]
            item = topic[key]
            topics.append(key)
            items.append(item)
        assert len(tasks) == len(topics) == len(items) == batch_size
        logger.debug(f"len(tasks) == {len(tasks)}, len(topics) == {len(topics)}, len(items) == {len(items)}, batch_size == {batch_size}")  # noqa: E501
        start = time()
        questions_1 = generate_question_1(llm, tasks, topics, items)
        estimated_cost += questions_1["estimated_cost"]
        end = time()
        logger.debug(f"generate_question_1: time == {end - start}, len(questions_1) == {len(questions_1['contents'])}")
        start = end
        answers_1 = generate_answer_1(llm, questions_1["contents"])
        estimated_cost += answers_1["estimated_cost"]
        end = time()
        logger.debug(f"generate_answer_1: time == {end - start}, len(answers_1) == {len(answers_1['contents'])}")
        start = end
        questions_2 = generate_question_2(llm, questions_1["contents"], answers_1["contents"])
        estimated_cost += questions_2["estimated_cost"]
        end = time()
        logger.debug(f"generate_question_2: time == {end - start}, len(questions_2) == {len(questions_2['contents'])}")
        start = end
        answers_2 = generate_answer_2(llm, questions_1["contents"], answers_1["contents"], questions_2["contents"])
        estimated_cost += answers_2["estimated_cost"]
        end = time()
        logger.debug(f"generate_answer_2: time == {end - start}, len(answers_2) == {len(answers_2['contents'])}")
        with open(outputs_jsonl, "a", encoding="utf-8") as f:
            for task, topic, item, q1, a1, q2, a2 in zip(tasks, topics, items, questions_1["contents"], answers_1["contents"], questions_2["contents"], answers_2["contents"]):  # noqa: E501
                messages = [
                    {"role": "user", "content": q1},
                    {"role": "assistant", "content": a1},
                    {"role": "user", "content": q2},
                    {"role": "assistant", "content": a2},
                ]
                record = {"messages": messages, "task": task, topic: item}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        loop_end = time()
        logger.debug(f"step: {i}, loop time == {loop_end - loop_start}, estimated_cost: {estimated_cost:.7f}")
    loop_end = time()
    logger.debug(f"loop time == {loop_end - loop_start}")


if __name__ == "__main__":
    main()
