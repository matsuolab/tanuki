# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

# Generate preference dataset using LLM-as-a-Judge method.
# https://arxiv.org/pdf/2306.05685
# https://arxiv.org/pdf/2405.07863


# Get an API key from NVIDIA page.
# Press "Get API Key" button on the following page.
# https://build.nvidia.com/nvidia/nemotron-4-340b-instruct

# Set the API key as an environment variable.
# export NVIDIA_API_KEY="nvapi-..."


import argparse
import json
from abc import ABC, abstractmethod
from logging import DEBUG, StreamHandler, getLogger

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from vllm import LLM, SamplingParams  # type: ignore

# Set logging level to DEBUG.
logger = getLogger(__name__)
logger.setLevel(DEBUG)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.addHandler(handler)


class LanguageModel(ABC):
    def __init__(self, model_name_or_path, temperature=0.7, max_tokens=512, seed=None):
        self.model_name_or_path = model_name_or_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        logger.debug(
            f"model_name_or_path: {model_name_or_path}, temperature: {temperature}, max_tokens: {max_tokens}, seed: {seed}"  # noqa: E501
        )

    @abstractmethod
    def __call__(self, messages_list):
        pass


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
        self.stop = ["<|endoftext|>", "<extra_id_0>", "<extra_id_1>", "\x11"]
        self.repetition_penalty = 1.1
        # TODO: test enforce_eager=False and see the performance and stability.
        self.vllm = LLM(
            model_name_or_path, tensor_parallel_size=8, distributed_executor_backend="ray", enforce_eager=True
        )  # noqa: E501

    def __call__(self, messages_list):
        prompts = [
            self.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            for messages in messages_list
        ]  # noqa: E501
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=self.stop,
            repetition_penalty=self.repetition_penalty,
        )  # noqa: E501
        outputs = self.vllm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        # logger.debug(f"len(outputs): {len(outputs)}")
        contents = [o.outputs[0].text for o in outputs]
        return {"contents": contents, "estimated_cost": 0.0}


class Calm3VLLM(LanguageModel):
    def __init__(self, model_name_or_path, temperature=0.7, max_tokens=512, seed=None):
        super().__init__(model_name_or_path, temperature=temperature, max_tokens=max_tokens, seed=seed)
        self.stop = ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]
        self.repetition_penalty = 1.1
        # TODO: test enforce_eager=True and see the performance and stability.
        self.vllm = LLM(model_name_or_path)

    def __call__(self, messages_list):
        tokenizer = self.vllm.get_tokenizer()
        prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]  # noqa: E501
        # logger.debug(f"prompts[0]: {prompts[0]}")
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            stop=self.stop,
            repetition_penalty=self.repetition_penalty,
        )  # noqa: E501
        outputs = self.vllm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
        # logger.debug(f"outputs[0].outputs[0].text: {outputs[0].outputs[0].text}")
        contents = [o.outputs[0].text for o in outputs]
        return {"contents": contents, "estimated_cost": 0.0}


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


def parse_judge_content(judge_content):
    is_a = is_b = is_c = False
    if "[[A]]" in judge_content:
        is_a = True
    if "[[B]]" in judge_content:
        is_b = True
    if "[[C]]" in judge_content:
        is_c = True
    judge = "C"
    if is_a and is_b:
        # raise ValueError(f"Both A and B are chosen: {judge_content}")
        logger.debug(f"Both A and B are chosen: {judge_content}")
        judge = "C"
    elif is_a:
        judge = "A"
    elif is_b:
        judge = "B"
    elif is_c:
        judge = "C"
    else:
        # raise ValueError(f"Unknown judge: {judge_content}")
        logger.debug(f"Unknown judge: {judge_content}")
        judge = "C"
    # logger.debug(f"judge: {judge}")
    return judge


def judge_q_a_a(llm, questions, answer_1s, answer_2s):
    messages_list = [
        [
            {"role": "system", "content": JUDGE_PROMPT["system_prompt"]},
            {
                "role": "user",
                "content": JUDGE_PROMPT["prompt_template"].format(
                    question=question, answer_a=answer_1, answer_b=answer_2
                ),
            },  # noqa: E501
        ]
        for question, answer_1, answer_2 in zip(questions, answer_1s, answer_2s)
    ]
    response = llm(messages_list)
    assert len(response["contents"]) == len(messages_list)
    return {
        "judge_contents": response["contents"],
        "judges": [parse_judge_content(judge_content) for judge_content in response["contents"]],
    }


parser = argparse.ArgumentParser(description="Generate synth dataset.")
parser.add_argument("--input_jsonl", type=str, help="Input jsonl file.", default=None)
parser.add_argument("--output_jsonl", type=str, help="Output jsonl file.", default="output.jsonl")
parser.add_argument("--skip_jsonl", type=str, help="Skip jsonl file.", default="skip.jsonl")
parser.add_argument("--dataset_path", type=str, help="Dataset path.", default=None)
parser.add_argument("--dataset_name", type=str, help="Dataset name.", default="default")
parser.add_argument("--dataset_split", type=str, help="Dataset split.", default="train")
parser.add_argument("--cache_dir", type=str, help="Cache directory.", default=None)
parser.add_argument("--base_url", type=str, help="API base URL.", default="")
parser.add_argument("--api_key", type=str, help="API key.", default="")
parser.add_argument(
    "--model", type=str, help="Model name.", default="mgoin/Nemotron-4-340B-Instruct-hf-FP8"
)  # noqa: E501
# parser.add_argument("--base_url", type=str, help="API base URL.", default="https://integrate.api.nvidia.com/v1")
# parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("NVIDIA_API_KEY"))
# parser.add_argument("--model", type=str, help="Model name.", default="nvidia/nemotron-4-340b-instruct")
# parser.add_argument("--base_url", type=str, help="API base URL.", default="https://api.deepinfra.com/v1/openai")
# parser.add_argument("--api_key", type=str, help="API key.", default=os.getenv("DEEPINFRA_API_KEY"))
# parser.add_argument("--model", type=str, help="Model name.", default="meta-llama/Meta-Llama-3.1-405B-Instruct")
parser.add_argument("--temperature", type=float, help="Temperature for inference.", default=0.0)
parser.add_argument("--max_tokens", type=int, help="Max tokens for inference.", default=512)
parser.add_argument("--seed", type=int, help="Random seed.", default=1)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=64)
args = parser.parse_args()

input_jsonl = args.input_jsonl
output_jsonl = args.output_jsonl
skip_jsonl = args.skip_jsonl
dataset_path = args.dataset_path
dataset_name = args.dataset_name
dataset_split = args.dataset_split
cache_dir = args.cache_dir
base_url = args.base_url if args.base_url.strip() != "" else None
api_key = args.api_key if args.api_key.strip() != "" else None
model = args.model
temperature = args.temperature
max_tokens = args.max_tokens
seed = 0 if args.seed < 0 else args.seed  # TODO: generate random seed
batch_size = args.batch_size

if base_url is None and api_key is None:
    if "nemotron" in model.lower():
        llm = NemotronVLLM(model, temperature=temperature, max_tokens=max_tokens, seed=seed)
    elif "calm3" in model.lower():
        llm = Calm3VLLM(model, temperature=temperature, max_tokens=max_tokens, seed=seed)
    else:
        raise ValueError(f"Unknown model: {model}")
else:
    llm = OpenAIAPI(
        model, base_url=base_url, api_key=api_key, temperature=temperature, max_tokens=max_tokens, seed=seed
    )  # noqa: E501

if input_jsonl is not None:
    dataset = load_dataset("json", data_files=input_jsonl, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
elif dataset_path is not None:
    dataset = load_dataset(dataset_path, name=dataset_name, split=dataset_split, cache_dir=cache_dir)
else:
    raise ValueError("You must specify either input_jsonl or dataset_path.")

# https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl
JUDGE_PROMPT = {
    "name": "pair-v2",
    "type": "pairwise",
    "system_prompt": 'Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.',  # noqa: E501
    "prompt_template": "[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]",  # noqa: E501
    "description": "Prompt for general questions",
    "category": "general",
    "output_format": "[[A]]",
}

end = len(dataset)
start = 0

with open(output_jsonl, "w", encoding="utf-8") as f_output, open(skip_jsonl, "w", encoding="utf-8") as f_skip:
    for i in tqdm(range(start, end, batch_size)):
        batch_start = i
        batch_end = min(i + batch_size, end)
        data = dataset[batch_start:batch_end]
        questions = data["question"]
        answer_1s = data["answer_1"]
        answer_2s = data["answer_2"]

        # I judged twice with the order of answer_1 and answer_2 reversed to avoid "order bias".
        # e.g. the first answer tends to be judged better than the second.
        # A means first answer is better than second answer.
        # B means second answer is better than first answer.
        # If answer_1 is better than answer_2, judge_1_2 should be "A" and judge_2_1 should be "B".
        # If answer_2 is better than answer_1, judge_1_2 should be "B" and judge_2_1 should be "A".
        judge_1_2s = judge_q_a_a(llm, questions, answer_1s, answer_2s)
        judge_2_1s = judge_q_a_a(llm, questions, answer_2s, answer_1s)

        for question, answer_1, answer_2, judge_1_2, judge_2_1, judge_content_1_2, judge_content_2_1 in zip(
            questions,
            answer_1s,
            answer_2s,
            judge_1_2s["judges"],
            judge_2_1s["judges"],
            judge_1_2s["judge_contents"],
            judge_2_1s["judge_contents"],
        ):
            judge = "C"
            chosen = answer_1
            rejected = answer_2
            if judge_1_2 == "A" and judge_2_1 == "B":  # 1 is better than 2
                judge = "A"
                # answer_1 should be chosen
                chosen = answer_1
                rejected = answer_2
            elif judge_1_2 == "B" and judge_2_1 == "A":  # 2 is better than 1
                judge = "B"
                # answer_2 should be chosen
                chosen = answer_2
                rejected = answer_1
            elif judge_1_2 == "A" and judge_2_1 == "A":  # disagreement
                # this might be an order bias or something wrong :D
                # TODO: check the answers and judge_content
                judge = "D"
            elif judge_1_2 == "B" and judge_2_1 == "B":  # disagreement
                # this might be an order bias or something wrong :D
                # TODO: check the answers and judge_content
                judge = "D"
            else:
                judge = "C"

            result = {
                "prompt": question,
                "chosen": chosen,
                "rejected": rejected,
                "judge": judge,
                "judge_1_2": judge_1_2,
                "judge_2_1": judge_2_1,
                "judge_content_1_2": judge_content_1_2,
                "judge_content_2_1": judge_content_2_1,
            }

            f = f_output if judge in ["A", "B"] else f_skip
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
