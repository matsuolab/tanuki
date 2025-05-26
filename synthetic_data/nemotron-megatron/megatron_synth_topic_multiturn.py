# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, Susumu Ota.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import random
import re
from time import time

import torch  # type: ignore
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel  # type: ignore
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel  # type: ignore
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector  # type: ignore  # noqa: E501
from nemo.core.config import hydra_runner  # type: ignore
from nemo.utils.app_state import AppState  # type: ignore
from nemo.utils.model_utils import inject_model_parallel_rank  # type: ignore
from omegaconf import OmegaConf, open_dict  # type: ignore
from pytorch_lightning import seed_everything  # type: ignore
from pytorch_lightning.trainer.trainer import Trainer  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

try:
    from megatron.core import parallel_state  # type: ignore

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


if not torch.cuda.is_available():
    raise EnvironmentError("GPU is needed for the inference")


class RequestDataSet(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.sentences = sentences

    def __len__(
        self,
    ):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


def inference(prompts, trainer, model, batch_size):
    ds = RequestDataSet(prompts)
    request_dl = DataLoader(dataset=ds, batch_size=batch_size)
    return trainer.predict(model, request_dl)


def convert_messages_to_prompt(messages):
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
    if messages[-1]["role"] == "system":
        prompts.append("<extra_id_1>User\n")
    elif messages[-1]["role"] == "user":
        prompts.append("<extra_id_1>Assistant\n")
    elif messages[-1]["role"] == "assistant":
        prompts.append("<extra_id_1>User\n")
    else:
        raise ValueError(f"Unknown role: {messages[-1]['role']}")
    return "".join(prompts)


def remove_stop_tokens(text):
    text = re.sub(r"<\|endoftext\|>$", "", text)
    text = re.sub(r"<extra_id_1>$", "", text)
    text = re.sub(r"<extra_id_0>$", "", text)
    text = re.sub(r"\x11$", "", text)
    text = re.sub(r"\n$", "", text)
    return text


def convert_response_to_messages(text):
    if m := re.match(r"<extra_id_0>System\n(.*)<extra_id_1>User\n(.*)<extra_id_1>Assistant\n(.*)<extra_id_1>User\n(.*)<extra_id_1>Assistant\n(.*)", text, flags=re.DOTALL):  # noqa: E501
        return [
            {"role": "system", "content": remove_stop_tokens(m.group(1))},
            {"role": "user", "content": remove_stop_tokens(m.group(2))},
            {"role": "assistant", "content": remove_stop_tokens(m.group(3))},
            {"role": "user", "content": remove_stop_tokens(m.group(4))},
            {"role": "assistant", "content": remove_stop_tokens(m.group(5))},
        ]
    elif m := re.match(r"<extra_id_0>System\n(.*)<extra_id_1>User\n(.*)<extra_id_1>Assistant\n(.*)<extra_id_1>User\n(.*)", text, flags=re.DOTALL):  # noqa: E501
        return [
            {"role": "system", "content": remove_stop_tokens(m.group(1))},
            {"role": "user", "content": remove_stop_tokens(m.group(2))},
            {"role": "assistant", "content": remove_stop_tokens(m.group(3))},
            {"role": "user", "content": remove_stop_tokens(m.group(4))},
        ]
    elif m := re.match(r"<extra_id_0>System\n(.*)<extra_id_1>User\n(.*)<extra_id_1>Assistant\n(.*)", text, flags=re.DOTALL):  # noqa: E501
        return [
            {"role": "system", "content": remove_stop_tokens(m.group(1))},
            {"role": "user", "content": remove_stop_tokens(m.group(2))},
            {"role": "assistant", "content": remove_stop_tokens(m.group(3))},
        ]
    elif m := re.match(r"<extra_id_0>System\n(.*)<extra_id_1>User\n(.*)", text, flags=re.DOTALL):
        return [
            {"role": "system", "content": remove_stop_tokens(m.group(1))},
            {"role": "user", "content": remove_stop_tokens(m.group(2))},
        ]
    elif m := re.match(r"<extra_id_0>System\n(.*)", text, flags=re.DOTALL):
        return [
            {"role": "system", "content": remove_stop_tokens(m.group(1))},
        ]
    else:
        raise ValueError(f"Unknown format: {text}")


PROMPT_TEMPLATE = """Create a {task} problem related to the following {topic}:

{item}

Note:

1. The {task} problem should be simple and involve basic {task} skills and knowledge. All average high school students can solve it correctly.
2. You should make full use of the {topic} description to create the {task} problem to ensure that the {task} problem is unique and specific to the {topic}.
3. Your response should always start with "Problem:". Your response should not include a solution to the created {task} problem.
4. 日本語で回答しなさい。
"""  # noqa: E501


def generate_question_1(tasks, topics, items, trainer, model, batch_size):
    prompts = []
    for task, topic, item in zip(tasks, topics, items):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": PROMPT_TEMPLATE.format(task=task.strip(), topic=topic.strip(), item=item.strip())},  # noqa: E501
        ]
        prompts.append(convert_messages_to_prompt(messages))
    responses = inference(prompts, trainer, model, batch_size)
    contents = []
    for r in responses:
        for s in r["sentences"]:
            m = convert_response_to_messages(s)
            content = m[-1]["content"]
            content = re.sub(r"^Problem[：:]", "", content, flags=re.DOTALL)
            content = re.sub(r"^問題[：:]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"[\(（]翻?訳[：:].*?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（][Aa]nswer:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"[\(（]答え.*?[\)）]", "", content, flags=re.DOTALL)
            content = re.sub(r"\n\n答え[：:].*?", "", content, flags=re.DOTALL)
            content = content.strip()
            contents.append(content)
    return contents


def generate_answer_1(questions, trainer, model, batch_size):
    prompts = []
    for question in questions:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{question}\n\n日本語で回答しなさい。"},
        ]
        prompts.append(convert_messages_to_prompt(messages))
    responses = inference(prompts, trainer, model, batch_size)
    contents = []
    for r in responses:
        for s in r["sentences"]:
            m = convert_response_to_messages(s)
            content = m[-1]["content"]
            content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = content.strip()
            contents.append(content)
    return contents


def generate_question_2(question_1, answer_1, trainer, model, batch_size):
    prompts = []
    for q1, a1 in zip(question_1, answer_1):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": "前述の問題をより理解するために、簡潔な追加の質問を一つ作りなさい。問題の一部を変更したり、条件を追加しても良いです。追加の質問だけを書き、決して答えを含めないでください。"},  # noqa: E501
        ]
        prompts.append(convert_messages_to_prompt(messages))
    responses = inference(prompts, trainer, model, batch_size)
    contents = []
    for r in responses:
        for s in r["sentences"]:
            m = convert_response_to_messages(s)
            content = m[-1]["content"]
            content = re.sub(r"^追加の質問[：:]", "", content, flags=re.DOTALL)
            content = re.sub(r"^質問[：:]", "", content, flags=re.DOTALL)
            content = re.sub(r"[\(（][Tt]ranslation[：:].*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = re.sub(r"^「(.+)」$", "\\1", content, flags=re.DOTALL)
            content = content.strip()
            contents.append(content)
    return contents


def generate_answer_2(question_1, answer_1, question_2, trainer, model, batch_size):
    prompts = []
    for q1, a1, q2 in zip(question_1, answer_1, question_2):
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": q1},
            {"role": "assistant", "content": a1},
            {"role": "user", "content": q2},
        ]
        prompts.append(convert_messages_to_prompt(messages))
    responses = inference(prompts, trainer, model, batch_size)
    contents = []
    for r in responses:
        for s in r["sentences"]:
            m = convert_response_to_messages(s)
            content = m[-1]["content"]
            content = re.sub(r"[\(（][Tt]ranslation:.*?[\)）]", "", content, flags=re.DOTALL)  # cspell: disable-line
            content = content.strip()
            contents.append(content)
    return contents


@hydra_runner(config_path="conf", config_name="megatron_synth_topic_multiturn")
def main(cfg) -> None:

    seed = getattr(cfg, "seed", -1)
    seed = 0 if seed < 0 else seed

    seed_everything(seed=seed)
    print(f"seed == {seed}")

    callbacks = []
    # enable_progress_bar is True by default. If cfg.trainer.enable_progress_bar=False, CustomProgressBar is not appended to callbacks  # noqa: E501
    if "enable_progress_bar" not in cfg.trainer or cfg.trainer.enable_progress_bar:
        callbacks.append(CustomProgressBar())
    # trainer required for restoring model parallel models
    trainer = Trainer(
        strategy=NLPDDPStrategy(timeout=datetime.timedelta(seconds=18000)),
        **cfg.trainer,
        callbacks=callbacks,
    )

    if cfg.gpt_model_file is not None:
        if (
            cfg.tensor_model_parallel_size < 0
            or cfg.pipeline_model_parallel_size < 0
            or cfg.get("pipeline_model_parallel_split_rank", -1) < 0
        ):
            save_restore_connector = NLPSaveRestoreConnector()
            if os.path.isdir(cfg.gpt_model_file):
                save_restore_connector.model_extracted_dir = cfg.gpt_model_file
            model_config = MegatronGPTModel.restore_from(
                restore_path=cfg.gpt_model_file,
                trainer=trainer,
                return_config=True,
                save_restore_connector=save_restore_connector,
            )

            # with dist checkpointing we don't need to set this
            if not model_config.get("mcore_gpt", False):
                with open_dict(cfg):
                    cfg.tensor_model_parallel_size = model_config.get("tensor_model_parallel_size", 1)
                    cfg.pipeline_model_parallel_size = model_config.get("pipeline_model_parallel_size", 1)
                    cfg.pipeline_model_parallel_split_rank = model_config.get("pipeline_model_parallel_split_rank", 0)

    assert (
        cfg.trainer.devices * cfg.trainer.num_nodes
        == cfg.tensor_model_parallel_size
        * cfg.pipeline_model_parallel_size
        * max(1, cfg.get("expert_model_parallel_size", 1))
    ), "devices * num_nodes should equal tensor_model_parallel_size * pipeline_model_parallel_size"

    if cfg.gpt_model_file:
        save_restore_connector = NLPSaveRestoreConnector()
        if os.path.isdir(cfg.gpt_model_file):
            save_restore_connector.model_extracted_dir = cfg.gpt_model_file

        pretrained_cfg = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            return_config=True,
            save_restore_connector=save_restore_connector,
        )
        OmegaConf.set_struct(pretrained_cfg, True)
        with open_dict(pretrained_cfg):
            pretrained_cfg.sequence_parallel = False
            pretrained_cfg.activations_checkpoint_granularity = None
            pretrained_cfg.activations_checkpoint_method = None
            pretrained_cfg.precision = trainer.precision
            pretrained_cfg["use_flash_attention"] = cfg.inference.get("use_flash_attention", False)
            pretrained_cfg["apply_rope_fusion"] = False
            if pretrained_cfg.get("mcore_gpt", False):
                # with dist checkpointing we can use the model parallel config specified by the user
                pretrained_cfg.tensor_model_parallel_size = cfg.tensor_model_parallel_size
                pretrained_cfg.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
                pretrained_cfg.expert_model_parallel_size = cfg.get("expert_model_parallel_size", 1)
                pretrained_cfg.micro_batch_size = 1
            if trainer.precision == "16":
                pretrained_cfg.megatron_amp_O2 = False
            elif trainer.precision in ["bf16", "bf16-mixed"] and cfg.get("megatron_amp_O2", False):
                pretrained_cfg.megatron_amp_O2 = True
        model = MegatronGPTModel.restore_from(
            restore_path=cfg.gpt_model_file,
            trainer=trainer,
            override_config_path=pretrained_cfg,
            save_restore_connector=save_restore_connector,
            map_location=f"cuda:{trainer.local_rank}",  # map_location is needed for converted models
        )
    elif cfg.checkpoint_dir:
        app_state = AppState()
        if (
            cfg.tensor_model_parallel_size > 1
            or cfg.pipeline_model_parallel_size > 1
            or cfg.get("expert_model_parallel_size", 1) > 1
        ):
            app_state.model_parallel_size = (
                cfg.tensor_model_parallel_size
                * cfg.pipeline_model_parallel_size
                * cfg.get("expert_model_parallel_size", 1)
            )
            app_state.tensor_model_parallel_size = cfg.tensor_model_parallel_size
            app_state.pipeline_model_parallel_size = cfg.pipeline_model_parallel_size
            app_state.expert_model_parallel_size = cfg.get("expert_model_parallel_size", 1)
            (
                app_state.tensor_model_parallel_rank,
                app_state.pipeline_model_parallel_rank,
                app_state.expert_model_parallel_rank,
                app_state.model_parallel_size,
                app_state.data_parallel_size,
                app_state.pipeline_model_parallel_split_rank,
                app_state.virtual_pipeline_model_parallel_rank,
            ) = fake_initialize_model_parallel(
                world_size=app_state.model_parallel_size,
                rank=trainer.global_rank,
                tensor_model_parallel_size_=cfg.tensor_model_parallel_size,
                pipeline_model_parallel_size_=cfg.pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank_=cfg.pipeline_model_parallel_split_rank,
                expert_model_parallel_size_=cfg.get("expert_model_parallel_size", 1),
            )
        checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        # checkpoint_path is a dir in case of distributed checkpointing
        if not os.path.isdir(checkpoint_path):
            # legacy checkpoint needs model parallel rank injection
            checkpoint_path = inject_model_parallel_rank(os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name))
        model = MegatronGPTModel.load_from_checkpoint(checkpoint_path, hparams_file=cfg.hparams_file, trainer=trainer)
    else:
        raise ValueError("need at least a nemo file or checkpoint dir")

    model.freeze()

    # Have to turn off activations_checkpoint_method for inference
    try:
        model.model.language_model.encoder.activations_checkpoint_method = None
    except AttributeError:
        pass

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    num_samples = getattr(cfg, "num_samples", 10000)
    print(f"num_samples == {num_samples}")
    batch_size = getattr(cfg, "batch_size", 256)
    print(f"batch_size == {batch_size}")

    if (task_jsonl := getattr(cfg, "task_jsonl", None)) is not None:
        with open(task_jsonl, "r") as fp:
            TASK_LIST = [json.loads(line)["task"] for line in fp]

    if (topic_jsonl := getattr(cfg, "topic_jsonl", None)) is not None:
        with open(topic_jsonl, "r") as fp:
            TOPIC_LIST = [json.loads(line) for line in fp]

    loop_start = time()
    for i in range(0, num_samples):
        seed_everything(seed=(seed+i))
        print(f"seed == {(seed+i)}")
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
        start = time()
        question_1 = generate_question_1(tasks, topics, items, trainer, model, batch_size)
        end = time()
        print(f"generate_question_1: time == {end - start}")
        start = end
        answer_1 = generate_answer_1(question_1, trainer, model, batch_size)
        end = time()
        print(f"generate_answer_1: time == {end - start}")
        start = end
        question_2 = generate_question_2(question_1, answer_1, trainer, model, batch_size)
        end = time()
        print(f"generate_question_2: time == {end - start}")
        start = end
        answer_2 = generate_answer_2(question_1, answer_1, question_2, trainer, model, batch_size)
        end = time()
        print(f"generate_answer_2: time == {end - start}")
        if (outputs_jsonl := getattr(cfg, "outputs_jsonl", None)) is not None:
            if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
                with open(outputs_jsonl, "a", encoding="utf-8") as fp:
                    for task, topic, item, q1, a1, q2, a2 in zip(tasks, topics, items, question_1, answer_1, question_2, answer_2):  # noqa: E501
                        messages = [
                            {"role": "user", "content": q1},
                            {"role": "assistant", "content": a1},
                            {"role": "user", "content": q2},
                            {"role": "assistant", "content": a2},
                        ]
                        record = {"messages": messages, "task": task, topic: item}
                        fp.write(json.dumps(record, ensure_ascii=False) + "\n")
    loop_end = time()
    print(f"loop time == {loop_end - loop_start}")


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
