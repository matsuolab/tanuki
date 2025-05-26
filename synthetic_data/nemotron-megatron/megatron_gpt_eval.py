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
from time import time

import torch
from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.modules.common.megatron.megatron_init import fake_initialize_model_parallel
from nemo.collections.nlp.parts.nlp_overrides import CustomProgressBar, NLPDDPStrategy, NLPSaveRestoreConnector
from nemo.core.config import hydra_runner
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import inject_model_parallel_rank
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer.trainer import Trainer
from torch.utils.data import DataLoader, Dataset

try:
    from megatron.core import parallel_state

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


def load_prompts(cfg):
    prompts = []
    if (prompts_jsonl := getattr(cfg, "prompts_jsonl", None)) is not None:
        with open(prompts_jsonl, "rt") as fp:
            prompts += list(map(json.loads, map(str.rstrip, fp)))
    # Make sure non-empty input
    assert len(prompts) > 0, "Expected at least one prompt"
    # Make sure all have the same type
    assert all(
        map(lambda x: isinstance(x, type(prompts[0])), prompts)
    ), "Expected all prompts to have the same datatype"
    return prompts


@hydra_runner(config_path="conf", config_name="megatron_gpt_inference")
def main(cfg) -> None:

    print("*" * 80)
    print("main: start")

    seed = getattr(cfg, "seed", -1)
    seed = None if seed < 0 else seed
    seed_everything(seed=seed)

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

    print("*" * 80)
    print("load_prompts: start")

    prompts = load_prompts(cfg)

    print(f"load_prompts: len(prompts) == {len(prompts)}")

    print("*" * 80)
    print("trainer.predict: start")

    config = OmegaConf.to_container(cfg.inference)
    model.set_inference_config(config)

    data_size = len(prompts)
    print(f"trainer.predict: data_size == {data_size}")
    chunk_size = getattr(cfg, "chunk_size", 1024)
    print(f"trainer.predict: chunk_size == {chunk_size}")
    batch_size = getattr(cfg, "batch_size", 256)
    print(f"trainer.predict: batch_size == {batch_size}")

    if (outputs_jsonl := getattr(cfg, "outputs_jsonl", None)) is not None:
        if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
            with open(outputs_jsonl, "w") as fp:
                fp.write("")

    loop_start = time()
    for i in range(0, data_size, chunk_size):
        chunk_start = i
        chunk_end = i + chunk_size if i + chunk_size < data_size else data_size
        chunk = prompts[chunk_start:chunk_end]
        ds = RequestDataSet(chunk)
        request_dl = DataLoader(dataset=ds, batch_size=batch_size)
        start = time()
        response = trainer.predict(model, request_dl)
        end = time()
        print(f"trainer.predict: time == {end - start}")
        print(f"trainer.predict: len(response) == {len(response)}")
        if (outputs_jsonl := getattr(cfg, "outputs_jsonl", None)) is not None:
            if parallel_state.is_pipeline_first_stage() and parallel_state.get_tensor_model_parallel_rank() == 0:
                with open(outputs_jsonl, "a") as fp:
                    for r in response:
                        for s in r["sentences"]:
                            fp.write(json.dumps(s, ensure_ascii=False) + "\n")
    loop_end = time()
    print(f"trainer.predict: loop time == {loop_end - loop_start}")
    print("*" * 80)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
