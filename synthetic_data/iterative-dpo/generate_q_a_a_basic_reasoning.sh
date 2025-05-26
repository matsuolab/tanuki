#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --job-name=gen_q_a_a
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm


input_jsonl="data/synth-topic-jp-basic-reasoning-calm3.20240809.shuf.jsonl"
output_jsonl="data/basic-reasoning-sftbest-1.20240810.tanuki_q_a_a.jsonl"
# input_jsonl="data/synth-topic-jp-basic-math-calm3.20240809.shuf.jsonl"
# output_jsonl="data/basic-math-sftbest-1.20240810.tanuki_q_a_a.jsonl"
# dataset_path="team-hatakeyama-phase2/synth-persona-jp-resoning-nemotron-4"
dataset_path=""
dataset_name="default"
dataset_split="train"
cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
# cache_dir=""
# model="/storage5/shared/Llama-3-8/HF/0809inst_iter_0000200"
model="/storage5/personal/shioya/sft_model/sftlab-experiments/8B/pass4_exp001-0809_exp_01-zero1"
tokenizer="team-hatakeyama-phase2/tanuki-tokenizer-with-space-not-add-special"
batch_size=512


time LD_LIBRARY_PATH="" python scripts/generate_q_a_a.py \
    --input_jsonl="$input_jsonl" \
    --output_jsonl="$output_jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir" \
    --model="$model" \
    --tokenizer="$tokenizer" \
    --batch_size="$batch_size" \
