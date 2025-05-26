#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-4
#SBATCH --time=08:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1600GB
#SBATCH --job-name=judge_reasoning_q_a_a
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm


input_jsonl="data/basic-reasoning-sftbest-1.20240810.tanuki_q_a_a.jsonl"
output_jsonl="data/basic-reasoning-sftbest-1.20240810.tanuki_q_a_a.nemotron_judge.jsonl"
skip_jsonl="data/basic-reasoning-sftbest-1.20240810.tanuki_q_a_a.nemotron_judge_skip.jsonl"
dataset_path=""
dataset_name="default"
dataset_split="train"
cache_dir="/storage5/shared/ota_s/askllm/hf_cache"
# base_url="https://integrate.api.nvidia.com/v1"
# api_key="$NVIDIA_API_KEY"
# model="nvidia/nemotron-4-340b-instruct"
# base_url="https://api.deepinfra.com/v1/openai"
# api_key="$DEEPINFRA_API_KEY"
# model="meta-llama/Meta-Llama-3.1-405B-Instruct"
# model="microsoft/Phi-3-medium-4k-instruct"
base_url=""
api_key=""
model="/storage7/tmp/ota/Nemotron-4-340B-Instruct-hf-FP8"
# model="cyberagent/calm3-22b-chat"
# model="/storage7/tmp/ota/calm3-22b-chat"
temperature=0.0
max_tokens=1024
seed=1
batch_size=64


time LD_LIBRARY_PATH="" python scripts/judge_q_a_a.py \
    --input_jsonl="$input_jsonl" \
    --output_jsonl="$output_jsonl" \
    --skip_jsonl="$skip_jsonl" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir" \
    --base_url="$base_url" \
    --api_key="$api_key" \
    --model="$model" \
    --temperature="$temperature" \
    --max_tokens="$max_tokens" \
    --seed="$seed" \
    --batch_size="$batch_size" \
