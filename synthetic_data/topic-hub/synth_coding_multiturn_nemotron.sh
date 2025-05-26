#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-7
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=160
#SBATCH --mem=1600GB
#SBATCH --job-name=synth_coding_nemotron
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate vllm


task_jsonl="data/coding_task.jsonl"
topic_jsonl="data/coding_topic.jsonl"
outputs_jsonl="data/outputs.topic.coding.nemotron.1.jsonl"
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
num_samples=1000000000
batch_size=64
temperature=0.7
max_tokens=1024
seed=1


time LD_LIBRARY_PATH="" python scripts/synth_topic_multiturn.py \
    --task_jsonl="$task_jsonl" \
    --topic_jsonl="$topic_jsonl" \
    --outputs_jsonl="$outputs_jsonl" \
    --base_url="$base_url" \
    --api_key="$api_key" \
    --model="$model" \
    --num_samples="$num_samples" \
    --batch_size="$batch_size" \
    --temperature="$temperature" \
    --max_tokens="$max_tokens" \
    --seed="$seed" \
