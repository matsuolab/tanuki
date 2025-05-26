#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --job-name=synth_api
#SBATCH --output=logs/%x_%j.log


# source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
# conda activate askllm311


task_jsonl="data/arithmetic_task.jsonl"
topic_jsonl="data/arithmetic_topic.jsonl"
outputs_jsonl="data/outputs.arithmetic.1.jsonl"
# base_url="https://integrate.api.nvidia.com/v1"
# api_key="$NVIDIA_API_KEY"
# model="nvidia/nemotron-4-340b-instruct"
base_url="https://api.deepinfra.com/v1/openai"
api_key="$DEEPINFRA_API_KEY"
# model="meta-llama/Meta-Llama-3.1-405B-Instruct"
model="microsoft/Phi-3-medium-4k-instruct"
num_samples=10
temperature=0.7
max_tokens=512
seed="-1"


time python scripts/synth_arithmetic_multiturn.py \
    --task_jsonl="$task_jsonl" \
    --topic_jsonl="$topic_jsonl" \
    --outputs_jsonl="$outputs_jsonl" \
    --base_url="$base_url" \
    --api_key="$api_key" \
    --model="$model" \
    --num_samples="$num_samples" \
    --temperature="$temperature" \
    --max_tokens="$max_tokens" \
    --seed="$seed" \
