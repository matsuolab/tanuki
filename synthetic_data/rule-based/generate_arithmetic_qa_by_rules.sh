#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --job-name=rule_arithmetic
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311


outputs_jsonl="data/outputs.arithmetic.1.jsonl"
num_samples=1000000000
seed="-1"
log_interval=10000


time python scripts/generate_arithmetic_qa_by_rules.py \
    --num_samples="$num_samples" \
    --outputs_jsonl="$outputs_jsonl" \
    --seed="$seed" \
    --log_interval="$log_interval" \
