#!/bin/bash

# SPDX-FileCopyrightText: 2024 Susumu OTA <1632335+susumuota@users.noreply.github.com>
# SPDX-License-Identifier: Apache-2.0

#SBATCH --nodelist=slurm0-a3-ghpc-0
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --job-name=sort_dataset
#SBATCH --output=logs/%x_%j.log


source $EXP_HOME/miniconda3/etc/profile.d/conda.sh
conda activate askllm311


input_jsonl="data/synth-persona-jp-resoning-nemotron-4.askllm_q1_a1_q2_a2.jsonl"
output_jsonl="data/synth-persona-jp-resoning-nemotron-4.sorted_q1.jsonl"
column="q1_solvable_score"
reverse="True"
dataset_path=""
dataset_name="default"
dataset_split="train"
cache_dir=""


time python scripts/sort_dataset.py \
    --input_jsonl="$input_jsonl" \
    --output_jsonl="$output_jsonl" \
    --column="$column" \
    --reverse="$reverse" \
    --dataset_path="$dataset_path" \
    --dataset_name="$dataset_name" \
    --dataset_split="$dataset_split" \
    --cache_dir="$cache_dir" \
