#!/bin/bash
#SBATCH -N 2
#SBATCH -J generation
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8

#SBATCH --time=01:00:00
#SBATCH --nodelist=slurm0-a3-ghpc-[1,2]
#SBATCH --cpus-per-task=24
#SBATCH --mem=1600GB
#SBATCH --output=logs/%x_%j.log

set -x

SCRIPTS="/storage5/shared/username/nemotron-inference/scripts"
LOGS="/storage5/shared/username/nemotron-inference/logs"
DATA="/storage5/shared/username/nemotron-inference/data"
MODEL="/storage5/shared/Nemotron-4-340B-Instruct"
# CONTAINER="nvcr.io/nvidia/nemo:24.05"
CONTAINER="/storage5/shared/username/nemotron-inference/nvidia+nemo+24.05.sqsh"

OUTFILE="${LOGS}/slurm-%j-%n.out"
ERRFILE="${LOGS}/error-%j-%n.out"
MOUNTS="${SCRIPTS}:/scripts,${DATA}:/data,${MODEL}:/model"

read -r -d '' COMMAND <<EOF
bash /scripts/nemo_inference.sh /model
EOF

time srun -o "${OUTFILE}" -e "${ERRFILE}" --container-writable --container-image="${CONTAINER}" --container-mounts="${MOUNTS}" bash -c "${COMMAND}"
