#!/bin/bash
#SBATCH --account=def-enaskt
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-6
#SBATCH --job-name=m1
#SBATCH --output=logs/m1/m1-%A_%a.out
# M1 scorer (decisions §61). Task -> (config, seed, gens). gpt2m_greedy s43 gens 0-1 were the smoke run.
set -eo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
source scripts/narval_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

CFG=(gpt2m_greedy gpt2m_greedy gpt2m_greedy pilot_gpt2m pilot_gpt2m pilot_gpt2m pilot_gpt2m)
SEED=(43 44 45 43 44 45 46)
i=$SLURM_ARRAY_TASK_ID
if [ "$i" -eq 0 ]; then GENS="2 3 4 5 6 7"; else GENS="0 1 2 3 4 5 6 7"; fi

echo "m1 task $i job ${SLURM_ARRAY_JOB_ID}_$i node $(hostname) head $(git rev-parse --short HEAD) cfg ${CFG[$i]} seed ${SEED[$i]} gens $GENS"
python scripts/m1_decompose.py --config configs/${CFG[$i]}.yaml --seed ${SEED[$i]} --gens $GENS
