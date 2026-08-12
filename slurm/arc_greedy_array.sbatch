#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:25:00
#SBATCH --array=1-24%12
#SBATCH --output=logs/arc_array/arc_array-%A_%a.out
# Eval-only: ARC-Easy on existing GPT-2 Medium checkpoints. Seeds 43-46, gens 0-7.
set -euo pipefail
source ~/modelcollapse/scripts/narval_env.sh

IDX=$((SLURM_ARRAY_TASK_ID - 1))
SEED=$((43 + IDX / 8))
GEN=$((IDX % 8))
RUN_ROOT="$SCRATCH/collapse/gpt2m_greedy"
MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
OUT="$RUN_ROOT/seed$SEED/gen$GEN/benchmarks_arc"

echo "[task $SLURM_ARRAY_TASK_ID] seed=$SEED gen=$GEN"
test -d "$MODEL_DIR" || { echo "MISSING $MODEL_DIR"; exit 1; }

lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16" \
    --tasks arc_easy \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$OUT"
