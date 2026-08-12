#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=logsamples
#SBATCH --output=logs/misc/%x-%j.out
set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh
RUN=${1:?run_name}; GEN=${2:?gen}; SEED=${3:?seed}
GD="$SCRATCH/collapse/$RUN/seed$SEED/gen$GEN"
[ -d "$GD/model" ] || { echo "no model at $GD/model"; exit 1; }
lm_eval --model hf \
  --model_args "pretrained=$GD/model,dtype=bfloat16" \
  --tasks hellaswag,arc_easy --num_fewshot 0 --batch_size auto \
  --log_samples --output_path "$GD/benchmarks_logsamples"
echo "=== done $RUN seed$SEED gen$GEN ==="
