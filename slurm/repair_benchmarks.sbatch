#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=repair-bench
#SBATCH --output=logs/misc/%x-%j.out

# Rerun lm-eval only on already-trained GPT-2 models (no finetuning).
# Usage: sbatch repair_benchmarks.sbatch <SEED>

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

SEED=${1:?usage: sbatch repair_benchmarks.sbatch SEED}
cd "$HOME/modelcollapse"
source scripts/narval_env.sh

RUN_ROOT="$SCRATCH/collapse/pilot_gpt2m"

for GEN in $(seq 0 7); do
    MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
    OUT_DIR="$RUN_ROOT/seed$SEED/gen$GEN/benchmarks"
    
    if [ ! -d "$MODEL_DIR" ]; then
        echo "SKIP seed$SEED gen$GEN — model dir missing"
        continue
    fi
    if [ -f "$OUT_DIR/results.json" ]; then
        echo "SKIP seed$SEED gen$GEN — already done"
        continue
    fi
    
    echo "=== seed$SEED gen$GEN | $(date) ==="
    mkdir -p "$OUT_DIR"
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16" \
        --tasks hellaswag,lambada_openai \
        --num_fewshot 0 \
        --batch_size auto \
        --output_path "$OUT_DIR"
    echo "gen$GEN done"
done

echo "=== repair complete for seed$SEED ==="
