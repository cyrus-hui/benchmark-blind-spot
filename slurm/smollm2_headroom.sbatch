#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=smollm2-headroom
#SBATCH --output=logs/misc/%x-%j.out

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

cd "$HOME/modelcollapse"
source scripts/narval_env.sh

MODEL="HuggingFaceTB/SmolLM2-1.7B"
OUT="$SCRATCH/collapse/smollm2_1.7b/headroom"
mkdir -p "$OUT"

echo "=== SmolLM2-1.7B headroom check (full stack) | $(date) ==="

# --- Call 1: 0-shot tasks (matches GPT-2 baseline fewshot setting) ---
# HellaSwag + LAMBADA already confirmed; ARC-Easy newly cached
lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
    --tasks hellaswag,lambada_openai,arc_easy \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$OUT/0shot"

echo "--- 0-shot complete | $(date) ---"

# --- Call 2: MMLU 5-shot (standard; Keisha et al. comparability) ---
# Subtask configs generated via _generate_configs.py; group task 'mmlu' should resolve
lm_eval --model hf \
    --model_args "pretrained=$MODEL,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path "$OUT/mmlu_5shot"

echo "--- MMLU 5-shot complete | $(date) ---"

echo "=== headroom check complete | $(date) ==="
echo ""
echo "Decision rules:"
echo "  HellaSwag acc_norm  >= 0.65  -> confirmed headroom (baseline: 0.713)"
echo "  LAMBADA acc         >= 0.60  -> confirmed headroom (baseline: 0.674)"
echo "  ARC-Easy acc_norm   >= 0.70  -> include in Phase 3 stack"
echo "  ARC-Easy acc_norm   <  0.60  -> check floor before proceeding"
echo "  MMLU acc (5-shot)   >= 0.40  -> include in Phase 3 stack"
echo "  MMLU acc (5-shot)   <  0.30  -> near random; reconsider"
