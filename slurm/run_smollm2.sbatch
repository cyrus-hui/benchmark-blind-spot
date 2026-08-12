#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=smollm2-gen
#SBATCH --output=logs/misc/%x-%j.out

# SmolLM2-1.7B generation step g:
#   1. fine-tune base on corpus_g          (real corpus if g=0, else gen{g-1} synthetic)
#   2. generate eval completions           (fixed prompt set, n=2000 for MAUVE)
#   3. compute metrics                     (ppl, distinct2, embvar, KL, MAUVE)
#   4. run benchmarks                      (HellaSwag + LAMBADA + ARC-Easy 0-shot;
#                                           MMLU 5-shot in separate lm-eval call)
#   5. generate synthetic corpus for g+1
#
# Usage:   sbatch slurm/run_smollm2.sbatch <GEN> [SEED]
# Example: sbatch slurm/run_smollm2.sbatch 0 43
#
# NOTE: time=12:00:00 (vs 6:00:00 for GPT-2) — 1.7B full FT is ~4x slower per generation.
# Monitor first gen-0 wall time and adjust if needed before submitting full chains.

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

GEN=${1:?usage: sbatch run_smollm2.sbatch GEN [SEED]}
SEED=${2:-43}

cd "$HOME/modelcollapse"
source scripts/narval_env.sh

CFG=configs/smollm2_1.7b.yaml
RUN_ROOT="$SCRATCH/collapse/smollm2_1.7b"
MODEL_BASE="HuggingFaceTB/SmolLM2-1.7B"

# corpus for this generation
if [ "$GEN" -eq 0 ]; then
    CORPUS="$RUN_ROOT/real_train.jsonl"
else
    PREV=$((GEN - 1))
    CORPUS="$RUN_ROOT/seed$SEED/gen$PREV/synthetic_corpus.jsonl"
fi
[ -f "$CORPUS" ] || { echo "ERROR: missing corpus $CORPUS — run previous generation first"; exit 1; }

echo "=== SmolLM2-1.7B | gen $GEN seed $SEED | corpus: $CORPUS | $(date) ==="

# Step 1: fine-tune
python src/finetune.py --config $CFG --generation $GEN --corpus "$CORPUS" --seed $SEED

# Step 2+3: generate eval completions and compute metrics
# n_eval_prompts=2000 for MAUVE (overrides config default of 1000 for eval completions;
# matches the mauve_n_eval_prompts field — confirm src/generate.py reads this)
python src/generate.py --config $CFG --generation $GEN --mode eval   --seed $SEED
python src/metrics.py  --config $CFG --generation $GEN --seed $SEED

MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
BENCH_DIR="$RUN_ROOT/seed$SEED/gen$GEN/benchmarks"

# Step 4a: 0-shot benchmarks (HellaSwag + LAMBADA + ARC-Easy)
# Matches GPT-2 fewshot setting for HellaSwag and LAMBADA;
# ARC-Easy added as knowledge-benchmark proxy (MMLU dropped at GPT-2 phase due to
# lm-eval config issue; ARC-Easy substituted; MMLU now restored below)
lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,trust_remote_code=True" \
    --tasks hellaswag,lambada_openai,arc_easy \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$BENCH_DIR/0shot"

# Step 4b: MMLU 5-shot (standard; Keisha et al. comparability)
# Subtask configs in lm_eval/tasks/mmlu/default/ (generated via _generate_configs.py)
# MMLU_TASK=mmlu confirmed by headroom job 63143870 (0-shot call); subtask cache populated on login node June 17 2026 — update if different
lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,trust_remote_code=True" \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path "$BENCH_DIR/mmlu_5shot"

# Step 5: generate synthetic corpus for next generation
python src/generate.py --config $CFG --generation $GEN --mode corpus --seed $SEED

echo "=== SmolLM2-1.7B | gen $GEN seed $SEED complete | $(date) ==="
echo "=== Submit gen $((GEN+1)) next ==="
