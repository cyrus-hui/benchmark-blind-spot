#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=collapse-abl
#SBATCH --output=logs/misc/%x-%j.out

# One full generation step g:
#   1. fine-tune base on corpus_g          (real corpus if g=0, else gen{g-1} synthetic)
#   2. generate eval completions           (fixed prompt set)
#   3. compute 5 metrics                   (ppl, distinct2, embvar, KL, MAUVE)
#   4. run benchmarks                      (HellaSwag + MMLU via lm-eval)
#   5. generate synthetic corpus for g+1
#
# Usage:  sbatch slurm/run_generation.sbatch <GEN> [SEED]
# Example pilot:  for g in 0 1 2; do submit g sequentially (each depends on previous)

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
GEN=${1:?usage: sbatch run_generation.sbatch GEN [SEED]}
SEED=${2:-42}

cd "$HOME/modelcollapse"
source scripts/narval_env.sh
CFG=${CFG:?set CFG=configs/....yaml}
RUN_NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))[chr(114)+chr(117)+chr(110)+chr(95)+chr(110)+chr(97)+chr(109)+chr(101)])" "$CFG")
RUN_ROOT="$SCRATCH/collapse/$RUN_NAME"

# corpus for this generation
if [ "$GEN" -eq 0 ]; then
    CORPUS="$RUN_ROOT/real_train.jsonl"
else
    PREV=$((GEN - 1))
    CORPUS="$RUN_ROOT/seed$SEED/gen$PREV/synthetic_corpus.jsonl"
fi
[ -f "$CORPUS" ] || { echo "missing corpus $CORPUS — run previous generation first"; exit 1; }

echo "=== gen $GEN seed $SEED | corpus: $CORPUS ==="

# ---- Phase 5.3 ratio mixing (decisions.md, Sept 15 2026 pre-registration) ----
# No-op unless the config defines mix_ratio: every pre-existing config is unaffected.
PROMPT_ARGS=()
read MIX_RATIO MIX_SLICE_FMT MIX_N < <(python -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]));print(c.get('mix_ratio','-'),c.get('mix_slice_fmt','-'),c.get('mix_expect_chunks','-'))" "$CFG")
if [ "$MIX_RATIO" != "-" ] && [ "$GEN" -gt 0 ]; then
    SLICE=$(printf "$MIX_SLICE_FMT" "$GEN")
    MIXED="$RUN_ROOT/mixed/seed$SEED/gen$GEN.jsonl"
    echo "5.3 mix: ratio=$MIX_RATIO gen=$GEN seed=$SEED slice=$SLICE synth=$CORPUS"
    python src/build_mixed_corpus.py --real-slice "$SLICE" --synth-corpus "$CORPUS" \
        --ratio "$MIX_RATIO" --seed "$SEED" --gen "$GEN" --out "$MIXED" \
        --expect-chunks "$MIX_N" --manifest "$RUN_ROOT/mix_manifest.tsv" --force \
        || { echo "5.3 mix: build_mixed_corpus FAILED"; exit 1; }
    CORPUS="$MIXED"
    PROMPT_ARGS=(--prompt-corpus "$MIXED")
fi
python src/finetune.py --config $CFG --generation $GEN --corpus "$CORPUS" --seed $SEED
python src/generate.py --config $CFG --generation $GEN --mode eval   --seed $SEED
python src/metrics.py  --config $CFG --generation $GEN --seed $SEED

MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16" \
    --tasks hellaswag,lambada_openai \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$RUN_ROOT/seed$SEED/gen$GEN/benchmarks"

python src/generate.py --config $CFG --generation $GEN --mode corpus --seed $SEED ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}

echo "=== gen $GEN complete. Submit gen $((GEN+1)) next. ==="
