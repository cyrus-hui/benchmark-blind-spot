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
# ---- Phase 7.1 corpus filter (decisions.md §62 + §64, registered Sept 26 2026) ----
# No-op unless the config has a filter: block: every pre-existing config is unaffected.
# Prompts are untouched: corpus mode still draws them from gen g-1's FULL synthetic corpus (§62.4).
read FILT_ARM FILT_Q FILT_DEC FILT_PAIR FILT_K1 < <(python -c "import yaml,sys;f=yaml.safe_load(open(sys.argv[1])).get('filter') or {};print(*(f.get(k,'-') for k in ('arm','q','decoder','pair_root','k1_ref_fmt')))" "$CFG")
if [ "$FILT_ARM" != "-" ] && [ "$GEN" -gt 0 ]; then
    [ "$MIX_RATIO" = "-" ] || { echo "7.1 filter: REFUSE config has both mix_ratio and filter"; exit 1; }
    for f in src/filter_corpus.py src/step0_score.py; do
        git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
            || { echo "7.1 filter: REFUSE $f is not committed and unmodified"; exit 1; }
    done
    FILTERED="$RUN_ROOT/filtered/seed$SEED/gen$GEN.jsonl"
    FARGS=(--arm "$FILT_ARM" --q "$FILT_Q" --decoder "$FILT_DEC" --seed "$SEED" --gen "$GEN"
           --corpus "$CORPUS" --scorer "$RUN_ROOT/seed$SEED/gen0/model" --out "$FILTERED"
           --manifest "$RUN_ROOT/filter_manifest.tsv"
           --scores-out "$RUN_ROOT/filter_scores/seed$SEED/gen$GEN.npz")
    [ "$FILT_ARM" = "R" ] && FARGS+=(--pair-manifest "$FILT_PAIR/filter_manifest.tsv")
    [ "$GEN" -eq 1 ] && FARGS+=(--k1-ref "$(printf "$FILT_K1" "$SEED")")
    echo "7.1 filter: arm=$FILT_ARM q=$FILT_Q dec=$FILT_DEC gen=$GEN seed=$SEED head=$(git rev-parse --short HEAD) synth=$CORPUS"
    python src/filter_corpus.py "${FARGS[@]}" || { echo "7.1 filter: filter_corpus FAILED"; exit 1; }
    CORPUS="$FILTERED"
    echo "7.1 filter: training corpus $CORPUS md5 $(md5sum "$CORPUS" | cut -d' ' -f1) lines $(wc -l < "$CORPUS")"
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
