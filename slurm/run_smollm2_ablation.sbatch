#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=smollm2-abl
#SBATCH --output=logs/misc/%x-%j.out
#SBATCH --exclude=ng11002

# Parameterized copy of run_smollm2.sbatch for decoding ablations.
# RUN_ROOT is derived from the config's run_name, never hardcoded.
# Usage: CFG=configs/smollm2_greedy.yaml sbatch slurm/run_smollm2_ablation.sbatch <GEN> [SEED]

set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

GEN=${1:?usage: CFG=... sbatch run_smollm2_ablation.sbatch GEN [SEED]}
SEED=${2:-43}
CFG=${CFG:?CFG must be set to a config PATH, e.g. configs/smollm2_greedy.yaml}

cd "$HOME/modelcollapse"
source scripts/narval_env.sh

[ -f "$CFG" ] || { echo "ERROR: CFG '$CFG' is not a file — must be a path, not a bare name"; exit 1; }
RUN_NAME=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CFG")
RUN_ROOT="$SCRATCH/collapse/$RUN_NAME"

if [ "$GEN" -eq 0 ]; then
    CORPUS="$RUN_ROOT/real_train.jsonl"
else
    CORPUS="$RUN_ROOT/seed$SEED/gen$((GEN-1))/synthetic_corpus.jsonl"
fi
[ -f "$CORPUS" ] || { echo "ERROR: missing corpus $CORPUS"; exit 1; }

echo "=== $RUN_NAME | gen $GEN seed $SEED | corpus: $CORPUS | $(date) ==="

MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
BENCH_DIR="$RUN_ROOT/seed$SEED/gen$GEN/benchmarks"

# gen-0 model is decoder-independent and pre-copied; skip fine-tune if present
if [ "$GEN" -eq 0 ] && [ -d "$MODEL_DIR" ]; then
    echo "--- gen-0 checkpoint present, skipping fine-tune ---"
else
    python src/finetune.py --config "$CFG" --generation "$GEN" --corpus "$CORPUS" --seed "$SEED"
fi

python src/generate.py --config "$CFG" --generation "$GEN" --mode eval --seed "$SEED"
python src/metrics.py  --config "$CFG" --generation "$GEN" --seed "$SEED"

if [ "$GEN" -eq 0 ] && [ -d "$BENCH_DIR" ]; then
    echo "--- gen-0 benchmarks pre-copied (decoder-independent), skipping lm-eval ---"
else
    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,trust_remote_code=True" \
        --tasks hellaswag,lambada_openai,arc_easy \
        --num_fewshot 0 --batch_size auto \
        --output_path "$BENCH_DIR/0shot"

    lm_eval --model hf \
        --model_args "pretrained=$MODEL_DIR,dtype=bfloat16,trust_remote_code=True" \
        --tasks mmlu \
        --num_fewshot 5 --batch_size auto \
        --output_path "$BENCH_DIR/mmlu_5shot"
fi

python src/generate.py --config "$CFG" --generation "$GEN" --mode corpus --seed "$SEED"
echo "=== $RUN_NAME | gen $GEN seed $SEED complete | $(date) ==="
