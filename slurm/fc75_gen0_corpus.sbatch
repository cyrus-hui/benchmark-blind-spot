#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --array=1-9%9
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/fc75/fc75_g0-%A_%a.out
# Phase 7.5 (decisions.md sec. 73): gen-0 TRAINING corpora at a corpus temperature.
# Generation only: no training, no eval completions, no metrics. Writes one synthetic_corpus.jsonl under
# $SCRATCH/collapse/fc75_g0_<T>/seed<S>/gen0 and nothing else. Tasks 1-3 t075, 4-6 t050, 7-9 t125; seeds 43-45.
set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh

TS=(t075 t050 t125)
IDX=$((SLURM_ARRAY_TASK_ID - 1)); T=${TS[$((IDX / 3))]}; SEED=$((43 + IDX % 3))
ARM=fc75_g0_$T; CFG=configs/$ARM.yaml; D="$SCRATCH/collapse/$ARM/seed$SEED/gen0"
echo "[task $SLURM_ARRAY_TASK_ID] arm=$ARM seed=$SEED dir=$D head=$(git rev-parse --short HEAD)"

grep -qx "run_name: $ARM" "$CFG"                 || { echo "REFUSE: run_name"; exit 1; }
grep -qE '^\s*corpus_temperature:' "$CFG"        || { echo "REFUSE: config has no corpus_temperature"; exit 1; }
grep -qE '^\s*mix_ratio:' "$CFG"                 && { echo "REFUSE: a gen-0 corpus config must not mix"; exit 1; }
[ -d "$D" ] && [ ! -L "$D" ]                     || { echo "REFUSE: gen0 dir missing or symlinked"; exit 1; }
[ -f "$D/model/config.json" ]                    || { echo "MISSING model"; exit 1; }
[ ! -e "$D/synthetic_corpus.jsonl" ]             || { echo "REFUSE: corpus exists; rm it to regenerate"; exit 1; }
git diff --quiet HEAD -- src/generate.py "$CFG"  || { echo "REFUSE: src/generate.py or $CFG differs from HEAD"; exit 1; }
echo "model -> $(readlink -f "$D/model")"

python src/generate.py --config $CFG --generation 0 --mode corpus --seed $SEED

NL=$(wc -l < "$D/synthetic_corpus.jsonl")
[ "$NL" -eq 18686 ] || { echo "FAIL: corpus has $NL lines, expected 18686"; exit 1; }
echo "g0 corpus $D/synthetic_corpus.jsonl lines $NL md5 $(md5sum "$D/synthetic_corpus.jsonl" | cut -d' ' -f1)"
echo "=== g0 corpus done | $(date) ==="
