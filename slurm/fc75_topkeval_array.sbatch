#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --array=1-13%13
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/fc75/fc75_topkeval-%A_%a.out
# Phase 7.5 (decisions.md sec. 73), condition type C: eval-only TOP-K pass over greedy-corpus checkpoints.
# One "ARM SEED GEN" line of slurm/fc75_topkeval_tasks.txt per task; task 1 is the gen-0 canary.
# Writes ONLY under $SCRATCH/collapse/<arm>_topkeval. Submit with --array=1-<number of task lines>.
set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/fc75_topkeval_tasks.txt)
read -r ARM SEED GEN <<< "$LINE"
[ -n "${GEN:-}" ] || { echo "REFUSE: no task line $SLURM_ARRAY_TASK_ID"; exit 1; }
CFG=configs/$ARM.yaml; SHADOW="$SCRATCH/collapse/$ARM"; D="$SHADOW/seed$SEED/gen$GEN"
echo "[task $SLURM_ARRAY_TASK_ID] arm=$ARM seed=$SEED gen=$GEN dir=$D head=$(git rev-parse --short HEAD)"

[[ "$ARM" == *_topkeval ]]                        || { echo "REFUSE: not a top-k-eval shadow"; exit 1; }
grep -qx "run_name: $ARM" "$CFG"                  || { echo "REFUSE: run_name"; exit 1; }
grep -qE '^\s*strategy: top_k' "$CFG"             || { echo "REFUSE: strategy"; exit 1; }
grep -qE '^\s*temperature: 1\.0' "$CFG"           || { echo "REFUSE: temperature"; exit 1; }
grep -qE '^\s*(corpus_temperature|mix_ratio):' "$CFG" && { echo "REFUSE: shadow config must not mix or set a corpus temperature"; exit 1; }
[ -d "$D" ] && [ ! -L "$D" ]                      || { echo "REFUSE: gen dir missing or symlinked"; exit 1; }
[[ "$(readlink -f "$D")" == "$(readlink -f "$SHADOW")"/* ]] || { echo "REFUSE: gen dir escapes the shadow"; exit 1; }
[ -f "$D/model/config.json" ]                     || { echo "MISSING model"; exit 1; }
[ ! -e "$D/metrics.json" ]                        || { echo "REFUSE: metrics.json exists; this pass never overwrites"; exit 1; }
git diff --quiet HEAD -- src/generate.py src/metrics.py "$CFG" || { echo "REFUSE: code or config differs from HEAD"; exit 1; }
echo "model -> $(readlink -f "$D/model")"

python src/generate.py --config $CFG --generation $GEN --mode eval --seed $SEED
python src/metrics.py  --config $CFG --generation $GEN --seed $SEED
echo "=== topkeval done | $(date) ==="
