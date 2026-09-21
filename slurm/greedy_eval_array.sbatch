#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=0:45:00
#SBATCH --array=1-24%12
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/greedy_eval/greedy_eval-%A_%a.out
# Eval-only greedy pass over control_gpt2m checkpoints (5.4 greedy outcome-label tau).
# Writes ONLY under $SCRATCH/collapse/control_gpt2m_greedyeval. gen0 = gpt2m_greedy gen0 weights (canary).
set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh

CFG=configs/control_gpt2m_greedyeval.yaml
SHADOW="$SCRATCH/collapse/control_gpt2m_greedyeval"
IDX=$((SLURM_ARRAY_TASK_ID - 1)); SEED=$((43 + IDX / 8)); GEN=$((IDX % 8))
D="$SHADOW/seed$SEED/gen$GEN"
echo "[task $SLURM_ARRAY_TASK_ID] seed=$SEED gen=$GEN dir=$D"

grep -qx 'run_name: control_gpt2m_greedyeval' "$CFG" || { echo "REFUSE: run_name"; exit 1; }
grep -qE '^\s*strategy: greedy' "$CFG"               || { echo "REFUSE: strategy"; exit 1; }
[ -d "$D" ] && [ ! -L "$D" ]                         || { echo "REFUSE: gen dir missing or symlinked"; exit 1; }
[[ "$(readlink -f "$D")" == "$(readlink -f "$SHADOW")"/* ]] || { echo "REFUSE: gen dir escapes shadow"; exit 1; }
test -f "$D/model/config.json"                       || { echo "MISSING model"; exit 1; }
echo "model -> $(readlink -f "$D/model")"

python src/generate.py --config $CFG --generation $GEN --mode eval --seed $SEED
python src/metrics.py  --config $CFG --generation $GEN --seed $SEED
echo "=== done | $(date) ==="
