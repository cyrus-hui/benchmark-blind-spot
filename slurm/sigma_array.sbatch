#!/bin/bash
#SBATCH --account=def-enaskt_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:40:00
#SBATCH --array=1-16%8
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/sigma/sigma-%A_%a.out
# SIGMA-UB baseline (5.4 baseline b). CPU-only: embeds eval completions, computes Algorithm 1.
# LIST selects the chain list ("ARM SEED" per line). Writes ONLY under $SCRATCH/collapse/sigma. Prints no values.
set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LIST=${LIST:-slurm/sigma_chains_topk.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
read -r ARM SEED <<<"$LINE"
[ -n "${ARM:-}" ] && [ -n "${SEED:-}" ] || { echo "REFUSE: no line $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "[task $SLURM_ARRAY_TASK_ID] list=$LIST arm=$ARM seed=$SEED host=$(hostname) sigma_md5=$(md5sum src/sigma.py | cut -d' ' -f1)"
python src/sigma.py --root "$SCRATCH/collapse" --arm "$ARM" --seed "$SEED" \
  --out-root "$SCRATCH/collapse/sigma" --device cpu
echo "=== done | $(date) ==="
