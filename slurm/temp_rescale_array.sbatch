#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=temp-rescale
#SBATCH --output=logs/misc/%x-%A_%a.out
#SBATCH --exclude=ng11002
#SBATCH --array=1-12%6

set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/temp_manifest.txt)
read -r CFG GEN SEED <<< "$LINE"
[ -f "$CFG" ] || { echo "ERROR: CFG '$CFG' not a file"; exit 1; }

RUN=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CFG")
GD="$SCRATCH/collapse/$RUN/seed$SEED/gen$GEN"
echo "=== $RUN gen$GEN seed$SEED ==="

python solve_temperature.py --config "$CFG" --generation "$GEN" --seed "$SEED"

T=$(python -c "import json;print(json.load(open('$GD/temperature.json'))['T_star'])")
echo "--- T* = $T ---"
mkdir -p "$GD/benchmarks_temp"

# T* and the T=1 control, so the contrast is measured in one run
for TT in "$T" 1.0; do
    python temp_rescale.py "$GD/model" "$TT" \
        --tasks arc_easy,hellaswag \
        --out "$GD/benchmarks_temp/results_T${TT}.json"
done
echo "=== done ==="
