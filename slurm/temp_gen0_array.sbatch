#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=temp-gen0
#SBATCH --output=logs/misc/%x-%A_%a.out
#SBATCH --exclude=ng11002
#SBATCH --array=1-12%6

set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" slurm/temp_gen0_manifest.txt)
read -r CFG GEN SEED <<< "$LINE"
[ -f "$CFG" ] || { echo "ERROR: CFG '$CFG' not a file"; exit 1; }
[ "$GEN" = "0" ] || { echo "ERROR: control must be gen 0, got '$GEN'"; exit 1; }

RUN=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['run_name'])" "$CFG")
GD="$SCRATCH/collapse/$RUN/seed$SEED/gen0"
TSRC="$SCRATCH/collapse/$RUN/seed$SEED/gen7/temperature.json"
[ -d "$GD/model" ] || { echo "ERROR: no gen0 model at $GD/model"; exit 1; }
[ -f "$TSRC" ]     || { echo "ERROR: no gen7 temperature.json at $TSRC"; exit 1; }

# the seed's OWN gen-7 T*, applied to the gen-0 checkpoint
T=$(python -c "import json;print(json.load(open('$TSRC'))['T_star'])")
echo "=== $RUN seed$SEED gen0 | T* from gen7 = $T ==="
mkdir -p "$GD/benchmarks_temp"

for TT in "$T" 1.0; do
    OUT="$GD/benchmarks_temp/results_T${TT}.json"
    python temp_rescale.py "$GD/model" "$TT" \
        --tasks arc_easy,hellaswag \
        --out "$OUT"
    [ -s "$OUT" ] || { echo "ERROR: empty/missing $OUT"; exit 1; }
done
echo "=== done ==="
