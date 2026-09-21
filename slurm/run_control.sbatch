#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=control-gen
#SBATCH --output=logs/control/%x-%j.out

# Phase 5.2 NEGATIVE CONTROL — one generation step g.
# Identical to run_generation.sbatch except the corpus: generation g fine-tunes the
# pristine base on a FRESH, disjoint REAL WikiText-103 slice (built by
# build_fresh_slices.py) instead of generation g-1's synthetic output. Fine-tuning
# steps, eval battery and frozen eval slices are unchanged, so the only manipulated
# variable is synthetic-vs-real.
#
# Generation 0 is NOT run here: it is byte-identical to pilot_gpt2m's generation 0
# (same corpus, same seed, same base) and is symlinked in. Shared by construction.
#
# Usage:  sbatch slurm/run_control.sbatch <GEN 1-7> <SEED>

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
GEN=${1:?usage: sbatch run_control.sbatch GEN SEED}
SEED=${2:?usage: sbatch run_control.sbatch GEN SEED}
[ "$GEN" -ge 1 ] || { echo "gen 0 is shared with pilot_gpt2m (symlinked); start at gen 1"; exit 1; }

cd "$HOME/modelcollapse"
source scripts/narval_env.sh
mkdir -p logs/control
# Scale-parameterised. Defaults reproduce the 345M arm byte-for-byte in behaviour.
# 1.7B: CFG=configs/control_smollm2.yaml ARM=control_smollm2 TASKS=... sbatch ...
CFG=${CFG:-configs/control_gpt2m.yaml}
ARM=${ARM:-control_gpt2m}
SLICES=${SLICES:-$SCRATCH/collapse/fresh_slices_gpt2m}
TASKS=${TASKS:-hellaswag,lambada_openai}
RUN_ROOT="$SCRATCH/collapse/$ARM"
CORPUS="$SLICES/gen${GEN}_real.jsonl"

[ -f "$CORPUS" ] || { echo "missing fresh slice $CORPUS — run build_fresh_slices.py"; exit 1; }
# integrity: slice must match the frozen manifest md5
python - "$CORPUS" << 'PY'
import hashlib, json, sys, os
p = sys.argv[1]
man = json.load(open(os.path.join(os.path.dirname(p), "manifest.json")))
want = next(s["md5"] for s in man["slices"] if s["path"].endswith(os.path.basename(p)))
h = hashlib.md5(open(p,"rb").read()).hexdigest()
assert h == want, f"md5 mismatch for {p}: {h} != {want}"
print(f"slice md5 OK {h}")
PY

echo "=== CONTROL $ARM gen $GEN seed $SEED | cfg $CFG | corpus $CORPUS ==="

python src/finetune.py --config $CFG --generation $GEN --corpus "$CORPUS" --seed $SEED
python src/generate.py --config $CFG --generation $GEN --mode eval   --seed $SEED
python src/metrics.py  --config $CFG --generation $GEN --seed $SEED

MODEL_DIR="$RUN_ROOT/seed$SEED/gen$GEN/model"
# EXTRA_MODEL_ARGS: 1.7B passes ",trust_remote_code=True" to match run_smollm2.sbatch.
# BENCH_SUBDIR: the 1.7B twin writes benchmarks/0shot; 345M arms write benchmarks/ flat.
# Both default to the 345M behaviour.
EXTRA_MODEL_ARGS=${EXTRA_MODEL_ARGS:-}
BENCH_SUBDIR=${BENCH_SUBDIR:-}
lm_eval --model hf \
    --model_args "pretrained=$MODEL_DIR,dtype=bfloat16$EXTRA_MODEL_ARGS" \
    --tasks "$TASKS" \
    --num_fewshot 0 \
    --batch_size auto \
    --output_path "$RUN_ROOT/seed$SEED/gen$GEN/benchmarks$BENCH_SUBDIR"

echo "=== CONTROL gen $GEN seed $SEED complete ==="
