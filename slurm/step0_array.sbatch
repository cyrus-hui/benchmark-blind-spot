#!/bin/bash
#SBATCH --job-name=step0_7p1
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=0:30:00
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/step0/%x-%A_%a.out
# 7.1 step 0 scorer array (decisions.md §41). One manifest row per task; row 1 is the header.
# Submit from ~/modelcollapse. Pilot: --array=1. Rest only after scripts/step0_gate.py --pilot.
set -eo pipefail
REPO=${REPO:-$HOME/modelcollapse}
cd "$REPO"
if [ -z "${STEP0_NOENV:-}" ]; then
  module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
  source scripts/narval_env.sh
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
fi
set -u
MAN=${MAN:-slurm/step0_manifest.tsv}
OUT=${OUT:-$SCRATCH/collapse/step0_7p1/scores}
TEXT_KEY=${TEXT_KEY:-text}
# Pin: the scorer and manifest must be committed and unmodified.
for f in src/step0_score.py "$MAN"; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "REFUSE: $f not tracked"; exit 2; }
  git diff --quiet HEAD -- "$f" || { echo "REFUSE: $f differs from HEAD"; exit 2; }
done
line=$(awk -v n=$((SLURM_ARRAY_TASK_ID + 1)) 'NR==n' "$MAN")
[ -n "$line" ] || { echo "REFUSE: no manifest row for task $SLURM_ARRAY_TASK_ID"; exit 2; }
IFS=$'\t' read -r task kind corpus scorer real_ref synth_ref expect_synth expect_rows k3 <<< "$line"
args=(--task "$task" --kind "$kind" --corpus "$corpus" --scorer "$scorer" --out-dir "$OUT"
      --expect-rows "$expect_rows" --text-key "$TEXT_KEY")
[ "$real_ref" != "-" ] && args+=(--real-ref "$real_ref")
[ "$synth_ref" != "-" ] && args+=(--synth-ref "$synth_ref")
[ "$expect_synth" != "0" ] && args+=(--expect-synth "$expect_synth")
[ "$k3" = "1" ] && args+=(--enforce-k3)
echo "task $SLURM_ARRAY_TASK_ID: $task  head $(git rev-parse --short HEAD)  node $(hostname)"
python src/step0_score.py "${args[@]}"
