#!/bin/bash
# SmolLM2 greedy ablation: gen-0 post-finetune step, then chained gens 1-7, seeds 43-45.
# Usage: bash scripts/submit_smollm2_greedy.sh [START_GEN] [SEEDS...]
set -euo pipefail
START_GEN=${1:-0}; shift || true
SEEDS=${@:-"43 44 45"}
END_GEN=${END_GEN:-7}
export CFG=configs/smollm2_greedy.yaml
SB=slurm/run_smollm2_ablation.sbatch

cd "$HOME/modelcollapse"
echo "=== smollm2_greedy | gens $START_GEN-$END_GEN | seeds $SEEDS | $(date) ==="
LAST=""
for SEED in $SEEDS; do
    PREV=""
    for GEN in $(seq $START_GEN $END_GEN); do
        if [ -z "$PREV" ]; then
            JOB=$(CFG=$CFG sbatch --parsable $SB $GEN $SEED)
        else
            JOB=$(CFG=$CFG sbatch --parsable --dependency=afterok:$PREV $SB $GEN $SEED)
        fi
        echo "  seed $SEED gen $GEN -> $JOB"
        PREV=$JOB
    done
    LAST=$PREV
done
sbatch --dependency=afterany:$LAST --account=def-enaskt_gpu --time=00:02:00 \
       --mem=1G --job-name=greedy-done --mail-user=cyrhui@gmail.com --mail-type=END \
       --output=logs/misc/%x-%j.out --wrap="echo smollm2_greedy chains finished" >/dev/null
echo "=== submitted | $(date) ==="
