#!/bin/bash
# Submit SmolLM2-1.7B gen-1 through gen-7 dependency chains for all 5 seeds in parallel.
# Gen-0 must be complete and rerun_eval_metrics_gen0 must have finished before running this.
#
# Usage: bash scripts/submit_smollm2_chains.sh [START_GEN] [SEEDS...]
#   START_GEN defaults to 1
#   SEEDS defaults to "43 44 45 46 47"
#
# Examples:
#   bash scripts/submit_smollm2_chains.sh          # gen 1-7, all 5 seeds
#   bash scripts/submit_smollm2_chains.sh 2        # gen 2-7, all 5 seeds (resume after failure)
#   bash scripts/submit_smollm2_chains.sh 1 43 44  # gen 1-7, seeds 43+44 only

START_GEN=${1:-1}
shift || true
SEEDS=${@:-"43 44 45 46 47"}
END_GEN=${END_GEN:-7}
SBATCH=slurm/run_smollm2.sbatch

cd "$HOME/modelcollapse" || { echo "ERROR: must run from ~/modelcollapse or adjust cd"; exit 1; }

echo "=== SmolLM2-1.7B chain submission | gens $START_GEN-$END_GEN | seeds $SEEDS | $(date) ==="

for SEED in $SEEDS; do
    echo "--- seed $SEED ---"
    PREV_JOB=""
    for GEN in $(seq $START_GEN $END_GEN); do
        if [ -z "$PREV_JOB" ]; then
            JOB=$(sbatch --parsable $SBATCH $GEN $SEED)
        else
            JOB=$(sbatch --parsable --dependency=afterok:$PREV_JOB $SBATCH $GEN $SEED)
        fi
        echo "  gen $GEN -> job $JOB"
        PREV_JOB=$JOB
    done
done

echo "=== all chains submitted | $(date) ==="
echo "Monitor with: squeue -u $USER -o '%.10i %.12j %.8T %.6M %R'"
