#!/bin/bash
# Submit full Phase 3 runs: 7 generations x N seeds, each seed as an
# independent dependency chain (gens sequential within seed, seeds parallel).
# Usage:  bash scripts/submit_phase3.sh [SEEDS...]
# Example: bash scripts/submit_phase3.sh 42 43 44 45 46
set -e
SEEDS=("${@:-42 43 44}")
N_GENS=7
cd "$HOME/modelcollapse"

for SEED in "${SEEDS[@]}"; do
    JID=$(sbatch --parsable slurm/run_generation.sbatch 0 $SEED)
    echo "seed$SEED gen0 -> job $JID"
    for G in $(seq 1 $N_GENS); do
        JID=$(sbatch --parsable --dependency=afterok:$JID slurm/run_generation.sbatch $G $SEED)
        echo "seed$SEED gen$G -> job $JID (afterok previous)"
    done
    echo "seed$SEED chain submitted ($(($N_GENS + 1)) jobs)"
done

echo ""
echo "All chains submitted. Monitor: squeue -u cyrh"
