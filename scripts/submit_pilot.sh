#!/bin/bash
# Submit the 3-generation pilot as a dependency chain (each gen needs the previous
# gen's synthetic corpus). Usage:  bash scripts/submit_pilot.sh [SEED]
set -e
SEED=${1:-42}
cd "$HOME/modelcollapse"

JID=$(sbatch --parsable slurm/run_generation.sbatch 0 $SEED)
echo "gen0 -> job $JID"
for G in 1 2; do
    JID=$(sbatch --parsable --dependency=afterok:$JID slurm/run_generation.sbatch $G $SEED)
    echo "gen$G -> job $JID (afterok previous)"
done
echo "pilot chain submitted. Monitor: squeue -u cyrh"
