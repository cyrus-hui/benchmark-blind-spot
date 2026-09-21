#!/bin/bash
# Phase 5.2 negative control. Generations are INDEPENDENT (each fine-tunes base on its own
# fresh real slice), so no dependency chain — all jobs run in parallel.
#
#   345M:  bash scripts/submit_control.sh 43 44 45 46
#   1.7B:  bash scripts/submit_control_1p7b.sh 43 44 45 46 47
set -e
cd "$HOME/modelcollapse"
SEEDS=("${@:-43 44 45 46}")
GENS=${GENS:-$(seq 1 7)}

export ARM=${ARM:-control_gpt2m}
export CFG=${CFG:-configs/control_gpt2m.yaml}
export SLICES=${SLICES:-$SCRATCH/collapse/fresh_slices_gpt2m}
export TASKS=${TASKS:-hellaswag,lambada_openai}
export EXTRA_MODEL_ARGS=${EXTRA_MODEL_ARGS:-}
export BENCH_SUBDIR=${BENCH_SUBDIR:-}

OVERRIDE=(--job-name="${JOBNAME:-${ARM}-gen}")
for v in TIME:--time MEM:--mem GRES:--gres CPUS:--cpus-per-task; do
  n=${v%%:*}; f=${v##*:}
  [ -n "${!n:-}" ] && OVERRIDE+=("$f=${!n}")
done

echo "arm=$ARM cfg=$CFG tasks=$TASKS bench_subdir='$BENCH_SUBDIR' extra='$EXTRA_MODEL_ARGS'"
echo "seeds=${SEEDS[*]} gens=$(echo $GENS | tr '\n' ' ')"
echo "overrides: ${OVERRIDE[*]}"
N=0
for SEED in "${SEEDS[@]}"; do
  for G in $GENS; do
    JID=$(sbatch --parsable "${OVERRIDE[@]}" slurm/run_control.sbatch $G $SEED)
    echo "$ARM seed$SEED gen$G -> job $JID"; N=$((N+1))
  done
done
echo "Submitted $N independent jobs. Monitor: squeue -u cyrh"
