#!/bin/bash
# Phase 5.2 negative control at 1.7B. Every setting below mirrors slurm/run_smollm2.sbatch,
# which produced the comparator arm smollm2_1.7b. Do not diverge without recording why.
#   full arm:   bash scripts/submit_control_1p7b.sh 43 44 45 46 47
#   one cell:   GENS=1 bash scripts/submit_control_1p7b.sh 43
# MMLU is deliberately NOT run: 57 subtasks at 5-shot, and it is not in the 5.2
# pre-registered rule, which reads Delta_cal only. Recorded as a decision.
set -e
export ARM=control_smollm2
export CFG=configs/control_smollm2.yaml
export TASKS=hellaswag,lambada_openai,arc_easy
export EXTRA_MODEL_ARGS=",trust_remote_code=True"
export BENCH_SUBDIR=/0shot
export MEM=${MEM:-32G}
export CPUS=${CPUS:-2}
export TIME=${TIME:-01:30:00}
export JOBNAME=control17b-gen
exec bash "$HOME/modelcollapse/scripts/submit_control.sh" "$@"
