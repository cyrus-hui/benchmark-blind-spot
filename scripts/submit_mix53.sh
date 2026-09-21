#!/bin/bash
# submit_mix53.sh — Phase 5.3 afterok chains. Run from ~/modelcollapse.
#   PILOT=1 DRY=1 scripts/submit_mix53.sh   # print the one pilot cell (mix25_topk s43 g1)
#   PILOT=1       scripts/submit_mix53.sh   # submit it; gate it with check_cell.sh
#   DRY=1         scripts/submit_mix53.sh   # print all chains
#                 scripts/submit_mix53.sh   # submit all chains
# A cell whose metrics.json already exists is skipped, and the next cell in that
# chain is submitted with no dependency (so the gated pilot is not rerun).
# A cell that exists WITHOUT metrics.json is refused: rm -rf it first (Sept 15 rule).
set -u
RATIOS=${RATIOS:-"25 50 75"}; DECODERS=${DECODERS:-"topk greedy"}; SEEDS=${SEEDS:-"43 44 45"}
TIME=${TIME:-01:00:00}; MEM=${MEM:-16G}; EXCLUDE=${EXCLUDE:-ng11002,ng11003}
DRY=${DRY:-0}; PILOT=${PILOT:-0}
[ "$PILOT" = 1 ] && { RATIOS=25; DECODERS=topk; SEEDS=43; GENS=1; }
GENS=${GENS:-"1 2 3 4 5 6 7"}
[ -n "${SCRATCH:-}" ] || { echo "FAIL \$SCRATCH unset"; exit 1; }
mkdir -p logs/mix53
# ---- preflight: every refusal happens here, before any sbatch call ----
BAD=0
for r in $RATIOS; do for d in $DECODERS; do
  ARM=mix${r}_${d}; ROOT=$SCRATCH/collapse/$ARM
  [ -f configs/$ARM.yaml ] || { echo "FAIL missing configs/$ARM.yaml"; BAD=1; }
  for s in $SEEDS; do
    [ -L "$ROOT/seed$s/gen0" ] || { echo "FAIL $ROOT/seed$s/gen0 is not a symlink"; BAD=1; }
    for g in $GENS; do C=$ROOT/seed$s/gen$g
      [ -e "$C" ] && [ ! -s "$C/metrics.json" ] && { echo "FAIL $C exists without metrics.json — rm -rf it first"; BAD=1; }
    done
  done
done; done
[ $BAD = 0 ] || { echo "PREFLIGHT FAILED — nothing submitted"; exit 1; }
N=0
for r in $RATIOS; do for d in $DECODERS; do
  ARM=mix${r}_${d}; CFG=configs/$ARM.yaml; ROOT=$SCRATCH/collapse/$ARM
  for s in $SEEDS; do
    PREV_JOB=""
    for g in $GENS; do
      CELL=$ROOT/seed$s/gen$g
      if [ -s "$CELL/metrics.json" ]; then echo "SKIP $ARM s$s g$g (metrics.json present)"; PREV_JOB=""; continue; fi
      DEP=(); [ -n "$PREV_JOB" ] && DEP=(--dependency=afterok:$PREV_JOB)
      CMD=(sbatch --parsable --time=$TIME --mem=$MEM --exclude=$EXCLUDE
           --job-name=${ARM}-s${s}-g${g} --output=logs/mix53/%x-%j.out
           --export=ALL,CFG=$CFG ${DEP[@]+"${DEP[@]}"} slurm/run_ablation.sbatch $g $s)
      if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; PREV_JOB="DRY$N"
      else
        J=$("${CMD[@]}") || { echo "FAIL sbatch for $ARM s$s g$g"; exit 1; }
        echo "$J $ARM s$s g$g ${PREV_JOB:+afterok:$PREV_JOB}" | tee -a logs/mix53/submitted.txt
        PREV_JOB=$J
      fi
      N=$((N+1))
    done
  done
done; done
echo "--- $N cell(s) $([ "$DRY" = 1 ] && echo printed || echo submitted)"
