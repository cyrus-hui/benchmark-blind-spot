#!/bin/bash
# submit_filt71.sh — Phase 7.1 filter chains (decisions.md §62, §64). Run from ~/modelcollapse.
#   PILOT=1 DRY=1 scripts/submit_filt71.sh          # print the pilot: greedy F(0.25) s43 g1
#   PILOT=1 GRES=gpu:1 scripts/submit_filt71.sh     # submit it; then scripts/filt71_gate.py --pilot
#   DRY=1 GRES=... TIME=... scripts/submit_filt71.sh   # print all 36 chains
#         GRES=... TIME=... scripts/submit_filt71.sh   # submit them
# GRES is the GPU request, e.g. gpu:1 (full A100) or gpu:a100_3g.20gb:1 (MIG). Use the SAME value
# as the pilot for every chain (§64.6); it is recorded in logs/filt71/submitted.txt.
# TIME has no default outside the pilot: set it from the pilot's observed elapsed time.
# Dependencies: F(g) afterok F(g-1). R(g) afterok R(g-1) AND F(g), because R reads its k from
# F's manifest row for the same generation (§64.1).
# A cell whose metrics.json exists is skipped. A cell dir without metrics.json is refused: rm -rf it.
set -u
QS=${QS:-"10 25 50"}; DECODERS=${DECODERS:-"greedy topk"}; SEEDS=${SEEDS:-"43 44 45"}
ARMS="F R"; GRES=${GRES:-}; TIME=${TIME:-}; MEM=${MEM:-16G}; EXCLUDE=${EXCLUDE:-ng11002,ng11003}
DRY=${DRY:-0}; PILOT=${PILOT:-0}
if [ "$PILOT" = 1 ]; then QS=25; DECODERS=greedy; SEEDS=43; GENS=1; ARMS=F; TIME=${TIME:-01:00:00}; fi
GENS=${GENS:-"1 2 3 4 5 6 7"}
[ -n "${SCRATCH:-}" ] || { echo "FAIL \$SCRATCH unset"; exit 1; }
[ -n "$GRES" ] || { echo "FAIL set GRES (e.g. gpu:1 or gpu:a100_3g.20gb:1)"; exit 1; }
[ -n "$TIME" ] || { echo "FAIL set TIME from the pilot's observed elapsed time"; exit 1; }
mkdir -p logs/filt71
# ---- preflight: every refusal happens here, before any sbatch call ----
BAD=0
for f in src/filter_corpus.py src/step0_score.py slurm/run_ablation.sbatch scripts/filt71_gate.py scripts/filt71_analysis.py; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
    || { echo "FAIL $f is not committed and unmodified"; BAD=1; }
done
git branch -r --contains HEAD 2>/dev/null | grep -q 'origin/' \
  || { echo "FAIL HEAD $(git rev-parse --short HEAD) is not on origin — push before the pilot (§62.9)"; BAD=1; }
grep -q 'Phase 7.1 corpus filter' slurm/run_ablation.sbatch || { echo "FAIL run_ablation.sbatch has no filter hook"; BAD=1; }
if [ "$PILOT" != 1 ]; then
  P=$SCRATCH/collapse/filt71_F25_greedy/seed43/gen1
  [ -s "$P/metrics.json" ] || { echo "FAIL pilot cell $P has no metrics.json — run and gate the pilot first"; BAD=1; }
  [ -s logs/filt71/gate_pilot.json ] && grep -q '"verdict": "PASS"' logs/filt71/gate_pilot.json \
    || { echo "FAIL logs/filt71/gate_pilot.json is not a PASS — run scripts/filt71_gate.py --pilot"; BAD=1; }
fi
for q in $QS; do for d in $DECODERS; do for a in $ARMS; do
  ARM=filt71_${a}${q}_${d}; ROOT=$SCRATCH/collapse/$ARM
  [ -f configs/$ARM.yaml ] || { echo "FAIL missing configs/$ARM.yaml"; BAD=1; }
  for s in $SEEDS; do
    [ -L "$ROOT/seed$s/gen0" ] || { echo "FAIL $ROOT/seed$s/gen0 is not a symlink"; BAD=1; }
    for g in $GENS; do C=$ROOT/seed$s/gen$g
      [ -e "$C" ] && [ ! -s "$C/metrics.json" ] && { echo "FAIL $C exists without metrics.json — rm -rf it first"; BAD=1; }
    done
  done
done; done; done
[ $BAD = 0 ] || { echo "PREFLIGHT FAILED — nothing submitted"; exit 1; }

HEAD=$(git rev-parse --short HEAD); N=0
sub(){ # $1 arm-name $2 seed $3 gen $4.. deps
  local ARM=$1 s=$2 g=$3; shift 3
  local DEPS=$(IFS=:; echo "$*"); local DEP=(); [ -n "$DEPS" ] && DEP=(--dependency=afterok:$DEPS)
  local CMD=(sbatch --parsable --gres=$GRES --time=$TIME --mem=$MEM --exclude=$EXCLUDE --mail-type=FAIL
             --job-name=${ARM}-s${s}-g${g} --output=logs/filt71/%x-%j.out
             --export=ALL,CFG=configs/$ARM.yaml ${DEP[@]+"${DEP[@]}"} slurm/run_ablation.sbatch $g $s)
  if [ "$DRY" = 1 ]; then echo "${CMD[*]}" >&2; echo "DRY$N"
  else
    local J; J=$("${CMD[@]}") || { echo "FAIL sbatch for $ARM s$s g$g" >&2; exit 1; }
    echo "$J $ARM s$s g$g gres=$GRES time=$TIME head=$HEAD ${DEPS:+afterok:$DEPS}" | tee -a logs/filt71/submitted.txt >&2
    echo "$J"
  fi
}
for q in $QS; do for d in $DECODERS; do for s in $SEEDS; do
  PF=""; PR=""
  for g in $GENS; do
    FA=filt71_F${q}_${d}; RA=filt71_R${q}_${d}; JF=""
    if [ -s "$SCRATCH/collapse/$FA/seed$s/gen$g/metrics.json" ]; then echo "SKIP $FA s$s g$g (metrics.json present)"; PF=""
    else JF=$(sub $FA $s $g $PF) || exit 1; PF=$JF; N=$((N+1)); fi
    case " $ARMS " in *" R "*)
      if [ -s "$SCRATCH/collapse/$RA/seed$s/gen$g/metrics.json" ]; then echo "SKIP $RA s$s g$g (metrics.json present)"; PR=""
      else JR=$(sub $RA $s $g $PR $JF) || exit 1; PR=$JR; N=$((N+1)); fi;;
    esac
  done
done; done; done
echo "--- $N cell(s) $([ "$DRY" = 1 ] && echo printed || echo submitted)"
