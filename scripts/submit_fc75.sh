#!/bin/bash
# submit_fc75.sh -- Phase 7.5 jobs (decisions.md sec. 73). Run from ~/modelcollapse.
#   STAGE=g0        scripts/submit_fc75.sh             # 9 gen-0 corpora at the corpus temperatures
#   STAGE=topkeval  scripts/submit_fc75.sh             # top-k eval pass over greedy-corpus checkpoints (type C)
#   STAGE=pilot  GRES=gpu:1 scripts/submit_fc75.sh     # one cell: fc75_r50_t075 s43 g1 (needs gate --g0 PASS)
#   STAGE=chains GRES=gpu:1 TIME=hh:mm:ss scripts/submit_fc75.sh   # the other 146 cells (needs gate --pilot PASS)
#   DRY=1 ...                                          # print, submit nothing
# GRES is the GPU request; use the SAME value for the pilot and every chain. TIME has no default outside the
# pilot: size it from the pilot's observed elapsed time. Every job is recorded in logs/fc75/submitted.txt.
# A cell whose metrics.json exists is skipped. A cell dir without metrics.json is refused: rm -rf it first.
set -u
STAGE=${STAGE:-}; DRY=${DRY:-0}; GRES=${GRES:-}; TIME=${TIME:-}; MEM=${MEM:-16G}
EXCLUDE=${EXCLUDE:-ng11002,ng11003}; SEEDS=${SEEDS:-"43 44 45"}
ARMS="mix10_topk mix90_topk fc75_r25_t075 fc75_r25_t050 fc75_r50_t075 fc75_r50_t050 fc75_r100_t125"
PILOT_ARM=fc75_r50_t075; PILOT_SEED=43
case "$STAGE" in g0|topkeval|pilot|chains) ;; *) echo "FAIL set STAGE to g0, topkeval, pilot or chains"; exit 1;; esac
[ -n "${SCRATCH:-}" ] || { echo "FAIL \$SCRATCH unset"; exit 1; }
mkdir -p logs/fc75
# ---- preflight: every refusal happens here, before any sbatch call ----
BAD=0
TRACKED="src/generate.py src/metrics.py src/finetune.py src/build_mixed_corpus.py src/detectors.py
  slurm/run_ablation.sbatch slurm/fc75_gen0_corpus.sbatch slurm/fc75_topkeval_array.sbatch slurm/fc75_topkeval_tasks.txt
  prepare_7p5.py scripts/submit_fc75.sh scripts/fc75_gate.py scripts/fc75_analysis.py fc75_conditions.json"
for a in $ARMS fc75_g0_t075 fc75_g0_t050 fc75_g0_t125; do TRACKED="$TRACKED configs/$a.yaml"; done
for f in $TRACKED; do
  git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
    || { echo "FAIL $f is not committed and unmodified"; BAD=1; }
done
git branch -r --contains HEAD 2>/dev/null | grep -q 'origin/' \
  || { echo "FAIL HEAD $(git rev-parse --short HEAD) is not on origin: push before the first 7.5 job (sec. 73)"; BAD=1; }
grep -q 'corpus_temperature' src/generate.py || { echo "FAIL src/generate.py has no corpus_temperature patch"; BAD=1; }
gate_ok(){ [ -s "logs/fc75/gate_$1.json" ] && grep -q '"verdict": "PASS"' "logs/fc75/gate_$1.json"; }
HEAD=$(git rev-parse --short HEAD)

if [ "$STAGE" = g0 ]; then
  for t in t075 t050 t125; do for s in 43 44 45; do
    D=$SCRATCH/collapse/fc75_g0_$t/seed$s/gen0
    [ -d "$D" ] && [ ! -L "$D" ] || { echo "FAIL $D missing or symlinked: run prepare_7p5.py"; BAD=1; }
    [ ! -e "$D/synthetic_corpus.jsonl" ] || { echo "FAIL $D/synthetic_corpus.jsonl exists"; BAD=1; }
  done; done
  [ $BAD = 0 ] || { echo "PREFLIGHT FAILED: nothing submitted"; exit 1; }
  CMD=(sbatch --parsable slurm/fc75_gen0_corpus.sbatch)
  if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
  J=$("${CMD[@]}") || { echo "FAIL sbatch"; exit 1; }
  echo "$J fc75_g0 array 1-9 head=$HEAD" | tee -a logs/fc75/submitted.txt; exit 0
fi

if [ "$STAGE" = topkeval ]; then
  NT=$(grep -c . slurm/fc75_topkeval_tasks.txt)
  [ "$NT" -ge 2 ] || { echo "FAIL slurm/fc75_topkeval_tasks.txt has $NT task(s): no C condition is registered"; BAD=1; }
  while read -r a s g; do
    [ -f "$SCRATCH/collapse/$a/seed$s/gen$g/model/config.json" ] || { echo "FAIL missing model for $a s$s g$g"; BAD=1; }
    [ ! -e "$SCRATCH/collapse/$a/seed$s/gen$g/metrics.json" ] || { echo "FAIL metrics.json exists for $a s$s g$g"; BAD=1; }
  done < slurm/fc75_topkeval_tasks.txt
  [ $BAD = 0 ] || { echo "PREFLIGHT FAILED: nothing submitted"; exit 1; }
  CMD=(sbatch --parsable --array=1-$NT%$NT slurm/fc75_topkeval_array.sbatch)
  if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
  J=$("${CMD[@]}") || { echo "FAIL sbatch"; exit 1; }
  echo "$J fc75_topkeval array 1-$NT head=$HEAD" | tee -a logs/fc75/submitted.txt; exit 0
fi

# ---- pilot and chains ----
[ -n "$GRES" ] || { echo "FAIL set GRES (e.g. gpu:1)"; BAD=1; }
GENS="1 2 3 4 5 6 7"
if [ "$STAGE" = pilot ]; then ARMS=$PILOT_ARM; SEEDS=$PILOT_SEED; GENS=1; TIME=${TIME:-01:00:00}; fi
[ -n "$TIME" ] || { echo "FAIL set TIME from the pilot's observed elapsed time"; BAD=1; }
gate_ok g0 || { echo "FAIL logs/fc75/gate_g0.json is not a PASS: run scripts/fc75_gate.py --g0"; BAD=1; }
if [ "$STAGE" = chains ]; then
  gate_ok pilot || { echo "FAIL logs/fc75/gate_pilot.json is not a PASS: run scripts/fc75_gate.py --pilot"; BAD=1; }
fi
for a in $ARMS; do
  ROOT=$SCRATCH/collapse/$a
  [ -f configs/$a.yaml ] || { echo "FAIL missing configs/$a.yaml"; BAD=1; }
  for s in $SEEDS; do
    [ -L "$ROOT/seed$s/gen0" ] || { echo "FAIL $ROOT/seed$s/gen0 is not a symlink"; BAD=1; }
    [ -s "$ROOT/seed$s/gen0/synthetic_corpus.jsonl" ] || { echo "FAIL $ROOT/seed$s/gen0 has no synthetic_corpus.jsonl"; BAD=1; }
    for g in $GENS; do C=$ROOT/seed$s/gen$g
      [ -e "$C" ] && [ ! -s "$C/metrics.json" ] && { echo "FAIL $C exists without metrics.json: rm -rf it first"; BAD=1; }
    done
  done
done
[ $BAD = 0 ] || { echo "PREFLIGHT FAILED: nothing submitted"; exit 1; }
N=0
for a in $ARMS; do
  CFG=configs/$a.yaml; ROOT=$SCRATCH/collapse/$a
  for s in $SEEDS; do
    PREV_JOB=""
    for g in $GENS; do
      if [ -s "$ROOT/seed$s/gen$g/metrics.json" ]; then echo "SKIP $a s$s g$g (metrics.json present)"; PREV_JOB=""; continue; fi
      DEP=(); [ -n "$PREV_JOB" ] && DEP=(--dependency=afterok:$PREV_JOB)
      CMD=(sbatch --parsable --gres=$GRES --time=$TIME --mem=$MEM --exclude=$EXCLUDE --mail-type=FAIL
           --job-name=${a}-s${s}-g${g} --output=logs/fc75/%x-%j.out
           --export=ALL,CFG=$CFG ${DEP[@]+"${DEP[@]}"} slurm/run_ablation.sbatch $g $s)
      if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; PREV_JOB="DRY$N"
      else
        J=$("${CMD[@]}") || { echo "FAIL sbatch for $a s$s g$g"; exit 1; }
        echo "$J $a s$s g$g gres=$GRES time=$TIME head=$HEAD ${PREV_JOB:+afterok:$PREV_JOB}" | tee -a logs/fc75/submitted.txt
        PREV_JOB=$J
      fi
      N=$((N+1))
    done
  done
done
echo "--- $N cell(s) $([ "$DRY" = 1 ] && echo printed || echo submitted)"
