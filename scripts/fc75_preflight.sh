#!/bin/bash
# fc75_preflight.sh -- read-only checks before Phase 7.5 (decisions.md sec. 73).
# Run from ~/modelcollapse on a Narval login node, after:  source scripts/narval_env.sh
# Every cell runs offline and needs: the gpt2-medium base weights (finetune), gpt2-large (MAUVE), MiniLM
# (embedding variance) and the hellaswag / lambada_openai dataset caches (lm_eval). The Nov 14 purge triage
# (sec. 72.3) rescued gpt2-medium to /project and let the others go, so check before queueing anything.
set -u
BAD=0
ok(){ echo "PASS  $*"; }
bad(){ echo "FAIL  $*"; BAD=$((BAD+1)); }
[ -n "${SCRATCH:-}" ] || { echo "FAIL  \$SCRATCH unset"; exit 1; }
C=$SCRATCH/collapse
HUB=${HF_HOME:-$SCRATCH/hf_cache}/hub
DS=${HF_DATASETS_CACHE:-$SCRATCH/hf_cache/datasets}
RESCUE=$HOME/projects/def-enaskt/cyrh/purge_rescue_2026-11-14

echo "=== 1. repo state"
if git status --short | grep -v '^??' | grep -q .; then git status --short | grep -v '^??'; bad "tracked files are modified (listed above)"
else ok "no modified tracked file"; fi
echo "INFO  HEAD $(git rev-parse --short HEAD); remote branches containing it: $(git branch -r --contains HEAD 2>/dev/null | tr -d ' ' | tr '\n' ' ')"

echo "=== 2. offline assets every cell needs"
snap(){ # $1 hub directory, $2 expected revision or ""
  local d=$HUB/$1 rev s broken
  [ -d "$d/snapshots" ] || { bad "$1: not in $HUB"; return; }
  rev=$(cat "$d/refs/main" 2>/dev/null); s=$d/snapshots/$rev
  [ -n "$rev" ] && [ -d "$s" ] || { bad "$1: refs/main does not resolve to a snapshot"; return; }
  broken=$(find "$s" -xtype l | wc -l)
  [ "$broken" -eq 0 ] || { bad "$1: $broken dangling file(s) in the snapshot (blobs purged)"; return; }
  ls "$s" | grep -qE 'model\.safetensors|pytorch_model\.bin' || { bad "$1: no weights file in the snapshot"; return; }
  [ -z "$2" ] || [ "$rev" = "$2" ] || { bad "$1: revision $rev != recorded $2"; return; }
  ok "$1 @ ${rev:0:12}"
}
snap models--gpt2-medium 6dcaa7a952f72f9298047fd5137cd6e4f05f41da
snap models--gpt2-large ""
snap models--sentence-transformers--all-MiniLM-L6-v2 ""
for d in hellaswag lambada; do
  f=$(find "$DS" -maxdepth 5 -ipath "*$d*" -type f 2>/dev/null | wc -l)
  [ "$f" -gt 0 ] && ok "dataset cache *$d*: $f file(s)" || bad "dataset cache *$d* not found in $DS"
done

echo "=== 3. gen-0 weights and corpora (pilot_gpt2m seeds 43-45)"
for s in 43 44 45; do
  G=$C/pilot_gpt2m/seed$s/gen0
  [ -f "$G/model/config.json" ] && [ -f "$G/metrics.json" ] || { bad "$G: model or metrics.json missing"; continue; }
  n=$(wc -l < "$G/synthetic_corpus.jsonl" 2>/dev/null || echo 0)
  [ "$n" -eq 18686 ] && ok "seed $s gen 0: model, metrics.json, corpus 18686 lines" || bad "$G/synthetic_corpus.jsonl has $n lines"
done
W=$C/pilot_gpt2m/seed43/gen0/model/model.safetensors
if [ -f "$W" ]; then m=$(md5sum "$W" | cut -d' ' -f1)
  [ "$m" = 8c5ce4e0ebc09c55d1680c94ea14e34f ] && ok "seed 43 gen-0 weights md5 = the Sept 12 record" || bad "seed 43 gen-0 weights md5 $m != 8c5ce4e0ebc09c55d1680c94ea14e34f"
else bad "missing $W"; fi

echo "=== 4. fresh real slices"
SB=0
for g in 1 2 3 4 5 6 7; do
  n=$(wc -l < "$C/fresh_slices_gpt2m/gen${g}_real.jsonl" 2>/dev/null || echo 0)
  [ "$n" -eq 18686 ] || { bad "fresh_slices_gpt2m/gen${g}_real.jsonl has $n lines"; SB=1; }
done
[ $SB -eq 0 ] && ok "7 slices, 18686 lines each"

echo "=== 5. greedy-corpus gen-7 checkpoints (the type-C conditions; a condition needs all three seeds)"
for a in mix25_greedy mix50_greedy mix75_greedy gpt2m_greedy; do
  n=0; for s in 43 44 45; do [ -f "$C/$a/seed$s/gen7/model/config.json" ] && n=$((n+1)); done
  [ $n -eq 3 ] && ok "$a: 3/3" || echo "WARN  $a: $n/3 present; prepare_7p5.py will drop this condition"
done
[ -f "$C/gpt2m_greedy/seed43/gen0/model/config.json" ] && ok "canary weights gpt2m_greedy seed 43 gen 0" || echo "WARN  canary weights missing"

echo "---"
if [ $BAD -eq 0 ]; then echo "PREFLIGHT PASS"; exit 0; fi
echo "$BAD CHECK(S) FAILED. Remedies, on a login node:"
echo "  gpt2-medium (must stay at the recorded revision):"
echo "    rsync -a $RESCUE/hf_cache/hub/models--gpt2-medium $HUB/"
echo "  gpt2-large, MiniLM, lm_eval datasets (re-download; offline flags off for this subshell only):"
echo "    ( unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE"
echo "      python -c \"from transformers import AutoTokenizer as T, AutoModelForCausalLM as M; T.from_pretrained('gpt2-large'); M.from_pretrained('gpt2-large')\""
echo "      python -c \"from sentence_transformers import SentenceTransformer as S; S('sentence-transformers/all-MiniLM-L6-v2')\""
echo "      python -c \"from datasets import load_dataset as L; L('hellaswag', trust_remote_code=True); L('EleutherAI/lambada_openai', 'default')\" )"
echo "  Then rerun this script. The pilot cell is the end-to-end check of the offline caches."
exit 1
