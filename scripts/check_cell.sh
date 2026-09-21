#!/bin/bash
# check_cell.sh — gate checks for one cell. Every check prints PASS or FAIL;
# a check that finds no evidence is a FAIL, never silence. Exit 1 on any FAIL.
#
# usage: check_cell.sh --cell DIR --log FILE [--corpus FILE --manifest FILE]
#                      [--expect-steps 1752] [--expect-keys 9] [--twin DIR]
#   --corpus/--manifest : md5 of the training corpus must appear in the manifest
#                         (5.3 mixed corpora). Without them, check 1 reads the
#                         driver's own 'slice md5' gate line from --log (5.2).
set -u
CELL= LOG= CORPUS= MANIFEST= TWIN= STEPS=1752 KEYS=9
while [ $# -gt 0 ]; do case $1 in
  --cell) CELL=$2;; --log) LOG=$2;; --corpus) CORPUS=$2;; --manifest) MANIFEST=$2;;
  --twin) TWIN=$2;; --expect-steps) STEPS=$2;; --expect-keys) KEYS=$2;;
  *) echo "unknown arg $1"; exit 2;; esac; shift 2; done
FAILS=0
pass(){ echo "PASS  $*"; }
fail(){ echo "FAIL  $*"; FAILS=$((FAILS+1)); }
[ -n "$CELL" ] && [ -d "$CELL" ] || { echo "FAIL  --cell missing or not a dir: '$CELL'"; exit 1; }
[ -n "$LOG" ] && [ -s "$LOG" ]  || { echo "FAIL  --log missing or empty: '$LOG'"; exit 1; }

echo "=== 1. training corpus identity"
if [ -n "$CORPUS" ]; then
  if [ ! -s "$CORPUS" ]; then fail "corpus missing/empty: $CORPUS"
  elif [ -z "$MANIFEST" ] || [ ! -s "$MANIFEST" ]; then fail "--manifest missing/empty: '$MANIFEST'"
  else
    M=$(md5sum "$CORPUS" | cut -d' ' -f1)
    # output md5 is column 9 of build_mixed_corpus.py's TSV row; matching that
    # column only stops a source-corpus md5 (cols 11/13) from passing by accident
    awk -F'\t' -v m="$M" '$9==m{f=1} END{exit !f}' "$MANIFEST" \
      && pass "corpus md5 $M is an OUTPUT row in $MANIFEST" \
      || fail "corpus md5 $M not in column 9 of $MANIFEST"
  fi
else
  L=$(tr '\r' '\n' < "$LOG" | grep -i 'slice md5' | tail -1)
  [ -n "$L" ] && pass "driver gate line: $L" || fail "no 'slice md5' line in $LOG"
fi

echo "=== 2. step parity (expected $STEPS)"
read RT SPS < <(tr '\r' '\n' < "$LOG" | python3 -c "
import re,sys
t=sys.stdin.read()
Q='[\\x27\\x22]?'
def hms(v):
    p=v.split(':'); return sum(float(x)*60**i for i,x in enumerate(reversed(p)))
rt=[hms(v) for v in re.findall(r'train_runtime'+Q+r'\s*[:=]\s*'+Q+r'([0-9]+(?::[0-9]+){0,2}(?:\.[0-9]+)?)',t)]
sp=re.findall(r'train_steps_per_second'+Q+r'\s*[:=]\s*'+Q+r'([0-9.]+)',t)
print(rt[-1] if rt else 'NA', sp[-1] if sp else 'NA')")
if [ "$RT" = NA ] || [ "$SPS" = NA ]; then fail "train_runtime/steps_per_second not found in $LOG"
else
  N=$(python3 -c "print(round($RT*$SPS))")
  [ "$N" -eq "$STEPS" ] && pass "$RT x $SPS = $N" || fail "$RT x $SPS = $N != $STEPS"
fi

echo "=== 3. metrics.json key count (expected $KEYS)"
MJ="$CELL/metrics.json"
if [ -s "$MJ" ]; then
  K=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))))" "$MJ")
  [ "$K" -eq "$KEYS" ] && pass "$K keys" || fail "$K keys != $KEYS"
else fail "missing $MJ"; fi

echo "=== 4. key-set diff vs twin"
if [ -z "$TWIN" ]; then echo "SKIP  no --twin given (not a pass)"
elif [ ! -s "$TWIN/metrics.json" ] || [ ! -s "$MJ" ]; then fail "metrics.json missing for cell or twin"
else
  python3 - "$MJ" "$TWIN/metrics.json" <<'PY'
import json,sys
a=set(json.load(open(sys.argv[1]))); b=set(json.load(open(sys.argv[2])))
print("INFO  missing vs twin:", sorted(b-a) or "none")
print("INFO  extra   vs twin:", sorted(a-b) or "none")
sys.exit(1 if a-b else 0)
PY
  [ $? -eq 0 ] && pass "no extra keys vs twin (read the missing list)" || fail "cell has keys the twin lacks"
fi

echo "---"; [ $FAILS -eq 0 ] && { echo "ALL CHECKS PASS"; exit 0; } || { echo "$FAILS CHECK(S) FAILED"; exit 1; }
