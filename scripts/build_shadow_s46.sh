#!/bin/bash
set -euo pipefail
SH=$SCRATCH/collapse/control_gpt2m_greedyeval
SRC=$SCRATCH/collapse/control_gpt2m
[ -d "$SH" ] || { echo "REFUSE: shadow root missing"; exit 1; }
[ -e "$SH/seed46" ] && { echo "REFUSE: seed46 already exists"; exit 1; }
for G in 0 1 2 3 4 5 6 7; do
  M="$SRC/seed46/gen$G/model"
  [ -e "$M" ] || { echo "REFUSE: missing $M"; exit 1; }
  echo "gen$G model -> $(readlink -f "$M")"
done
[ "${APPLY:-0}" = "1" ] || { echo "PLAN ONLY. rerun with APPLY=1"; exit 0; }
for G in 0 1 2 3 4 5 6 7; do
  mkdir -p "$SH/seed46/gen$G"
  ln -s "$SRC/seed46/gen$G/model" "$SH/seed46/gen$G/model"
done
find "$SH" -mindepth 2 -maxdepth 2 -type l | wc -l   # must be 0
ls -d "$SH"/seed46/gen*                              # must be 8 real dirs
