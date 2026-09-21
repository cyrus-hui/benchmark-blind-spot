#!/usr/bin/env python
"""tau_SIGMA from CONTROL chains only (decisions.md Sept 17, pre-registered).

tau = max over control seeds and gens 1-7 of -dU(g), per decoder and per sample setting (sub / full).
Mirrors compute_label_thresholds.py: refuses any non-control arm, so no grid SIGMA value can enter.
  python compute_sigma_thresholds.py                                   # top-k: control_gpt2m 43-46
  ARM=control_gpt2m_greedyeval SEEDS=43,44,45 python compute_sigma_thresholds.py
"""
import json, os, sys
from pathlib import Path

ALLOWED = {"control_gpt2m", "control_gpt2m_greedyeval"}
ARM = os.environ.get("ARM", "control_gpt2m")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46").split(",")]
ROOT = Path(os.environ.get("SIGMA_ROOT", os.path.expandvars("$SCRATCH/collapse/sigma")))

if ARM not in ALLOWED:
    sys.exit(f"REFUSE: {ARM} is not a control arm {sorted(ALLOWED)}")
rows = {}
for s in SEEDS:
    f = ROOT / ARM / f"seed{s}.json"
    if not f.exists():
        sys.exit(f"REFUSE: missing {f}")
    r = json.load(open(f))
    if r["arm"] != ARM or r["seed"] != s:
        sys.exit(f"REFUSE: {f} records arm={r['arm']} seed={r['seed']}")
    rows[s] = r
prompts = {r["prompt_md5"] for r in rows.values()}; idx = {r["index_md5"] for r in rows.values()}
if len(prompts) != 1 or len(idx) != 1:
    sys.exit("REFUSE: control chains disagree on prompt_md5 or index_md5")
print(f"arm={ARM} seeds={SEEDS} n_k={rows[SEEDS[0]]['n_k']} index_md5={idx.pop()}")
for key in ("dU_sub", "dU_full", "dG_KF_sub"):
    print(f"\n-{key}(g), control shifts (collapse direction = positive here)")
    print("seed  " + "  ".join(f"g{g:>8}" for g in range(1, 8)))
    best = (float("-inf"), None)
    for s in SEEDS:
        vals = [-rows[s]["gens"][str(g)][key] for g in range(1, 8)]
        print(f"{s:<5} " + "  ".join(f"{v:+9.3f}" for v in vals))
        for g, v in zip(range(1, 8), vals):
            if v > best[0]:
                best = (v, f"s{s} g{g}")
    tag = "reported only" if key == "dG_KF_sub" else "tau"
    print(f"{tag} {key}: {best[0]:+.3f} (set by {best[1]})")
