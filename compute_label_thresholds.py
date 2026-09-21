#!/usr/bin/env python3
"""Phase 5.4 outcome-label thresholds from the 5.2 NEGATIVE CONTROL ONLY. Read-only.

tau_M = max over control seeds and gens 1-7 of |M(g) - M(0)|, per metric, per decoder.
Must be run, and its numbers recorded in decisions.md, BEFORE any 5.3 metrics.json
is opened. It never reads a mix* directory.

  python compute_label_thresholds.py                       # top-k: control_gpt2m, seeds 43-46
  ARM=<greedy-eval root> SEEDS=43,44,45 CANARY=0 python compute_label_thresholds.py
"""
import json, os, pathlib, sys

ARM = os.environ.get("ARM", "control_gpt2m")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46").split(",")]
CANARY = os.environ.get("CANARY", "1") == "1"
ROOT = pathlib.Path(os.environ["SCRATCH"]) / "collapse" / ARM
KEYS = ["distinct2", "kl_unigram", "mauve"]
# results.md, GPT-2 Medium top-k gen-0 cross-seed means, seeds 43-46 (control gen 0 = pilot gen 0)
GEN0_MEANS = {"distinct2": 0.5816, "kl_unigram": 1.8132, "mauve": 0.9246}

if "mix" in ARM:
    sys.exit("REFUSED: this script must never read a 5.3 grid arm")

m = {}
for s in SEEDS:
    for g in range(8):
        p = ROOT / f"seed{s}" / f"gen{g}" / "metrics.json"
        if not p.is_file():
            sys.exit(f"CANARY FAIL: missing {p}")
        d = json.load(open(p))
        for k in KEYS:
            if k not in d:
                sys.exit(f"CANARY FAIL: key {k} absent in {p}; keys = {sorted(d)}")
        m[s, g] = {k: float(d[k]) for k in KEYS}

if CANARY:
    if SEEDS != [43, 44, 45, 46]:
        sys.exit("CANARY needs SEEDS=43,44,45,46 (the recorded means); set CANARY=0 otherwise")
    for k, ref in GEN0_MEANS.items():
        mean = sum(m[s, 0][k] for s in SEEDS) / len(SEEDS)
        ok = abs(mean - ref) < 5e-5
        print(f"canary gen-0 mean {k:10s} {mean:.6f} vs recorded {ref}  {'OK' if ok else 'FAIL'}")
        if not ok:
            sys.exit("CANARY FAIL — stop; do not record thresholds")

print(f"\narm={ARM} seeds={SEEDS}  shift M(g)-M(0), gens 1-7")
tau = {}
for k in KEYS:
    print(f"\n{k}")
    worst = 0.0
    for s in SEEDS:
        sh = [m[s, g][k] - m[s, 0][k] for g in range(1, 8)]
        worst = max(worst, max(abs(x) for x in sh))
        print(f"  s{s}: " + " ".join(f"{x:+.4f}" for x in sh))
    tau[k] = worst
    print(f"  tau_{k} = {worst:.4f}")
print("\nRECORD IN decisions.md:", json.dumps({k: round(v, 4) for k, v in tau.items()}))
print("(MAUVE is reported, NOT used in the label at 345M — see the pre-registration)")
