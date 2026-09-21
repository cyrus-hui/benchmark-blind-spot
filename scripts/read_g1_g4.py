#!/usr/bin/env python
"""G1-G4 read. Refuses unless the registry is complete and its md5 is quoted.

  python scripts/read_g1_g4.py --plan
  python scripts/read_g1_g4.py --authorise-grid-read <md5>
"""
import argparse, hashlib, json, math, os, sys
from pathlib import Path

ROOT = Path(os.path.expandvars("$SCRATCH/collapse"))
REG  = Path("registered_thresholds.json")
TAU_DCAL_PATH = "dcal.tau"   # verified against control_gpt2m by canary()
SEEDS = (43, 44, 45)
GENS  = range(1, 8)
TOL, SAT, G2_MIN = 0.002, 0.8, 0.005

ARMS = {("topk",0):"control_gpt2m", ("greedy",0):"control_gpt2m",
        ("topk",100):"pilot_gpt2m",  ("greedy",100):"gpt2m_greedy"}
for r in (25, 50, 75):
    ARMS[("topk", r)]   = f"mix{r}_topk"
    ARMS[("greedy", r)] = f"mix{r}_greedy"

def pending(node, prefix=""):
    """Walk the NESTED registry. A top-level scan is the Sept 20 (2) defect."""
    out = []
    for k, v in node.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict): out += pending(v, p + ".")
        elif v is None:         out.append(p)
    return out

def dig(node, path):
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            sys.exit(f"REFUSE: no registry path {path}")
        node = node[part]
    return node

def authorise(quoted):
    raw = REG.read_bytes()
    got = hashlib.md5(raw).hexdigest()
    reg = json.loads(raw)
    p = pending(reg)
    if p:            sys.exit(f"REFUSE: registry pending {sorted(p)}")
    if quoted != got: sys.exit(f"REFUSE: quoted {quoted}, registry is {got}")
    print(f"authorised  registry md5 {got}  pending []")
    return reg

def cells(r, dec, seed):
    return [ROOT / ARMS[(dec, r)] / f"seed{seed}" / f"gen{g}" / "metrics.json"
            for g in range(0, 8)]

def delta_series(r, dec, seed):
    vals = []
    for p in cells(r, dec, seed):
        m = json.load(open(p))
        vals.append(math.log(m["perplexity"]) - m["entropy"])
    return [v - vals[0] for v in vals]           # Delta(g), index 0..7

def classify(d7, tau):
    rs = sorted(d7)
    for i in range(len(rs)):
        for j in range(i + 1, len(rs)):
            if d7[rs[j]] < d7[rs[i]] - TOL: return "non-monotone"
    if d7[100] <= 0:            return "undefined (r100 <= 0)"
    if d7[25] >= SAT * d7[100]: return "saturating"
    if d7[25] > tau:            return "graded, detectable"
    return "graded, below floor"

CONTROL_SEEDS = (43, 44, 45, 46)

def canary(registered):
    """tau_Delta from control_gpt2m only, before any grid arm is opened."""
    own = max(delta_series(0, "topk", s)[g] for s in CONTROL_SEEDS for g in GENS)
    ok = abs(own - registered) <= 5e-5
    print(("PASS  " if ok else "REFUSE  ") +
          f"tau_Delta recomputed {own:+.6f} vs registered {registered}")
    if not ok: sys.exit(1)
    return own

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorise-grid-read"); ap.add_argument("--plan", action="store_true")
    ap.add_argument("--canary", action="store_true")
    a = ap.parse_args()

    if a.plan:
        miss = 0
        for dec in ("topk", "greedy"):
            for r in (0, 25, 50, 75, 100):
                for s in SEEDS:
                    for p in cells(r, dec, s):
                        if not p.exists(): print("MISSING", p); miss += 1
        print(f"plan: {2*5*3*8} cells, {miss} missing. NO VALUE READ.")
        return
    if a.canary:
        canary(dig(json.loads(REG.read_bytes()), TAU_DCAL_PATH))
        print("canary: control_gpt2m only. NO GRID VALUE READ.")
        return
    if not a.authorise_grid_read: sys.exit("REFUSE: --authorise-grid-read required")
    tau = canary(dig(authorise(a.authorise_grid_read), TAU_DCAL_PATH))
    print(f"tau_Delta = {tau:+.6f}  (full precision)")

    D = {(dec, s): {r: delta_series(r, dec, s) for r in (0, 25, 50, 75, 100)}
         for dec in ("topk", "greedy") for s in SEEDS}

    print("\n(G1) class per (decoder, seed), on Delta(7)")
    classes = {}
    for k, per_r in D.items():
        d7 = {r: v[7] for r, v in per_r.items()}
        classes[k] = classify(d7, tau)
        print("  %-6s s%d  " % (k[0], k[1]) +
              "  ".join(f"r{r}={d7[r]:+.4f}" for r in (0,25,50,75,100)) +
              f"   -> {classes[k]}")
    top = max(set(classes.values()), key=list(classes.values()).count)
    n = list(classes.values()).count(top)
    print(f"  VERDICT: {top} ({n}/6)" if n >= 5 else f"  VERDICT: no majority (best {top} {n}/6)")

    print("\n(G2) D_s(r) = Delta(7)_greedy - Delta(7)_topk")
    for r in (25, 50, 75):
        ds = [D[("greedy", s)][r][7] - D[("topk", s)][r][7] for s in SEEDS]
        sign = all(x > 0 for x in ds) or all(x < 0 for x in ds)
        mag  = all(abs(x) > G2_MIN for x in ds)
        print("  r=%-3d " % r + "  ".join(f"{x:+.4f}" for x in ds) +
              f"   effect={'YES' if (sign and mag) else 'no'}")
    for dec in ("topk", "greedy"):
        moved = [D[(dec, s)][75][7] - D[(dec, s)][25][7] > G2_MIN for s in SEEDS]
        print(f"  r moves Delta(7) at {dec}: {sum(moved)}/3")

    print("\n(G3) latency: first g with Delta(g) > tau")
    for dec in ("topk", "greedy"):
        for r in (0, 25, 50, 75, 100):
            lat = [next((g for g in GENS if D[(dec, s)][r][g] > tau), None) for s in SEEDS]
            print("  %-6s r=%-3d " % (dec, r) + "  ".join("--" if x is None else str(x) for x in lat))

    print("\n(G4) Fixed-budget mixing; GPT-2 Medium 345M; gens 1-7; n=3 chains per "
          "(r, decoder); corpus-mode prompts from the generation's own training "
          "mixture; WikiText-103 real component scored on WikiText-2 test. NOT "
          "accumulation. The 6 G1 cells share slice 7 and, per decoder, a gen-0 model.")

if __name__ == "__main__":
    main()
