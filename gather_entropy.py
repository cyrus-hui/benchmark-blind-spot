#!/usr/bin/env python3
"""Sharpening / calibration analysis from the P0 entropy keys.
   gap = log(PPL) - entropy = per-token KL(data || model), in nats.
   Usage: python gather_entropy.py {smollm2|gpt2}"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy import stats

M = {"smollm2": ("smollm2_1.7b", [43, 44, 45, 46, 47]),
     "gpt2":    ("pilot_gpt2m",  [43, 44, 45, 46]),
     "greedy":  ("gpt2m_greedy", [43, 44, 45]),
     "nucleus": ("gpt2m_nucleus",[43, 44, 45])}
SLICES = ["", "_ood_c4", "_ood_ccnews", "_ood_wiki"]
LABEL = {"": "in_domain", "_ood_c4": "c4", "_ood_ccnews": "ccnews", "_ood_wiki": "wiki"}

def main(tag):
    run, seeds = M[tag]
    root = Path(f"{Path.home()}/../../scratch/{Path.home().name}/collapse/{run}").resolve()
    import os
    root = Path(os.environ["SCRATCH"]) / "collapse" / run
    print(f"\n### MODEL: {tag}   ROOT: {root}   seeds: {seeds}")

    D = {}          # D[gen][seed] = metrics dict
    for g in range(0, 20):
        for s in seeds:
            f = root / f"seed{s}" / f"gen{g}" / "metrics.json"
            if f.exists():
                d = json.loads(f.read_text())
                if "entropy" in d:
                    D.setdefault(g, {})[s] = d
    gens = sorted(g for g in D if len(D[g]) == len(seeds))
    if not gens:
        print("no complete generations with entropy keys"); return
    print(f"complete gens with entropy: {gens[0]}-{gens[-1]}")

    def col(g, key):
        return np.array([D[g][s][key] for s in seeds])

    for sl in SLICES:
        pk, ek, tk = f"perplexity{sl}", f"entropy{sl}", f"top1_prob{sl}"
        if ek not in D[gens[0]][seeds[0]]:
            continue
        print(f"\n=== {LABEL[sl].upper()} — entropy / sharpness / calibration (cross-seed mean) ===")
        print("gen |      PPL |  xent |  entropy |  gap(nats) |  top1_prob")
        print("-" * 62)
        g0 = None
        for g in gens:
            ppl, ent, top = col(g, pk), col(g, ek), col(g, tk)
            xent = np.log(ppl)
            gap = xent - ent
            if g0 is None:
                g0 = dict(ent=ent, gap=gap, top=top)
            print(f"{g:3d} | {ppl.mean():8.4f} | {xent.mean():5.3f} | "
                  f"{ent.mean():8.4f} | {gap.mean():10.4f} | {top.mean():10.4f}")
        gl = gens[-1]
        for name, key, direction in (("entropy", ek, "fall"),
                                     ("top1_prob", tk, "rise")):
            a, b = col(gens[0], key), col(gl, key)
            _, p2 = stats.ttest_rel(b, a)
            pct = 100 * (b.mean() - a.mean()) / a.mean()
            print(f"  {name:10s} gen{gens[0]}->{gl}: {a.mean():.4f} -> {b.mean():.4f} "
                  f"({pct:+.2f}%)  p={p2:.2e}   [{direction} = sharpening]")
        a = np.log(col(gens[0], pk)) - col(gens[0], ek)
        b = np.log(col(gl, pk)) - col(gl, ek)
        _, p2 = stats.ttest_rel(b, a)
        print(f"  {'gap':10s} gen{gens[0]}->{gl}: {a.mean():+.4f} -> {b.mean():+.4f} nats "
              f"(delta {b.mean()-a.mean():+.4f})  p={p2:.2e}"
              f"   [widen = confidently wrong; NO pct - gen0 baseline ~= 0]")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smollm2")
