"""Phase 5.1 — leave-one-scale-out threshold transfer for Δ_cal.
Reads ONLY figures/figdata.json. Canary against the four published gen-7 in-domain cells."""
import json, math, sys
import numpy as np
from scipy import stats

D = json.load(open(sys.argv[1]))["arms"]
ARMS = ["gpt2_topk", "gpt2_greedy", "smollm2_topk", "smollm2_greedy"]

def gap_series(arm):
    a = D[arm]; gens = sorted(int(g) for g in a["ppl"])
    seeds = sorted(a["ppl"]["0"])
    return gens, seeds, {s: [math.log(a["ppl"][str(g)][s]) - a["entropy"][str(g)][s] for g in gens] for s in seeds}

# ---- canary: published gen-7 in-domain cross-seed mean gaps (results.md)
CANARY = {"gpt2_topk": 0.9078, "gpt2_greedy": 2.8452, "smollm2_topk": 0.6729, "smollm2_greedy": 2.7667}
for arm, want in CANARY.items():
    gens, seeds, g = gap_series(arm)
    got = np.mean([g[s][7] for s in seeds])
    status = "OK" if abs(got - want) < 5e-4 else "FAIL"
    print(f"canary {arm:15s} gen7 gap {got:+.4f} vs {want:+.4f} {status}")
    assert status == "OK"
print()

# ---- noise floor: GPT-2 gen-0 retrain (greedy gen0 is an independent training of the same config)
_, sA, gA = gap_series("gpt2_topk"); _, sB, gB = gap_series("gpt2_greedy")
common = sorted(set(sA) & set(sB))
retrain = [gB[s][0] - gA[s][0] for s in common]
print("gen-0 anchors  gpt2_topk:", {s: round(gA[s][0],4) for s in sA})
print("gen-0 anchors  gpt2_greedy:", {s: round(gB[s][0],4) for s in sB})
print(f"retrain nuisance |Δgap| at gen 0, per seed: {[round(abs(x),5) for x in retrain]}  max={max(abs(x) for x in retrain):.5f}")
for arm in ARMS:
    _, seeds, g = gap_series(arm)
    g0 = [g[s][0] for s in seeds]
    print(f"gen-0 cross-seed range {arm:15s}: {max(g0)-min(g0):.5f}")
print()

# ---- per-seed gap increments vs gen 0
print("Δ_cal(g) − Δ_cal(0), cross-seed mean [min,max] per generation")
for arm in ARMS:
    gens, seeds, g = gap_series(arm)
    row = []
    for gi, gen in enumerate(gens):
        d = [g[s][gi] - g[s][0] for s in seeds]
        row.append(f"g{gen}:{np.mean(d):+.3f}[{min(d):+.3f},{max(d):+.3f}]")
    print(f"{arm:15s} " + " ".join(row))
print()

# ---- flag table
KS = [0.05, 0.1, 0.2, 0.3, 0.5]
def first_flag(arm, k):
    gens, seeds, g = gap_series(arm)
    per_seed = []
    for s in seeds:
        f = next((gens[i] for i in range(len(gens)) if g[s][i] - g[s][0] > k), None)
        per_seed.append(f)
    return per_seed, (max(per_seed) if None not in per_seed else None)

print("First-flag generation (all seeds flagged), rule: gap(g)-gap(0) > k")
print(f"{'k':>5} " + " ".join(f"{a:>15s}" for a in ARMS))
for k in KS:
    cells = []
    for arm in ARMS:
        ps, cell = first_flag(arm, k)
        cells.append(f"{str(cell):>5s} {str(ps):>9s}")
    print(f"{k:>5} " + " ".join(f"{c:>15s}" for c in cells))
print()

# ---- benchmark comparator: first significant FALL, one-tailed paired t, p<0.05
BENCH = ["hellaswag_acc_norm", "lambada_openai_acc", "arc_easy_acc_norm", "mmlu_acc"]
print("First generation with a significant FALL (one-tailed paired t, p<0.05); 'never' = none within available gens")
for arm in ARMS:
    a = D[arm]; gens = sorted(int(g) for g in a["ppl"]); seeds = sorted(a["ppl"]["0"])
    out = []
    for b in BENCH:
        if b not in a: continue
        first = None
        for gen in gens[1:]:
            x0 = np.array([a[b]["0"][s] for s in seeds]); xg = np.array([a[b][str(gen)][s] for s in seeds])
            t, p2 = stats.ttest_rel(xg, x0)
            p1 = p2/2 if t < 0 else 1 - p2/2
            if p1 < 0.05: first = gen; break
        out.append(f"{b}:{first if first is not None else 'never(≤'+str(gens[-1])+')'}")
    print(f"{arm:15s} " + "  ".join(out))
