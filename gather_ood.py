#!/usr/bin/env python3
"""OOD-PPL breadth-narrowing analysis. Reads OOD keys merged into metrics.json.
Usage:
  python gather_ood.py smollm2   # seeds 43-47
  python gather_ood.py gpt2      # seeds 43-46
"""
import json, os, sys
import numpy as np
from scipy import stats

MODELS = {
    "smollm2": ("$SCRATCH/collapse/smollm2_1.7b", [43, 44, 45, 46, 47]),
    "gpt2":    ("$SCRATCH/collapse/pilot_gpt2m",  [43, 44, 45, 46]),
}
if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
    print("usage: python gather_ood.py {smollm2|gpt2}"); sys.exit(1)

MODEL = sys.argv[1]
root_tmpl, SEEDS = MODELS[MODEL]
ROOT = os.path.expandvars(root_tmpl)
MAXGEN = int(__import__("os").environ.get("MAXGEN", "7"))
GENS = range(0, MAXGEN + 1)
print(f"[gather_ood] gens {min(GENS)}-{max(GENS)}")
IN_KEY = "perplexity"
OOD = ["perplexity_ood_c4", "perplexity_ood_ccnews", "perplexity_ood_wiki"]
ALL_PPL = [IN_KEY] + OOD

def load_metrics(seed, gen):
    p = os.path.join(ROOT, f"seed{seed}", f"gen{gen}", "metrics.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

# data[key][gen][seed] = value
data = {k: {g: {} for g in GENS} for k in ALL_PPL}
missing, missing_ood = [], []
for seed in SEEDS:
    for gen in GENS:
        m = load_metrics(seed, gen)
        if m is None:
            missing.append(f"seed{seed}/gen{gen}/metrics.json"); continue
        for k in ALL_PPL:
            if k in m: data[k][gen][seed] = m[k]
            elif k in OOD: missing_ood.append(f"seed{seed}/gen{gen}:{k}")

print(f"\n### MODEL: {MODEL}   ROOT: {ROOT}   seeds: {SEEDS}\n")

# ---- cross-seed mean PPL by gen ----
print("=== PPL BY GENERATION (cross-seed mean [n]) ===\n")
hdr = "gen | " + " | ".join(f"{k.replace('perplexity_ood_','ood_').replace('perplexity','in_domain'):>14}" for k in ALL_PPL)
print(hdr); print("-"*len(hdr))
for gen in GENS:
    row = [f"{gen:>3}"]
    for k in ALL_PPL:
        vals = list(data[k][gen].values())
        row.append(f"{np.mean(vals):>10.4f}[{len(vals)}]" if vals else f"{'--':>14}")
    print(" | ".join(row))

# ---- three-way divergence: gen0->end %rise, OOD vs in-domain, seed-paired ----
print("\n=== OOD-PPL THREE-WAY DIVERGENCE (gen0->end %rise; OOD vs in-domain) ===\n")
seeds0 = sorted(data[IN_KEY][0].keys())

def pct_rise_per_seed(metric):
    out = {}
    for s in seeds0:
        if s in data[metric][0] and s in data[metric][MAXGEN]:
            v0, v7 = data[metric][0][s], data[metric][MAXGEN][s]
            out[s] = (v7 - v0) / v0 * 100
    return out

ind = pct_rise_per_seed(IN_KEY)
print(f"{'corpus':>22} | {'gen0->end %rise':>13} | {'vs in-dom (pp gap)':>18} | {'paired p (faster)':>17}")
print("-"*80)
print(f"{'in-domain (WikiText2)':>22} | {np.mean(list(ind.values())):>+12.2f}% | {'(reference)':>18} | {'--':>17}")
for m in OOD:
    rise = pct_rise_per_seed(m)
    common = [s for s in seeds0 if s in rise and s in ind]
    if not common:
        print(f"{m.replace('perplexity_ood_',''):>22} | {'-- no data --':>13} | {'--':>18} | {'--':>17}"); continue
    ood_arr = np.array([rise[s] for s in common])
    ind_arr = np.array([ind[s] for s in common])
    gap = np.mean(ood_arr) - np.mean(ind_arr)
    t, p_two = stats.ttest_rel(ood_arr, ind_arr)
    p_one = p_two/2 if (np.mean(ood_arr - ind_arr) > 0) else 1 - p_two/2
    print(f"{m.replace('perplexity_ood_',''):>22} | {np.mean(ood_arr):>+12.2f}% | {gap:>+17.2f} | {p_one:>17.4f}")


# ---- register-matched decision test: wiki vs far-OOD (pre-registered) ----
wiki = pct_rise_per_seed("perplexity_ood_wiki")
far  = {}
for s in seeds0:
    c4  = pct_rise_per_seed("perplexity_ood_c4").get(s)
    ccn = pct_rise_per_seed("perplexity_ood_ccnews").get(s)
    if c4 is not None and ccn is not None: far[s] = (c4 + ccn) / 2
common = [s for s in seeds0 if s in wiki and s in far]
if common:
    w = np.array([wiki[s] for s in common]); f = np.array([far[s] for s in common])
    t, p_two = stats.ttest_rel(w, f)
    p_one = p_two/2 if np.mean(w - f) > 0 else 1 - p_two/2
    print(f"\n=== REGISTER-MATCHED DECISION: wiki vs far-OOD mean (paired, n={len(common)}) ===")
    print(f"  wiki %rise {np.mean(w):+.2f}  vs  far-OOD %rise {np.mean(f):+.2f}  "
          f"(gap {np.mean(w-f):+.2f} pp; one-tailed p[wiki faster] = {p_one:.4f})")
    print("  wiki ~ far-OOD (or slower) => register alignment CONFIRMED")
    print("  wiki >> far-OOD, toward in-domain => register alignment FAILS")

# ---- OOD/in-domain ratio trajectory (widening => breadth narrowing) ----
print("\n=== OOD / in-domain PPL RATIO by generation (cross-seed mean) ===\n")
print("gen | " + " | ".join(f"{c.replace('perplexity_ood_',''):>10}" for c in OOD))
print("-"*40)
for gen in GENS:
    iv = list(data[IN_KEY][gen].values())
    if not iv: print(f"{gen:>3} |   -- no in-domain --"); continue
    ind_g = np.mean(iv)
    cells = []
    for m in OOD:
        vals = list(data[m][gen].values())
        cells.append(f"{np.mean(vals)/ind_g:>10.4f}" if vals else f"{'--':>10}")
    print(f"{gen:>3} | " + " | ".join(cells))

if missing: print("\n=== MISSING metrics.json ==="); [print(" ",x) for x in missing]
if missing_ood: print("\n=== MISSING OOD KEYS ==="); [print(" ",x) for x in missing_ood]
if not missing and not missing_ood: print("\n=== All files + OOD keys present ===")
