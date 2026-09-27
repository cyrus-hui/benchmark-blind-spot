#!/usr/bin/env python3
"""M1 analysis (decisions §61; descriptive only, no test, no p-value, nothing about filtering).

Inputs: m1/<run>/seed<s>/gen<g>.npz (+ .json) from scripts/m1_decompose.py (2ef0f1d7...),
and <run>/seed<s>/gen<g>/metrics.json for the canary.
  Δ_d(g) = (NLL_d - H_d)/n_d ;  δ_d(g) = Δ_d(g) - Δ_d(0) ;  c_d(g) = n_d·δ_d(g)/N
  Σ_d c_d(g) = Δ_cal(g) - Δ_cal(0)  (dcal_series in src/detectors.py)
Canaries, before any statistic (exit 3):
  K0  all 56 cells present; each .json carries scorer md5, real_test md5, B <= 1e-4, C <= 1e-6
  K1  n_d identical across every cell (same docs, same tokenizer), all n_d > 0
  K2  Σ_d c_d(g) reproduces stored dcal_series to <= 1e-6 for every chain and gen
Reported (§61): (1) concentration at gens 1, 7: top-10% and top-1% share of Σ c_d, share carried
by positive c_d, fraction of docs with δ_d > 0, Lorenz deciles (csv);
(2) Spearman of δ_d, adjacent gens 1-2..6-7 and 1-7 within chain; seed pairs within decoder at 1, 7;
(3) Spearman of δ_d(1) with gen-0 mean top-1 probability (P0b link).
--check: K0-K2 on existence, md5s and counts only; prints no statistic of δ.
"""
import argparse, hashlib, itertools, json, math, sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCORER_MD5 = "2ef0f1d7c1bec884b283c45c43797a6d"
REAL_TEST_MD5 = "c5ba60eab60aab3e7d9aa3ed93123a2c"
CHAINS = {"greedy": ("gpt2m_greedy", [43, 44, 45]), "topk": ("pilot_gpt2m", [43, 44, 45, 46])}
GENS = range(8)
TOL_DCAL = 1e-6


def die(msg, code=3):
    print(f"REFUSE: {msg}", flush=True)
    sys.exit(code)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/cyrh/collapse")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    me = hashlib.md5(open(__file__, "rb").read()).hexdigest()
    print(f"m1_analysis md5 {me}  root {root}  mode {'CHECK' if a.check else 'READ'}")

    # ---- K0
    cell = {}
    for dec, (run, seeds) in CHAINS.items():
        for s in seeds:
            for g in GENS:
                p = root / "m1" / run / f"seed{s}" / f"gen{g}.npz"
                if not p.is_file() or not p.with_suffix(".json").is_file():
                    die(f"K0: missing {p} or its .json")
                j = json.load(open(p.with_suffix(".json")))
                if j["script_md5"] != SCORER_MD5 or j["real_test_md5"] != REAL_TEST_MD5:
                    die(f"K0: md5 mismatch in {p.with_suffix('.json')}")
                if not (max(j["canary_B"]) <= 1e-4 and j["canary_C"] <= 1e-6):
                    die(f"K0: canary record out of tolerance in {p.with_suffix('.json')}")
                d = np.load(p)
                cell[(dec, s, g)] = {k: d[k] for k in ("n", "nll", "h", "top1")}
    print(f"K0: {len(cell)} cells present, scorer {SCORER_MD5[:8]}, real_test {REAL_TEST_MD5[:8]}")

    # ---- K1
    n0 = next(iter(cell.values()))["n"]
    if len(n0) == 0 or (n0 <= 0).any():
        die("K1: empty or non-positive n_d")
    for k, v in cell.items():
        if not np.array_equal(v["n"], n0):
            die(f"K1: n_d differs at {k}")
    N = int(n0.sum())
    print(f"K1: n_d identical across all cells; D = {len(n0)} docs, N = {N} tokens, "
          f"n_d range [{n0.min()}, {n0.max()}]")

    # ---- K2
    worst = 0.0
    for dec, (run, seeds) in CHAINS.items():
        for s in seeds:
            st = []
            for g in GENS:
                mp = root / run / f"seed{s}" / f"gen{g}" / "metrics.json"
                if not mp.is_file():
                    die(f"K2: missing {mp}")
                m = json.load(open(mp))
                st.append(math.log(m["perplexity"]) - m["entropy"])
            gap = lambda g: (cell[(dec, s, g)]["nll"] - cell[(dec, s, g)]["h"]).sum() / N
            for g in GENS:
                worst = max(worst, abs((gap(g) - gap(0)) - (st[g] - st[0])))
    print(f"K2: Σ c_d vs stored dcal_series, max |diff| {worst:.2e} (tol {TOL_DCAL:g})")
    if not worst <= TOL_DCAL:
        die("K2 fired — nothing below is reported")
    if a.check:
        print("CHECK PASS — no statistic of δ_d computed or printed")
        return
    print("CANARIES PASS\n" + "=" * 78)

    Dd = lambda dec, s, g: (cell[(dec, s, g)]["nll"] - cell[(dec, s, g)]["h"]) / n0
    dl = lambda dec, s, g: Dd(dec, s, g) - Dd(dec, s, 0)
    D = len(n0)
    k10, k1 = math.ceil(0.10 * D), math.ceil(0.01 * D)
    rows = []

    print(f"\n(1) CONCENTRATION of Δ_cal(g) - Δ_cal(0) over documents "
          f"(top 10% = {k10} docs, top 1% = {k1} docs)")
    for dec, (run, seeds) in CHAINS.items():
        for g in (1, 7):
            for s in seeds:
                c = n0 * dl(dec, s, g) / N
                T = c.sum()
                srt = np.sort(c)[::-1]
                if not T > 0:
                    print(f"  {dec:6s} s{s} g{g}: total {T:+.5f} nats <= 0, shares undefined")
                    continue
                pos = c[c > 0].sum() / T
                print(f"  {dec:6s} s{s} g{g}: total {T:+.5f} nats  top10% {srt[:k10].sum()/T:6.3f}  "
                      f"top1% {srt[:k1].sum()/T:6.3f}  positive-c share {pos:6.3f}  "
                      f"frac δ>0 {(c > 0).mean():.3f}")
                cum = np.cumsum(srt) / T
                rows.append([dec, s, g] + [cum[math.ceil(q * D) - 1] for q in np.arange(0.1, 1.01, 0.1)])
    out = root / "m1" / "lorenz_deciles.csv"
    with open(out, "w") as f:
        f.write("decoder,seed,gen," + ",".join(f"q{q}" for q in range(10, 101, 10)) + "\n")
        for r in rows:
            f.write(",".join(map(str, r[:3])) + "," + ",".join(f"{v:.6f}" for v in r[3:]) + "\n")
    print(f"  Lorenz deciles (docs sorted by c_d, descending) -> {out}")

    print("\n(2) RANK STABILITY — Spearman of δ_d")
    for dec, (run, seeds) in CHAINS.items():
        for s in seeds:
            pr = [(g, g + 1) for g in range(1, 7)] + [(1, 7)]
            print(f"  {dec:6s} s{s}: " + "  ".join(
                f"{x}-{y} {spearmanr(dl(dec, s, x), dl(dec, s, y))[0]:+.3f}" for x, y in pr))
    for dec, (run, seeds) in CHAINS.items():
        for g in (1, 7):
            print(f"  {dec:6s} g{g} seed pairs: " + "  ".join(
                f"{p}v{q} {spearmanr(dl(dec, p, g), dl(dec, q, g))[0]:+.3f}"
                for p, q in itertools.combinations(seeds, 2)))

    print("\n(3) P0b LINK — Spearman(δ_d(1), gen-0 mean top-1 probability)")
    for dec, (run, seeds) in CHAINS.items():
        print(f"  {dec:6s}: " + "  ".join(
            f"s{s} {spearmanr(dl(dec, s, 1), cell[(dec, s, 0)]['top1'] / n0)[0]:+.3f}" for s in seeds))
    print("\nM1 ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
