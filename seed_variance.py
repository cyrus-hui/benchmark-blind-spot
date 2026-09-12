#!/usr/bin/env python3
"""
seed_variance.py -- per-seed variance reporting for the first-signal claim.

Reads ONLY figures/figdata.json. Walks no result tree: make_figures.harvest()
already pinned the canonical benchmark directory per (arm, task) and resolved
metrics.json aliases, and series() already returns a balanced paired panel.
Re-globbing those trees by hand is the defect class that dropped ARC once and
let benchmarks_logsamples win a last-write-wins merge once.

Two objects, both descriptive, no inference beyond the test already in the paper:

  (1) JACKKNIFE FIRST SIGNAL. Recompute the paper's one-tailed paired t-test
      n times, dropping one seed each time. Reports the range of first-signal
      generations. Asks: does the lag survive losing any one seed?
      Only run for arms with n>=4 -- at n=3 the residual panel is n=2, one
      degree of freedom, and the p-value is not worth printing. Those arms get
      the spread table only, matching the paper's own n=3 rule.

  (2) PER-SEED EFFECT SPREAD. Each seed's total change vs. generation 0 at the
      arm's final generation, as min-max. This is what "underreported variance"
      literally names.

DO NOT run this on `entropy`, `top1` or the calibration gap. Entropy averages
over ~250k frozen tokens, cross-seed variance is near zero, t-statistics come
out in the thousands, and the ~0 gap baseline makes percent change undefined.
DIRECTION does not contain them, and the script refuses them explicitly.

Usage:
    python3 seed_variance.py                 # text tables + canary
    python3 seed_variance.py --latex         # also emit the Appendix A tabular
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import make_figures as mf  # ARMS, DIRECTION, PRETTY, series

FIGDATA = Path(mf.OUTDIR) / "figdata.json"

BANNED = ("entropy", "top1", "gap")

# (arm, metric) -> first-signal generation as printed in Table 2 of the paper.
# Verified before anything is written. Standing rule: any script feeding a
# table carries a canary against numbers already in the docs.
CANARY = {
    ("gpt2_topk", "ppl"): 1,
    ("gpt2_topk", "lambada_openai_acc"): 6,
    ("smollm2_topk", "kl"): 1,
    ("smollm2_greedy", "arc_easy_acc_norm"): 1,
    ("smollm2_greedy", "hellaswag_acc_norm"): 1,
    ("gpt2_greedy", "lambada_openai_acc"): 3,
}

ORDER = ["ppl", "kl", "distinct2", "mauve", "embvar",
         "hellaswag_acc_norm", "lambada_openai_acc", "arc_easy_acc_norm",
         "mmlu_acc", "lambada_openai_ppl"]

PRETTY_EXTRA = {
    "hellaswag_acc_norm": "HellaSwag", "lambada_openai_acc": "LAMBADA-acc",
    "arc_easy_acc_norm": "ARC-Easy", "mmlu_acc": "MMLU (5-shot)",
    "lambada_openai_ppl": "LAMBADA-ppl",
}


def pretty(metric):
    return mf.PRETTY.get(metric, PRETTY_EXTRA.get(metric, metric))


def first_signal(vals, direction, alpha=0.05):
    """vals: [gen x seed]. Returns first gen index with one-tailed p<alpha in
    the collapse direction, or None. Mirrors the paper's test exactly."""
    base = vals[0]
    for g in range(1, vals.shape[0]):
        d = direction * (vals[g] - base)
        if np.allclose(d, d[0]):          # zero variance -> t undefined
            continue
        t, p_two = stats.ttest_1samp(d, 0.0)
        p_one = p_two / 2 if t > 0 else 1.0 - p_two / 2
        if p_one < alpha:
            return g
    return None


def jackknife(vals, seeds, direction):
    """Drop one seed at a time; return {dropped_seed: first_signal_or_None}."""
    out = {}
    for i, s in enumerate(seeds):
        keep = [j for j in range(len(seeds)) if j != i]
        out[s] = first_signal(vals[:, keep], direction)
    return out


def fmt(g):
    return "never" if g is None else f"gen {g}"


def pct(a, b):
    """Total change b vs a, in percent. Guarded: percent is undefined on a
    near-zero baseline (the reason the calibration gap is never percented)."""
    if abs(a) < 1e-9:
        return None
    return 100.0 * (b - a) / a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    if not FIGDATA.exists():
        sys.exit(f"missing {FIGDATA} -- run make_figures.py --harvest first")
    data = json.loads(FIGDATA.read_text())

    # ---- canary -------------------------------------------------------
    failures = []
    for (arm, metric), expect in CANARY.items():
        gens, seeds, vals = mf.series(data, arm, metric)
        if vals is None:
            failures.append(f"{arm}/{metric}: no series")
            continue
        got = first_signal(vals, mf.DIRECTION[metric], args.alpha)
        if got != expect:
            failures.append(f"{arm}/{metric}: reproduced {fmt(got)}, "
                            f"Table 2 says gen {expect}")
    if failures:
        print("CANARY FAILED -- not writing anything:", file=sys.stderr)
        for f in failures:
            print("  " + f, file=sys.stderr)
        sys.exit(1)
    print(f"[canary] {len(CANARY)}/{len(CANARY)} first-signal cells reproduce "
          f"Table 2 from figdata.json\n")

    rows = {}
    for arm, cfg in mf.ARMS.items():
        n_full = len(cfg["seeds"])
        for metric in ORDER:
            if any(b in metric for b in BANNED):
                continue
            gens, seeds, vals = mf.series(data, arm, metric)
            if vals is None:
                continue
            direction = mf.DIRECTION[metric]
            fs = first_signal(vals, direction, args.alpha)
            spread = [pct(vals[0, i], vals[-1, i]) for i in range(len(seeds))]
            spread = [x for x in spread if x is not None]
            jk = jackknife(vals, seeds, direction) if len(seeds) >= 4 else None
            rows[(arm, metric)] = dict(
                n=len(seeds), gens=gens, fs=fs, jk=jk, spread=spread,
                final=int(gens[-1]))

    # ---- text ---------------------------------------------------------
    for arm, cfg in mf.ARMS.items():
        n = len(cfg["seeds"])
        print(f"=== {cfg['label']}  (n={n}, gens 0-{cfg['maxgen']}) ===")
        if n < 4:
            print("    jackknife suppressed: n=3 leaves a 2-seed panel "
                  "(1 d.o.f.); spread only, per the paper's n=3 rule")
        hdr = f"{'metric':16s} {'first sig':>10s} {'jackknife':>16s}  per-seed total %"
        print(hdr)
        for metric in ORDER:
            r = rows.get((arm, metric))
            if r is None:
                continue
            if r["jk"]:
                vs = [r["jk"][s] for s in sorted(r["jk"])]
                lo = [v for v in vs if v is not None]
                jt = ("never (all)" if not lo else
                      f"{fmt(min(lo))}-{fmt(max(lo))}"
                      + ("+never" if any(v is None for v in vs) else ""))
            else:
                jt = "-"
            sp = r["spread"]
            spt = f"{min(sp):+.2f} .. {max(sp):+.2f}" if sp else "n/a"
            print(f"{pretty(metric):16s} {fmt(r['fs']):>10s} {jt:>16s}  {spt}")
        print()

    # ---- latex --------------------------------------------------------
    if args.latex:
        print(r"% --- generated by seed_variance.py; do not hand-edit ---")
        print(r"\begin{table}[H]")
        print(r"  \caption{\textbf{Per-seed variance behind the first-signal "
              r"claim.} \emph{Jackknife} recomputes the one-tailed paired test "
              r"$n$ times, dropping one seed each; the range is over those "
              r"recomputations. Suppressed for the two $n{=}3$ arms, where the "
              r"residual panel is $n{=}2$. \emph{Per-seed} is each seed's total "
              r"change over the arm's own window, min to max.}")
        print(r"  \label{tab:seedvar}")
        print(r"  \centering\small")
        print(r"  \begin{tabular}{llrrr}")
        print(r"    \toprule")
        print(r"    Arm & Metric & First sig & Jackknife & Per-seed total $\Delta\%$ \\")
        print(r"    \midrule")
        for arm, cfg in mf.ARMS.items():
            for metric in ORDER:
                r = rows.get((arm, metric))
                if r is None:
                    continue
                if r["jk"]:
                    vs = [r["jk"][s] for s in sorted(r["jk"])]
                    lo = [v for v in vs if v is not None]
                    jt = ("never" if not lo else
                          (f"{min(lo)}" if min(lo) == max(lo)
                           else f"{min(lo)}--{max(lo)}"))
                    if lo and any(v is None for v in vs):
                        jt += r", never"
                else:
                    jt = "---"
                sp = r["spread"]
                spt = (f"${min(sp):+.1f}$ to ${max(sp):+.1f}$" if sp else "---")
                print(f"    {cfg['label']} & {pretty(metric)} & "
                      f"{fmt(r['fs'])} & {jt} & {spt} \\\\")
            print(r"    \midrule")
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")


if __name__ == "__main__":
    main()
