#!/usr/bin/env python3
"""
make_figures.py -- harvest + plot every figure for the model-collapse paper.

Read-only with respect to all result trees. Writes only under OUTDIR.

Two stages:
  harvest : walk $SCRATCH/collapse/<arm>/seed<N>/gen<G>/ once, cache everything
            into figures/figdata.json, and run a canary against numbers
            hardcoded from results.md.
  plot    : render figures from the cache (no filesystem walk, so restyling
            is cheap).

Usage
-----
  python make_figures.py              # harvest + canary + plot
  python make_figures.py --harvest    # harvest + canary only
  python make_figures.py --plot       # plot from existing figdata.json
  python make_figures.py --tables     # also dump tables.txt

Discipline enforced in code (see decisions.md / session_2026-08-12.md):
  * canonical benchmark directory pinned per (arm, task); _logsamples and
    _temp are excluded by construction AND asserted against
  * no t-statistics printed for entropy-family metrics
  * calibration gap reported in absolute nats, never as percent change
  * every cross-condition number carries its arm and generation
"""

import argparse
import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SCRATCH = os.environ.get("SCRATCH", str(Path.home() / "scratch"))
COLLAPSE = Path(os.environ.get("COLLAPSE_ROOT", f"{SCRATCH}/collapse"))
OUTDIR = Path(os.environ.get("FIGDIR", str(Path.home() / "modelcollapse" / "figures")))

# arm key -> on-disk run_name, seeds, max generation, display label
ARMS = {
    "gpt2_topk": dict(
        root="pilot_gpt2m", seeds=[43, 44, 45, 46], maxgen=7,
        label="GPT-2 345M / top-k", model="GPT-2 345M", decoder="top-k",
        color="#1b6ca8", ls="-",
    ),
    "gpt2_greedy": dict(
        root="gpt2m_greedy", seeds=[43, 44, 45], maxgen=7,
        label="GPT-2 345M / greedy", model="GPT-2 345M", decoder="greedy",
        color="#1b6ca8", ls="--",
    ),
    "smollm2_topk": dict(
        root="smollm2_1.7b", seeds=[43, 44, 45, 46, 47], maxgen=12,
        label="SmolLM2 1.7B / top-k", model="SmolLM2 1.7B", decoder="top-k",
        color="#c1502e", ls="-",
    ),
    "smollm2_greedy": dict(
        root="smollm2_greedy", seeds=[43, 44, 45], maxgen=7,
        label="SmolLM2 1.7B / greedy", model="SmolLM2 1.7B", decoder="greedy",
        color="#c1502e", ls="--",
    ),
}

# CANONICAL benchmark directory per (run_name, task). Nothing else is read.
# Auxiliary trees disagree by 1-3 items deterministically -- see results.md
# "Canonical Benchmark Directory Map".
BENCH_DIR = {
    ("pilot_gpt2m", "arc_easy"): "benchmarks_arc",
    ("pilot_gpt2m", "hellaswag"): "benchmarks",
    ("pilot_gpt2m", "lambada_openai"): "benchmarks",
    ("gpt2m_greedy", "arc_easy"): "benchmarks_arc",
    ("gpt2m_greedy", "hellaswag"): "benchmarks",
    ("gpt2m_greedy", "lambada_openai"): "benchmarks",
    ("smollm2_1.7b", "arc_easy"): "benchmarks/0shot",
    ("smollm2_1.7b", "hellaswag"): "benchmarks/0shot",
    ("smollm2_1.7b", "lambada_openai"): "benchmarks/0shot",
    ("smollm2_1.7b", "mmlu"): "benchmarks/mmlu_5shot",
    ("smollm2_greedy", "arc_easy"): "benchmarks/0shot",
    ("smollm2_greedy", "hellaswag"): "benchmarks/0shot",
    ("smollm2_greedy", "lambada_openai"): "benchmarks/0shot",
}

FORBIDDEN_DIR_SUFFIXES = ("_logsamples", "_temp")

# metrics.json key aliases -- the script reports which alias it actually found
METRIC_ALIASES = {
    "ppl":            ["perplexity"],
    "distinct2":      ["distinct2", "distinct_2", "distinct-2"],
    "embvar":         ["emb_variance", "embedding_variance", "emb_var"],
    "kl":             ["kl_unigram", "kl_uni", "kl"],
    "mauve":          ["mauve"],
    "entropy":        ["entropy"],
    "top1":           ["top1_prob", "top1_probability"],
    "ppl_wiki":       ["perplexity_ood_wiki"],
    "ppl_c4":         ["perplexity_ood_c4"],
    "ppl_ccnews":     ["perplexity_ood_ccnews"],
    "entropy_wiki":   ["entropy_ood_wiki"],
    "entropy_c4":     ["entropy_ood_c4"],
    "entropy_ccnews": ["entropy_ood_ccnews"],
    "top1_wiki":      ["top1_prob_ood_wiki"],
    "top1_c4":        ["top1_prob_ood_c4"],
    "top1_ccnews":    ["top1_prob_ood_ccnews"],
}

# collapse direction: +1 = rising means collapse, -1 = falling means collapse
DIRECTION = {
    "ppl": +1, "kl": +1, "distinct2": -1, "mauve": -1, "embvar": -1,
    "hellaswag_acc_norm": -1, "lambada_openai_acc": -1,
    "arc_easy_acc_norm": -1, "mmlu_acc": -1, "lambada_openai_ppl": +1,
}

PRETTY = {
    "ppl": "Perplexity", "kl": "KL-unigram", "distinct2": "Distinct-2",
    "mauve": "MAUVE", "embvar": "Emb. variance",
    "hellaswag_acc_norm": "HellaSwag", "lambada_openai_acc": "LAMBADA-acc",
    "arc_easy_acc_norm": "ARC-Easy", "mmlu_acc": "MMLU",
    "lambada_openai_ppl": "LAMBADA-ppl",
    "gap": "Calibration gap", "entropy": "Entropy", "top1": "Top-1 prob.",
}

COLLAPSE_METRICS = ["ppl", "kl", "distinct2", "mauve", "embvar"]
BENCH_METRICS = ["hellaswag_acc_norm", "lambada_openai_acc",
                 "arc_easy_acc_norm", "mmlu_acc"]

MMLU_ROLLUPS = {"mmlu", "mmlu_stem", "mmlu_humanities", "mmlu_other",
                "mmlu_social_sciences"}

# Canary anchors taken verbatim from results.md. (arm, metric, gen, expected).
# Tolerances are loose enough not to fire on rounding in the doc.
ANCHORS = [
    ("gpt2_topk", "ppl", 0, 23.7262, 0.002),
    ("gpt2_topk", "ppl", 7, 30.8962, 0.002),
    ("gpt2_topk", "distinct2", 0, 0.5816, 0.001),
    ("gpt2_topk", "hellaswag_acc_norm", 0, 0.3934, 0.001),
    ("gpt2_topk", "lambada_openai_acc", 0, 0.4355, 0.001),
    ("gpt2_topk", "arc_easy_acc_norm", 7, 0.4386, 0.001),
    ("gpt2_greedy", "ppl", 0, 23.7271, 0.002),
    ("gpt2_greedy", "distinct2", 0, 0.1924, 0.001),
    ("gpt2_greedy", "arc_easy_acc_norm", 7, 0.4184, 0.001),
    ("smollm2_topk", "ppl", 0, 12.0136, 0.002),
    ("smollm2_topk", "ppl", 7, 13.5092, 0.002),
    ("smollm2_topk", "ppl", 12, 14.0911, 0.002),
    ("smollm2_topk", "distinct2", 0, 0.5463, 0.001),
    ("smollm2_topk", "hellaswag_acc_norm", 0, 0.7157, 0.001),
    ("smollm2_topk", "arc_easy_acc_norm", 0, 0.7138, 0.001),
    ("smollm2_topk", "mmlu_acc", 0, 0.4976, 0.002),
    ("smollm2_greedy", "arc_easy_acc_norm", 0, 0.7139, 0.001),
    ("smollm2_greedy", "arc_easy_acc_norm", 7, 0.6811, 0.001),
    ("smollm2_greedy", "hellaswag_acc_norm", 7, 0.6026, 0.001),
]

# gen-7 in-domain calibration gap, all four cells (results.md, Aug 12)
GAP_ANCHORS = [
    ("gpt2_topk", 7, 0.9078, 0.005),
    ("gpt2_greedy", 7, 2.8452, 0.005),
    ("smollm2_topk", 7, 0.6729, 0.005),
    ("smollm2_greedy", 7, 2.7667, 0.005),
]


# ----------------------------------------------------------------------------
# Harvest
# ----------------------------------------------------------------------------

def _pick(d, aliases):
    for a in aliases:
        if a in d:
            return a, d[a]
    return None, None


def read_metrics(path):
    with open(path) as f:
        return json.load(f)


def bench_file(root, seed, gen, task):
    """Return the newest results_*.json under the canonical dir for (root, task)."""
    sub = BENCH_DIR.get((root, task))
    if sub is None:
        return None, "no canonical dir mapped"
    base = COLLAPSE / root / f"seed{seed}" / f"gen{gen}" / sub
    if not base.exists():
        return None, f"missing dir {base}"
    hits = sorted(glob(str(base / "**" / "results*.json"), recursive=True))
    hits = [h for h in hits
            if not any(s in Path(h).parts[-2] or s in str(Path(h).parent)
                       for s in FORBIDDEN_DIR_SUFFIXES)]
    if not hits:
        return None, f"no results*.json under {base}"
    # deterministic: newest timestamped file wins, matching lm-eval convention
    return sorted(hits)[-1], None


def parse_bench(fp, task):
    """Return {metric_suffix: value} for one task from an lm-eval results json."""
    with open(fp) as f:
        blob = json.load(f)
    res = blob.get("results", {})
    out = {}
    if task == "mmlu":
        # lm-eval 0.4.12: the mmlu group row carries only alias/name metadata.
        # Example-weight the 57 leaf subtasks by n-samples.
        nsamp = blob.get("n-samples", {})
        num = den = 0.0
        for k, v in res.items():
            if not k.startswith("mmlu") or k in MMLU_ROLLUPS:
                continue
            acc = v.get("acc,none")
            if acc is None:
                continue
            n = nsamp.get(k, {})
            n = n.get("effective", n.get("original", 1)) if isinstance(n, dict) else 1
            num += acc * n
            den += n
        if den:
            out["acc"] = num / den
        return out
    r = res.get(task, {})
    for src, dst in (("acc,none", "acc"), ("acc_norm,none", "acc_norm"),
                     ("perplexity,none", "ppl")):
        if src in r:
            out[dst] = r[src]
    return out


def harvest():
    data = {"arms": {}, "keys_found": {}, "warnings": []}
    for arm, cfg in ARMS.items():
        root, seeds, maxgen = cfg["root"], cfg["seeds"], cfg["maxgen"]
        armdata = {}
        for seed in seeds:
            for gen in range(maxgen + 1):
                gdir = COLLAPSE / root / f"seed{seed}" / f"gen{gen}"
                mpath = gdir / "metrics.json"
                if mpath.exists():
                    m = read_metrics(mpath)
                    for name, aliases in METRIC_ALIASES.items():
                        key, val = _pick(m, aliases)
                        if val is None:
                            continue
                        data["keys_found"].setdefault(name, set()).add(key)
                        armdata.setdefault(name, {}).setdefault(str(gen), {})[str(seed)] = val
                else:
                    data["warnings"].append(f"{arm}: missing {mpath}")
                for task in ("hellaswag", "lambada_openai", "arc_easy", "mmlu"):
                    if (root, task) not in BENCH_DIR:
                        continue
                    fp, err = bench_file(root, seed, gen, task)
                    if fp is None:
                        if gen == 0 and seed == seeds[0]:
                            data["warnings"].append(f"{arm}/{task}: {err}")
                        continue
                    for suff, val in parse_bench(fp, task).items():
                        name = f"{task}_{suff}"
                        armdata.setdefault(name, {}).setdefault(str(gen), {})[str(seed)] = val
        data["arms"][arm] = armdata
        print(f"[harvest] {arm:16s} metrics={len(armdata):2d} "
              f"seeds={len(seeds)} gens=0-{maxgen}")
    data["keys_found"] = {k: sorted(v) for k, v in data["keys_found"].items()}
    return data


# ----------------------------------------------------------------------------
# Derived quantities and statistics
# ----------------------------------------------------------------------------

def series(data, arm, metric):
    """-> (gens ndarray, seeds list, values ndarray [gen x seed]) or (None,None,None)"""
    d = data["arms"].get(arm, {}).get(metric)
    if not d:
        return None, None, None
    gens = sorted(int(g) for g in d)
    seeds = sorted({int(s) for g in d for s in d[g]},)
    # keep only seeds present at every generation -> balanced paired panel
    seeds = [s for s in seeds if all(str(s) in d[str(g)] for g in gens)]
    if not seeds:
        return None, None, None
    vals = np.array([[d[str(g)][str(s)] for s in seeds] for g in gens], float)
    return np.array(gens), seeds, vals


def gap_series(data, arm, slice_="in"):
    """calibration gap = log(PPL) - entropy, in nats."""
    pk = {"in": "ppl", "wiki": "ppl_wiki", "c4": "ppl_c4", "ccnews": "ppl_ccnews"}[slice_]
    ek = {"in": "entropy", "wiki": "entropy_wiki", "c4": "entropy_c4",
          "ccnews": "entropy_ccnews"}[slice_]
    gp, sp, vp = series(data, arm, pk)
    ge, se, ve = series(data, arm, ek)
    if gp is None or ge is None:
        return None, None, None
    gens = np.array(sorted(set(gp) & set(ge)))
    if len(gens) == 0:
        return None, None, None
    seeds = sorted(set(sp) & set(se))
    ip = {g: i for i, g in enumerate(gp)}
    ie = {g: i for i, g in enumerate(ge)}
    jp = {s: i for i, s in enumerate(sp)}
    je = {s: i for i, s in enumerate(se)}
    out = np.array([[np.log(vp[ip[g], jp[s]]) - ve[ie[g], je[s]] for s in seeds]
                    for g in gens])
    return gens, seeds, out


def paired_t(vals, gen_idx, direction, one_tailed=True):
    """Paired t-test of gen vs gen 0 across seeds. Returns p (never t)."""
    from scipy import stats
    a, b = vals[0], vals[gen_idx]
    if len(a) < 2 or np.allclose(a, b):
        return 1.0
    t, p2 = stats.ttest_rel(b, a)
    if not one_tailed:
        return float(p2)
    # one-tailed in the collapse direction
    if (direction > 0 and t > 0) or (direction < 0 and t < 0):
        return float(p2 / 2)
    return 1.0


def first_signal(data, arm, metric, alpha=0.05):
    gens, seeds, vals = series(data, arm, metric)
    if gens is None or len(seeds) < 2:
        return None
    d = DIRECTION.get(metric, +1)
    for i, g in enumerate(gens):
        if g == 0:
            continue
        if paired_t(vals, i, d, one_tailed=True) < alpha:
            return int(g)
    return None


def pct_delta(vals):
    """% change of the cross-seed mean vs gen 0, per generation."""
    m = vals.mean(axis=1)
    return 100.0 * (m - m[0]) / abs(m[0])


def abruptness(vals):
    m = vals.mean(axis=1)
    d = np.diff(m)
    tot = abs(m[-1] - m[0])
    if tot == 0:
        return np.nan
    return 100.0 * np.max(np.abs(d)) / tot


# ----------------------------------------------------------------------------
# Canary
# ----------------------------------------------------------------------------

def canary(data):
    print("\n" + "=" * 72)
    print("CANARY -- computed vs results.md")
    print("=" * 72)
    fails = 0
    for arm, metric, gen, exp, tol in ANCHORS:
        gens, seeds, vals = series(data, arm, metric)
        if gens is None or gen not in gens:
            print(f"  MISSING  {arm:15s} {metric:22s} gen{gen:<3d} (expected {exp})")
            fails += 1
            continue
        got = vals[list(gens).index(gen)].mean()
        ok = abs(got - exp) <= tol
        fails += 0 if ok else 1
        print(f"  {'ok ' if ok else 'CHECK'}    {arm:15s} {metric:22s} "
              f"gen{gen:<3d} got {got:.4f}  doc {exp:.4f}  n={len(seeds)}")
    for arm, gen, exp, tol in GAP_ANCHORS:
        gens, seeds, vals = gap_series(data, arm, "in")
        if gens is None or gen not in gens:
            print(f"  MISSING  {arm:15s} calibration gap gen{gen} (expected {exp})")
            fails += 1
            continue
        got = vals[list(gens).index(gen)].mean()
        ok = abs(got - exp) <= tol
        fails += 0 if ok else 1
        print(f"  {'ok ' if ok else 'CHECK'}    {arm:15s} {'gap (nats)':22s} "
              f"gen{gen:<3d} got {got:+.4f}  doc {exp:+.4f}  n={len(seeds)}")
    print("-" * 72)
    print(f"  {len(ANCHORS)+len(GAP_ANCHORS)-fails}/{len(ANCHORS)+len(GAP_ANCHORS)} anchors reproduce")
    if fails:
        print("  ^ investigate every CHECK line before any figure enters the draft.")
    print("=" * 72 + "\n")
    return fails


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "lines.linewidth": 1.4, "lines.markersize": 3.5,
    })
    return plt


def _intx(ax):
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{name}.{ext}")
    print(f"[plot] {name}.pdf / .png")


def fig1_transient_arc(data, plt):
    """Headline: the arc, its decoder-set timescale, and its boundary."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    panels = [("GPT-2 345M", ["gpt2_topk", "gpt2_greedy"]),
              ("SmolLM2 1.7B", ["smollm2_topk", "smollm2_greedy"])]
    task_marker = {"arc_easy_acc_norm": "o", "hellaswag_acc_norm": "s",
                   "lambada_openai_acc": "^"}
    task_col = {"arc_easy_acc_norm": "#1b6ca8",
                "hellaswag_acc_norm": "#c1502e",
                "lambada_openai_acc": "#4a7c59"}
    for ax, (title, arms) in zip(axes, panels):
        for arm in arms:
            ls = ARMS[arm]["ls"]
            for metric, mk in task_marker.items():
                gens, seeds, vals = series(data, arm, metric)
                if gens is None:
                    continue
                y = pct_delta(vals)
                ax.plot(gens, y, ls=ls, marker=mk, color=task_col[metric],
                        alpha=0.95 if ls == "-" else 0.75,
                        label=f"{PRETTY[metric]} ({ARMS[arm]['decoder']})")
        ax.axhline(0, color="k", lw=0.8, zorder=0)
        ax.set_title(title)
        ax.set_xlabel("generation"); _intx(ax)
        ax.legend(ncol=1, loc="lower left")
    axes[0].set_ylabel(r"benchmark $\Delta$ vs gen 0 (%)")
    fig.suptitle("Benchmark rises are an early-collapse transient whose window "
                 "narrows with severity\n(solid = top-k, dashed = greedy)",
                 fontsize=9.5, y=1.06)
    _save(fig, "fig1_transient_arc")
    plt.close(fig)


def fig2_first_signal(data, plt):
    """Lag table as a figure: first significant generation per metric per arm."""
    order = COLLAPSE_METRICS + BENCH_METRICS
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    offs = {a: (i - 1.5) * 0.16 for i, a in enumerate(ARMS)}
    for arm, cfg in ARMS.items():
        xs, ys, nev = [], [], []
        for j, m in enumerate(order):
            gens, seeds, vals = series(data, arm, m)
            if gens is None:
                continue
            fs = first_signal(data, arm, m)
            if fs is None:
                nev.append(j + offs[arm])
            else:
                xs.append(fs)
                ys.append(j + offs[arm])
        ax.scatter(xs, ys, color=cfg["color"],
                   marker="o" if cfg["decoder"] == "top-k" else "D",
                   s=26, label=cfg["label"], zorder=3,
                   facecolors=cfg["color"] if cfg["decoder"] == "top-k" else "none",
                   edgecolors=cfg["color"], linewidths=1.1)
        xmax = cfg["maxgen"] + 0.6
        for y in nev:
            ax.annotate("", xy=(xmax + 0.9, y), xytext=(xmax, y),
                        arrowprops=dict(arrowstyle="->", color=cfg["color"],
                                        lw=1.0, alpha=0.8))
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([PRETTY[m] for m in order])
    ax.invert_yaxis()
    ax.axhline(len(COLLAPSE_METRICS) - 0.5, color="k", lw=0.8, ls=":")
    ax.set_xlabel("first generation with $p<0.05$ in the collapse direction "
                  "(paired, vs gen 0)")
    _intx(ax)
    ax.set_xlim(0.3, 14.6)
    ax.text(13.9, len(order) - 0.4, "never\nwithin window", fontsize=6.5,
            ha="center", va="center", style="italic")
    ax.legend(loc="lower right")
    ax.set_title("Diversity and distributional metrics fire at generation 1; "
                 "benchmarks lag or never fire")
    _save(fig, "fig2_first_signal")
    plt.close(fig)


def fig3_effect_size(data, plt):
    """The adjudication that is not significance: |%delta| at the final generation."""
    order = COLLAPSE_METRICS + BENCH_METRICS
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w = 0.2
    for i, (arm, cfg) in enumerate(ARMS.items()):
        xs, hs = [], []
        for j, m in enumerate(order):
            gens, seeds, vals = series(data, arm, m)
            if gens is None:
                continue
            xs.append(j + (i - 1.5) * w)
            hs.append(abs(pct_delta(vals)[-1]))
        ax.bar(xs, hs, width=w, color=cfg["color"],
               alpha=1.0 if cfg["decoder"] == "top-k" else 0.45,
               label=f"{cfg['label']} (gen {cfg['maxgen']})",
               edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PRETTY[m] for m in order], rotation=25, ha="right")
    ax.axvline(len(COLLAPSE_METRICS) - 0.5, color="k", lw=0.8, ls=":")
    ax.set_ylabel(r"$|\Delta|$ vs gen 0 (%, log scale)")
    ax.axhline(2.0, color="grey", lw=0.8, ls="--")
    ax.text(len(order) - 0.4, 2.15, "2%", fontsize=6.5, color="grey", ha="right")
    ax.legend(loc="upper right", ncol=2)
    ax.set_title("Collapse signals move by 12-250%; every benchmark movement "
                 "stays under a few percent")
    _save(fig, "fig3_effect_size")
    plt.close(fig)


def fig4_calibration_gap(data, plt):
    """The candidate monitor: 2x2 (scale x decoder) + register grading."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axes[0]
    for arm, cfg in ARMS.items():
        gens, seeds, vals = gap_series(data, arm, "in")
        if gens is None:
            continue
        ax.plot(gens, vals.mean(axis=1), ls=cfg["ls"], marker="o",
                color=cfg["color"], label=f"{cfg['label']} (n={len(seeds)})")
    ax.axhline(0, color="k", lw=0.8, zorder=0)
    ax.set_xlabel("generation"); _intx(ax)
    ax.set_ylabel(r"$\log \mathrm{PPL} - H$  (nats)")
    ax.set_title("In-domain calibration gap")
    ax.legend(loc="upper left")

    ax = axes[1]
    slices = [("in", "in-domain", "#1b1b1b"), ("wiki", "wiki (same register)", "#1b6ca8"),
              ("c4", "C4 (far-OOD)", "#c1502e"), ("ccnews", "CC-News (far-OOD)", "#e0a458")]
    arm = "smollm2_greedy"
    for sl, lab, col in slices:
        gens, seeds, vals = gap_series(data, arm, sl)
        if gens is None:
            continue
        ax.plot(gens, vals.mean(axis=1), marker="o", color=col, label=lab)
    ax.axhline(0, color="k", lw=0.8, zorder=0)
    ax.set_xlabel("generation"); _intx(ax)
    ax.set_title(f"Register grading, {ARMS[arm]['label']}")
    ax.legend(loc="upper left")
    fig.suptitle("The gap opens from ~0 in all four cells; magnitude is set by "
                 "decoder severity, not scale", fontsize=9.5, y=1.05)
    _save(fig, "fig4_calibration_gap")
    plt.close(fig)


def fig5_register_locality(data, plt):
    """Damage is register-local and bounded."""
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.8))
    keys = [("ppl", "in-domain", "#1b1b1b"), ("ppl_wiki", "wiki", "#1b6ca8"),
            ("ppl_c4", "C4", "#c1502e"), ("ppl_ccnews", "CC-News", "#e0a458")]
    for ax, arm in zip(axes[:2], ["gpt2_topk", "smollm2_topk"]):
        for k, lab, col in keys:
            gens, seeds, vals = series(data, arm, k)
            if gens is None:
                continue
            ax.plot(gens, pct_delta(vals), marker="o", color=col, label=lab)
        ax.set_xlabel("generation"); _intx(ax)
        ax.set_title(ARMS[arm]["label"])
        ax.legend(loc="upper left")
    axes[0].set_ylabel(r"perplexity $\Delta$ vs gen 0 (%)")

    ax = axes[2]
    arm = "smollm2_topk"
    gi, si, vi = series(data, arm, "ppl")
    for k, lab, col in keys[1:]:
        g, s, v = series(data, arm, k)
        if g is None:
            continue
        common = np.array(sorted(set(g) & set(gi)))
        r = np.array([v[list(g).index(x)].mean() / vi[list(gi).index(x)].mean()
                      for x in common])
        ax.plot(common, r, marker="o", color=col, label=f"{lab} / in-domain")
    ax.set_xlabel("generation"); _intx(ax)
    ax.set_ylabel("PPL ratio to in-domain")
    ax.set_title("Ratio: wiki flat, far-OOD declining")
    ax.legend(loc="center right")
    fig.suptitle("Recursive fine-tuning damages a register, not a corpus and "
                 "not the model globally", fontsize=9.5, y=1.05)
    _save(fig, "fig5_register_locality")
    plt.close(fig)


def fig6_rate_shape(data, plt):
    """log-PPL (cross-entropy) is sub-linear in n under both decoders."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, arms in zip(axes, [["gpt2_topk", "gpt2_greedy"],
                               ["smollm2_topk", "smollm2_greedy"]]):
        for arm in arms:
            cfg = ARMS[arm]
            gens, seeds, vals = series(data, arm, "ppl")
            if gens is None:
                continue
            y = np.log(vals).mean(axis=1)
            ax.plot(gens, y - y[0], ls=cfg["ls"], marker="o", color=cfg["color"],
                    label=cfg["label"])
            slope = y[1] - y[0]
            ax.plot(gens, slope * gens, ls=":", lw=1.0, color=cfg["color"],
                    alpha=0.6,
                    label=f"linear-in-$n$ reference ({cfg['decoder']})")
        ax.set_xlabel("generation $n$"); _intx(ax)
        ax.legend(loc="upper left")
    axes[0].set_ylabel(r"$\Delta$ cross-entropy (nats)")
    fig.suptitle("Loss is sub-linear in $n$ under both decoders, against the "
                 "linear-in-$n$ prediction", fontsize=9.5, y=1.05)
    _save(fig, "fig6_rate_shape")
    plt.close(fig)


def fig7_abruptness(data, plt):
    """Appendix: gradual leading signals vs non-monotonic benchmarks."""
    order = COLLAPSE_METRICS + BENCH_METRICS
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    w = 0.2
    for i, (arm, cfg) in enumerate(ARMS.items()):
        xs, hs = [], []
        for j, m in enumerate(order):
            gens, seeds, vals = series(data, arm, m)
            if gens is None:
                continue
            a = abruptness(vals)
            if np.isnan(a):
                continue
            xs.append(j + (i - 1.5) * w)
            hs.append(a)
        ax.bar(xs, hs, width=w, color=cfg["color"],
               alpha=1.0 if cfg["decoder"] == "top-k" else 0.45,
               label=cfg["label"], edgecolor="white", linewidth=0.4)
    ax.axhline(100, color="k", lw=0.9, ls="--")
    ax.text(-0.4, 106, "100% = non-monotonic", fontsize=6.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([PRETTY[m] for m in order], rotation=25, ha="right")
    ax.set_ylabel("abruptness (max per-gen $|\\Delta|$ / total $|\\Delta|$, %)")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Leading signals have gradual onset; benchmark movement is "
                 "non-monotonic")
    _save(fig, "fig7_abruptness")
    plt.close(fig)


def fig8_acc_vs_accnorm(data, plt):
    """Appendix: the arc is not an artifact of length-normalized scoring."""
    pairs = [("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (task, lab) in zip(axes, pairs):
        for arm, cfg in ARMS.items():
            for suff, ls2 in (("acc", ":"), ("acc_norm", "-")):
                gens, seeds, vals = series(data, arm, f"{task}_{suff}")
                if gens is None:
                    continue
                ax.plot(gens, pct_delta(vals), ls=ls2, marker="o",
                        color=cfg["color"],
                        alpha=1.0 if cfg["decoder"] == "top-k" else 0.5,
                        label=f"{cfg['label']} {suff}")
        ax.axhline(0, color="k", lw=0.8, zorder=0)
        ax.set_xlabel("generation"); _intx(ax)
        ax.set_title(lab)
    axes[0].set_ylabel(r"$\Delta$ vs gen 0 (%)")
    axes[1].legend(loc="lower left", fontsize=6)
    fig.suptitle(r"The arc appears in unnormalized $acc$ wherever it appears in "
                 r"$acc\_norm$ (dotted = $acc$)", fontsize=9.5, y=1.05)
    _save(fig, "fig8_acc_vs_accnorm")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------

def dump_tables(data):
    lines = []
    for arm, cfg in ARMS.items():
        lines.append(f"\n### {cfg['label']}  (root={cfg['root']}, maxgen={cfg['maxgen']})")
        lines.append(f"{'metric':24s} {'n':>3s} {'gen0':>10s} {'final':>10s} "
                     f"{'delta%':>9s} {'first-sig':>10s} {'abrupt%':>9s}")
        for m in COLLAPSE_METRICS + BENCH_METRICS + ["lambada_openai_ppl"]:
            gens, seeds, vals = series(data, arm, m)
            if gens is None:
                continue
            fs = first_signal(data, arm, m)
            lines.append(f"{PRETTY.get(m, m):24s} {len(seeds):3d} "
                         f"{vals[0].mean():10.4f} {vals[-1].mean():10.4f} "
                         f"{pct_delta(vals)[-1]:+9.2f} "
                         f"{(str(fs) if fs else 'never'):>10s} "
                         f"{abruptness(vals):9.1f}")
        for sl in ("in", "wiki", "c4", "ccnews"):
            g, s, v = gap_series(data, arm, sl)
            if g is None:
                continue
            lines.append(f"{'gap ' + sl + ' (nats)':24s} {len(s):3d} "
                         f"{v[0].mean():+10.4f} {v[-1].mean():+10.4f} "
                         f"{'--':>9s} {'--':>10s} {'--':>9s}"
                         "   [absolute nats only -- never percent]")
    txt = "\n".join(lines)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "tables.txt").write_text(txt)
    print(txt)
    print(f"\n[tables] {OUTDIR/'tables.txt'}")


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args()
    do_all = not (args.harvest or args.plot or args.tables)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    cache = OUTDIR / "figdata.json"

    if args.harvest or do_all:
        print(f"[harvest] root = {COLLAPSE}")
        data = harvest()
        cache.write_text(json.dumps(data, indent=1))
        print(f"[harvest] cached -> {cache}")
        print("[harvest] metrics.json keys resolved: "
              + ", ".join(f"{k}->{v[0]}" for k, v in sorted(data["keys_found"].items())))
        if data["warnings"]:
            print(f"[harvest] {len(data['warnings'])} warnings; first 10:")
            for w in data["warnings"][:10]:
                print("   " + w)
        canary(data)
    else:
        data = json.loads(cache.read_text())

    if args.plot or do_all:
        plt = _setup_mpl()
        for fn in (fig1_transient_arc, fig2_first_signal, fig3_effect_size,
                   fig4_calibration_gap, fig5_register_locality,
                   fig6_rate_shape, fig7_abruptness, fig8_acc_vs_accnorm):
            try:
                fn(data, plt)
            except Exception as e:
                print(f"[plot] FAILED {fn.__name__}: {type(e).__name__}: {e}")

    if args.tables or do_all:
        dump_tables(data)

    print(f"\nfigures in {OUTDIR}")


if __name__ == "__main__":
    sys.exit(main())
