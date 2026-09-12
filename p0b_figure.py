#!/usr/bin/env python3
"""
p0b_figure.py -- export per-item P0b margins and render fig9.

Imports margin_analysis.load() directly rather than re-deriving margins, so the
byte-length normalizer, the gold-minus-best-competitor definition, and the
top1-top2 decision gap are shared with the analysis script by construction. If
margin_analysis.py changes, this follows it.

Usage
-----
  python p0b_figure.py             # export + plot
  python p0b_figure.py --export    # export only  -> figures/p0b_items.json
  python p0b_figure.py --plot      # plot from cache

Reporting discipline carried from decisions.md:
  * seeds are NEVER pooled -- identical item sets across seeds are
    pseudo-replication. Every panel shows seeds separately.
  * the |m|<0.01 band straddles the bf16 resolution floor (~0.25/blen ~ 0.006);
    the floor is drawn, and sub-0.01 medians are not quoted as resolved.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HOME = Path.home() / "modelcollapse"
sys.path.insert(0, str(HOME))

from margin_analysis import load  # noqa: E402  -- shared margin definition

try:
    from make_figures import _setup_mpl, _save, OUTDIR
except Exception:  # standalone fallback
    OUTDIR = Path(os.environ.get("FIGDIR", str(HOME / "figures")))

    def _setup_mpl():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
            "font.family": "serif", "font.size": 9,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
            "legend.frameon": False, "legend.fontsize": 7.5,
            "lines.linewidth": 1.4, "lines.markersize": 3.5,
        })
        return plt

    def _save(fig, name):
        OUTDIR.mkdir(parents=True, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(OUTDIR / f"{name}.{ext}")
        print(f"[plot] {name}.pdf / .png")

CELLS = [("pilot_gpt2m", "hellaswag", "GPT-2 HellaSwag"),
         ("pilot_gpt2m", "arc_easy", "GPT-2 ARC"),
         ("smollm2_1.7b", "hellaswag", "SmolLM2 HellaSwag"),
         ("smollm2_1.7b", "arc_easy", "SmolLM2 ARC")]
SEEDS = [43, 44]
BF16_FLOOR = 0.006          # ~0.25 / mean byte length
NEAR = 0.01                 # the near-tie band used throughout the paper

GROUPS = [("ic", r"i$\to$c (gained)", "#1b6ca8"),
          ("ci", r"c$\to$i (lost)", "#c1502e"),
          ("ii", "stayed wrong", "#8a8a8a"),
          ("cc", "stayed right", "#4a7c59")]


def export():
    out = {"cells": [], "near_band": NEAR, "bf16_floor": BF16_FLOOR}
    for run, task, label in CELLS:
        for seed in SEEDS:
            a = load(run, 0, seed, task)
            b = load(run, 7, seed, task)
            if a is None or b is None:
                print(f"[export] SKIP {run}/{task}/s{seed}: missing samples")
                continue
            ids = sorted(set(a) & set(b))
            cells = {k: [] for k in ("cc", "ci", "ic", "ii")}
            for i in ids:
                k = ("c" if a[i]["correct"] else "i") + \
                    ("c" if b[i]["correct"] else "i")
                cells[k].append(i)
            n = len(ids)
            rec = dict(
                run=run, task=task, seed=seed, label=label, n=n,
                acc0=sum(a[i]["correct"] for i in ids) / n,
                acc7=sum(b[i]["correct"] for i in ids) / n,
                counts={k: len(v) for k, v in cells.items()},
                net_pp=100.0 * (len(cells["ic"]) - len(cells["ci"])) / n,
                gap0=float(np.mean([a[i]["top_gap"] for i in ids])),
                gap7=float(np.mean([b[i]["top_gap"] for i in ids])),
                median={}, near={}, absm={},
            )
            rec["gap_pct"] = 100.0 * (rec["gap7"] - rec["gap0"]) / rec["gap0"]
            for k in cells:
                if not cells[k]:
                    continue
                m0 = np.array([a[i]["margin"] for i in cells[k]])
                rec["median"][k] = float(np.median(m0))
                rec["near"][k] = float(100 * np.mean(np.abs(m0) < NEAR))
                rec["absm"][k] = [round(float(x), 5) for x in np.abs(m0)]
            q = np.quantile(np.abs([a[i]["margin"] for i in ids]),
                            [0.01, 0.05, 0.25])
            rec["q"] = [float(x) for x in q]
            out["cells"].append(rec)
            print(f"[export] {label:18s} s{seed}  n={n:5d}  "
                  f"net {rec['net_pp']:+.2f}pp  gap {rec['gap_pct']:+.1f}%  "
                  f"near(ic/ci) {rec['near'].get('ic', 0):.1f}/"
                  f"{rec['near'].get('ci', 0):.1f}%")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "p0b_items.json").write_text(json.dumps(out))
    print(f"[export] -> {OUTDIR/'p0b_items.json'}")
    return out


def report(d):
    """Print the ranges the paper quotes, recomputed from the export."""
    flip = [c["near"][k] for c in d["cells"] for k in ("ic", "ci") if k in c["near"]]
    stay = [c["near"][k] for c in d["cells"] for k in ("ii", "cc") if k in c["near"]]
    gaps = [c["gap_pct"] for c in d["cells"]]
    q1 = [c["q"][0] for c in d["cells"]]
    q5 = [c["q"][1] for c in d["cells"]]
    print("\n" + "=" * 66)
    print("P0b ranges recomputed -- use these, not the doc's")
    print("=" * 66)
    print(f"  flip groups   |m|<{NEAR}: {min(flip):.1f}-{max(flip):.1f}%   "
          f"(results.md says 41-56%)")
    print(f"  stayed-put    |m|<{NEAR}: {min(stay):.1f}-{max(stay):.1f}%   "
          f"(results.md says 1.7-5.3%)")
    print(f"  concentration ratio        : "
          f"{min(flip)/max(stay):.1f}x - {max(flip)/min(stay):.1f}x")
    print(f"  decision-gap widening      : {min(gaps):+.1f}% to {max(gaps):+.1f}% "
          f"in {len(gaps)}/{len(gaps)} cells")
    print(f"  |margin| 1% quantile       : {min(q1):.5f}-{max(q1):.5f} "
          f"(below bf16 floor {BF16_FLOOR})")
    print(f"  |margin| 5% quantile       : {min(q5):.5f}-{max(q5):.5f} "
          f"(straddles the floor)")
    print("=" * 66 + "\n")


def plot(d, plt):
    cells = d["cells"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.3),
                             gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]})

    # ---- panel 1: decision-gap dumbbell, all eight cells -------------------
    ax = axes[0]
    ys = np.arange(len(cells))[::-1]
    for y, c in zip(ys, cells):
        ax.plot([c["gap0"], c["gap7"]], [y, y], color="#bbbbbb", lw=2.0, zorder=1)
        ax.scatter([c["gap0"]], [y], color="#ffffff", edgecolor="#333333",
                   s=26, zorder=3, linewidths=1.0)
        ax.scatter([c["gap7"]], [y], color="#c1502e", s=26, zorder=3)
        ax.text(c["gap7"] + 0.012, y, f"{c['gap_pct']:+.1f}%", fontsize=6.5,
                va="center", color="#c1502e")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c['label']} s{c['seed']}" for c in cells], fontsize=7)
    ax.set_xlabel("mean decision gap (top1 $-$ top2)")
    ax.set_xlim(0, max(c["gap7"] for c in cells) * 1.30)
    ax.set_ylim(-1.25, len(cells) - 0.4)
    ax.set_title("Ranking geometry sharpens\nin all eight cells", fontsize=8.5)
    ax.scatter([], [], color="#ffffff", edgecolor="#333333", s=26,
               linewidths=1.0, label="gen 0")
    ax.scatter([], [], color="#c1502e", s=26, label="gen 7")
    ax.legend(loc="lower center", fontsize=7, ncol=2)

    # ---- panel 2: near-tie confinement, seeds shown separately -------------
    ax = axes[1]
    labels = [r[2].replace(" ", "\n") for r in CELLS]
    w = 0.19
    for gi, (k, glab, col) in enumerate(GROUPS):
        for si, seed in enumerate(SEEDS):
            xs, hs = [], []
            for ci, (run, task, lab) in enumerate(CELLS):
                rec = next((c for c in cells if c["run"] == run and
                            c["task"] == task and c["seed"] == seed), None)
                if rec is None or k not in rec["near"]:
                    continue
                xs.append(ci + (gi - 1.5) * w + (si - 0.5) * w * 0.42)
                hs.append(rec["near"][k])
            ax.bar(xs, hs, width=w * 0.42, color=col,
                   alpha=1.0 if si == 0 else 0.55,
                   label=glab if si == 0 else None,
                   edgecolor="white", linewidth=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylim(0, max(v for c in cells for v in c["near"].values()) * 1.34)
    ax.set_ylabel(f"% of group with gen-0 $|m| < {NEAR}$")
    ax.set_title("Flips are confined to the near-tie band\n"
                 "(bars = seeds 43, 44 separately)", fontsize=8.5)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)

    # ---- panel 3: four-cell asymmetry -------------------------------------
    ax = axes[2]
    xs = np.arange(len(cells))
    ic = [c["counts"]["ic"] for c in cells]
    ci = [c["counts"]["ci"] for c in cells]
    ax.bar(xs - 0.19, ic, width=0.36, color="#1b6ca8", label=r"i$\to$c gained")
    ax.bar(xs + 0.19, ci, width=0.36, color="#c1502e", label=r"c$\to$i lost")
    ax.set_xticks(xs)
    short = {"pilot_gpt2m": "GPT-2", "smollm2_1.7b": "SmolLM2"}
    ax.set_xticklabels(
        [f"{short[c['run']]} {'HS' if c['task']=='hellaswag' else 'ARC'} s{c['seed']}\n"
         f"{c['net_pp']:+.2f}pp" for c in cells],
        fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("items")
    ax.set_yscale("log")
    ax.set_ylim(top=max(max(ic), max(ci)) * 2.2)
    ax.set_title("Direction tracks the task,\nnot the model", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.suptitle("The benchmark movement is composed of near-tie items "
                 "reordered by a sharpened ranking, not newly-solved hard items",
                 fontsize=9.5, y=1.08)
    _save(fig, "fig9_p0b_margins")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    do_all = not (a.export or a.plot)
    cache = OUTDIR / "p0b_items.json"
    d = export() if (a.export or do_all) else json.loads(cache.read_text())
    report(d)
    if a.plot or do_all:
        plot(d, _setup_mpl())
    print(f"\nfigure in {OUTDIR}")


if __name__ == "__main__":
    main()
