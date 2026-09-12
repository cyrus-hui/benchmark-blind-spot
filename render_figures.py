#!/usr/bin/env python3
"""
render_figures.py -- standalone re-render of the paper figures.

Reads ONLY the cached harvest (figures/figdata.json, figures/p0b_items.json).
No cluster access, no re-harvest, no lm-eval. Safe to run locally.

    python3 render_figures.py --datadir figures --out figures

Fixes applied vs. the Aug-12 renders:
  fig2_first_signal   legend moved outside the axes; MMLU arrow restored;
                      third marker state for non-diagnostic fires; wrong-dir
                      arrows hatched; "never" annotation no longer collides
                      with the tick labels.
  fig1_transient_arc  legends moved outside; inset on the +/-2.5% band where
                      the arc actually lives.
  fig4_calibration_gap  legends moved out of the data.
  fig3_effect_size    2% reference label brought inside the axes.
  fig7_abruptness     symlog y-axis so the 27-45% leading signals are legible
                      next to the 640% blip.
"""
import argparse, json, math, os
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------- config ----
ARMS = ["gpt2_topk", "gpt2_greedy", "smollm2_topk", "smollm2_greedy"]
ARM_LABEL = {
    "gpt2_topk": "GPT-2 345M / top-$k$",
    "gpt2_greedy": "GPT-2 345M / greedy",
    "smollm2_topk": "SmolLM2 1.7B / top-$k$",
    "smollm2_greedy": "SmolLM2 1.7B / greedy",
}
BLUE, RED = "#1f77b4", "#d62728"
ARM_STYLE = {  # colour, filled?
    "gpt2_topk": (BLUE, True),
    "gpt2_greedy": (BLUE, False),
    "smollm2_topk": (RED, True),
    "smollm2_greedy": (RED, False),
}
ARM_SHORT = {"gpt2_topk": "345M top-$k$", "gpt2_greedy": "345M greedy",
             "smollm2_topk": "1.7B top-$k$", "smollm2_greedy": "1.7B greedy"}
MAXGEN = {"gpt2_topk": 7, "gpt2_greedy": 7, "smollm2_topk": 12, "smollm2_greedy": 7}

# +1 = rises under collapse, -1 = falls under collapse
DIR = dict(ppl=+1, kl=+1, distinct2=-1, mauve=-1, embvar=-1,
           hellaswag_acc_norm=-1, lambada_openai_acc=-1,
           arc_easy_acc_norm=-1, mmlu_acc=-1, lambada_openai_ppl=+1)
LABEL = dict(ppl="Perplexity", kl="KL-unigram", distinct2="Distinct-2",
             mauve="MAUVE", embvar="Emb. variance",
             hellaswag_acc_norm="HellaSwag", lambada_openai_acc="LAMBADA-acc",
             arc_easy_acc_norm="ARC-Easy", mmlu_acc="MMLU",
             lambada_openai_ppl="LAMBADA-ppl")
# order top-to-bottom in fig 2; collapse metrics above the divider
ROWS = ["ppl", "kl", "distinct2", "mauve", "embvar",
        "hellaswag_acc_norm", "lambada_openai_acc", "arc_easy_acc_norm", "mmlu_acc"]
DIVIDER_AFTER = "embvar"

# fires that reach p<0.05 but carry no diagnostic information (see paper 4.1)
NON_DIAGNOSTIC = {("gpt2_topk", "hellaswag_acc_norm"),
                  ("gpt2_greedy", "hellaswag_acc_norm")}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ------------------------------------------------------------- utilities ----
def load(datadir):
    with open(os.path.join(datadir, "figdata.json")) as f:
        fig = json.load(f)
    p0b = None
    p = os.path.join(datadir, "p0b_items.json")
    if os.path.exists(p):
        with open(p) as f:
            p0b = json.load(f)
    return fig["arms"], p0b


def seeds(cell):
    return sorted(cell["0"])


def mean_at(cell, g):
    return st.mean([cell[str(g)][s] for s in seeds(cell)])


def traj(arms, arm, metric):
    """(gens, mean values) or None if the metric is absent for this arm."""
    cell = arms[arm].get(metric)
    if not cell:
        return None
    gens = sorted((int(k) for k in cell), key=int)
    return gens, [mean_at(cell, g) for g in gens]


def pct_traj(arms, arm, metric):
    t = traj(arms, arm, metric)
    if not t:
        return None
    gens, v = t
    b = v[0]
    return gens, [100 * (x - b) / b for x in v]


def ttest_rel_p(cur, base):
    """One-sided-ready paired t: returns (t, two-sided p). scipy-free."""
    d = [a - b for a, b in zip(cur, base)]
    n = len(d)
    m = st.mean(d)
    if n < 2:
        return 0.0, 1.0
    sd = st.stdev(d)
    if sd == 0:
        return (math.inf if m > 0 else -math.inf if m < 0 else 0.0), (0.0 if m else 1.0)
    t = m / (sd / math.sqrt(n))
    # two-sided p from Student's t via incomplete beta
    df = n - 1
    x = df / (df + t * t)
    p = _betainc(df / 2.0, 0.5, x)
    return t, p


def _betainc(a, b, x):
    """Regularised incomplete beta, continued fraction (Numerical Recipes)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _cf(a, b, x) / a
    return 1 - front * _cf(b, a, 1 - x) / b


def _cf(a, b, x, itmax=300, eps=1e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def first_signal(arms, arm, metric):
    """(first_gen | None, wrong_direction: bool)."""
    cell = arms[arm].get(metric)
    if not cell:
        return None, None
    ss = seeds(cell)
    base = [cell["0"][s] for s in ss]
    sign = DIR[metric]
    gens = sorted((int(k) for k in cell), key=int)
    fin, b0 = mean_at(cell, gens[-1]), mean_at(cell, 0)
    wrong = (fin - b0) * sign < 0
    for g in gens[1:]:
        cur = [cell[str(g)][s] for s in ss]
        t, p2 = ttest_rel_p(cur, base)
        tt = t if sign > 0 else -t
        p1 = p2 / 2 if tt > 0 else 1 - p2 / 2
        if p1 < 0.05:
            return g, wrong
    return None, wrong


def gap_traj(arms, arm, ppl_key="ppl", ent_key="entropy"):
    cp, ce = arms[arm].get(ppl_key), arms[arm].get(ent_key)
    if not cp or not ce:
        return None
    gens = sorted((int(k) for k in cp), key=int)
    return gens, [math.log(mean_at(cp, g)) - mean_at(ce, g) for g in gens]


# --------------------------------------------------------------- fig 2 ------
def fig_first_signal(arms, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    ylab, ypos = [], []
    for i, m in enumerate(ROWS):
        y = len(ROWS) - i
        ylab.append(LABEL[m]); ypos.append(y)
        for j, arm in enumerate(ARMS):
            if m not in arms[arm]:
                continue
            colour, filled = ARM_STYLE[arm]
            off = (j - 1.5) * 0.16
            g, wrong = first_signal(arms, arm, m)
            if g is not None:
                nd = (arm, m) in NON_DIAGNOSTIC
                ax.plot(g, y + off, marker="o" if filled else "D",
                        ms=4.6 if filled else 4.2,
                        mfc=("white" if (nd or not filled) else colour),
                        mec=("0.45" if nd else colour),
                        mew=1.5 if nd else 1.1, ls="none", zorder=3)
                if nd:  # slash through = significant but non-diagnostic
                    ax.plot(g, y + off, marker="x", ms=3.4, color="0.35",
                            mew=1.0, ls="none", zorder=4)
            else:
                x0 = MAXGEN[arm] + 0.15
                ax.annotate("", xy=(x0 + 1.15, y + off), xytext=(x0, y + off),
                            arrowprops=dict(arrowstyle="-|>", color=colour,
                                            lw=1.1,
                                            ls=(":" if wrong else "-"),
                                            shrinkA=0, shrinkB=0), zorder=2)
    div = len(ROWS) - ROWS.index(DIVIDER_AFTER) - 0.5
    ax.axhline(div, color="0.3", ls=":", lw=0.9)
    ax.text(9.9, div + 0.22, "collapse metrics", fontsize=6.3, color="0.4",
            ha="center", va="bottom")
    ax.text(9.9, div - 0.22, "benchmarks (detectors under test)", fontsize=6.3,
            color="0.4", ha="center", va="top")

    ax.set_yticks(ypos); ax.set_yticklabels(ylab)
    ax.set_ylim(0.3, len(ROWS) + 0.7)
    ax.set_xlim(0.2, 14.2)
    ax.set_xticks([1, 2, 4, 6, 8, 10, 12])
    ax.set_xlabel("first generation with $p<0.05$ in the collapse direction "
                  "(paired, vs gen 0)")
    ax.set_title("Diversity and distributional metrics fire at generation 1; "
                 "benchmarks lag or never fire", pad=8)

    handles = [Line2D([], [], ls="none",
                      marker="o" if ARM_STYLE[a][1] else "D",
                      mfc=ARM_STYLE[a][0] if ARM_STYLE[a][1] else "white",
                      mec=ARM_STYLE[a][0], ms=5, label=ARM_LABEL[a])
               for a in ARMS]
    handles += [
        Line2D([], [], ls="none", marker="o", mfc="white", mec="0.45", mew=1.5,
               ms=5, label="significant, not diagnostic"),
        Line2D([], [], color="0.35", lw=1.2,
               marker=">", markevery=[-1], ms=4,
               label="never fires (end of window)"),
        Line2D([], [], color="0.35", lw=1.2, ls=":",
               marker=">", markevery=[-1], ms=4,
               label="never fires: moved wrong way"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, handletextpad=0.5, borderaxespad=0)
    fig.savefig(os.path.join(out, "fig2_first_signal.pdf"))
    plt.close(fig)


# --------------------------------------------------------------- fig 1 ------
def fig_transient_arc(arms, out):
    """2x2: full range on top, the +/-2.5% band where the arc lives beneath."""
    tasks = [("arc_easy_acc_norm", "ARC-Easy", "#1f77b4", "o"),
             ("hellaswag_acc_norm", "HellaSwag", "#d62728", "s"),
             ("lambada_openai_acc", "LAMBADA-acc", "#2ca02c", "^")]
    panels = [("GPT-2 345M", "gpt2_topk", "gpt2_greedy"),
              ("SmolLM2 1.7B", "smollm2_topk", "smollm2_greedy")]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.3),
                             gridspec_kw=dict(height_ratios=[1.25, 1.0],
                                              hspace=0.32))
    for col, (title, tk, gr) in enumerate(panels):
        for row in (0, 1):
            ax = axes[row][col]
            for arm, ls, alpha in ((tk, "-", 1.0), (gr, "--", 0.8)):
                for key, lab, colr, mk in tasks:
                    p = pct_traj(arms, arm, key)
                    if not p:
                        continue
                    g, v = p
                    ax.plot(g, v, ls=ls, color=colr, alpha=alpha,
                            marker=mk, ms=2.6, lw=1.2)
            ax.axhline(0, color="0.2", lw=0.8)
            if row == 0:
                ax.set_title(title, pad=4)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_ylim(-2.5, 2.5)
                ax.set_xlabel("generation")
                ax.set_title("same data, $\\pm2.5\\%$ band",
                             fontsize=7.5, color="0.3", pad=3)
    axes[0][0].set_ylabel("$\\Delta$ vs gen 0 (%)")
    axes[1][0].set_ylabel("$\\Delta$ vs gen 0 (%)")
    handles = [Line2D([], [], color=c, marker=m, ms=3, lw=1.2, label=l)
               for _, l, c, m in tasks]
    handles += [Line2D([], [], color="0.3", ls="-", lw=1.2, label="top-$k$"),
                Line2D([], [], color="0.3", ls="--", lw=1.2, label="greedy")]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.045))
    fig.suptitle("Benchmark rises are an early-collapse transient whose window "
                 "narrows with severity", y=0.98)
    fig.savefig(os.path.join(out, "fig1_transient_arc.pdf"))
    plt.close(fig)


# --------------------------------------------------------------- fig 4 ------
def fig_calibration_gap(arms, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax = axes[0]
    for arm in ARMS:
        t = gap_traj(arms, arm)
        if not t:
            continue
        g, v = t
        colour, filled = ARM_STYLE[arm]
        ax.plot(g, v, color=colour, ls="-" if filled else "--",
                marker="o", ms=2.8, lw=1.2,
                label=f"{ARM_SHORT[arm]} ($n$={len(seeds(arms[arm]['ppl']))})")
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("generation")
    ax.set_ylabel("$\\log$ PPL $-\\ H$  (nats)")
    ax.set_title("In-domain calibration gap")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.22), frameon=False,
              ncol=2, fontsize=6.0, columnspacing=0.9, handletextpad=0.4,
              handlelength=1.4)

    ax = axes[1]
    arm = "smollm2_greedy"
    for key_p, key_e, lab, col in (
            ("ppl", "entropy", "in-domain", "0.15"),
            ("ppl_wiki", "entropy_wiki", "wiki (same reg.)", BLUE),
            ("ppl_c4", "entropy_c4", "C4 (far-OOD)", RED),
            ("ppl_ccnews", "entropy_ccnews", "CC-News (far-OOD)", "#e8b44a")):
        t = gap_traj(arms, arm, key_p, key_e)
        if not t:
            continue
        g, v = t
        ax.plot(g, v, color=col, marker="o", ms=2.8, lw=1.2, label=lab)
    ax.axhline(0, color="0.2", lw=0.8)
    ax.set_xlabel("generation")
    ax.set_title("Register grading, SmolLM2 1.7B / greedy")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.22), frameon=False,
              ncol=2, fontsize=6.0, columnspacing=0.9, handletextpad=0.4,
              handlelength=1.4)
    fig.suptitle("The gap opens from $\\approx$0 in all four cells; magnitude "
                 "is set by decoder severity, not scale", y=1.03)
    fig.savefig(os.path.join(out, "fig4_calibration_gap.pdf"))
    plt.close(fig)


# --------------------------------------------------------------- fig 3 ------
def fig_effect_size(arms, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    w, n = 0.2, len(ARMS)
    xs = range(len(ROWS) + 1)
    keys = ROWS + ["lambada_openai_ppl"]
    for j, arm in enumerate(ARMS):
        vals, pos = [], []
        for i, m in enumerate(keys):
            p = pct_traj(arms, arm, m)
            if not p:
                continue
            vals.append(abs(p[1][-1])); pos.append(i + (j - (n - 1) / 2) * w)
        colour, filled = ARM_STYLE[arm]
        ax.bar(pos, vals, width=w, color=colour, alpha=1.0 if filled else 0.45,
               label=f"{ARM_LABEL[arm]} (gen {MAXGEN[arm]})")
    ax.set_yscale("log")
    ax.axhline(2, color="0.35", ls="--", lw=0.9)
    ax.text(0.15, 2.35, "2%", fontsize=6.5, color="0.35")   # inside the axes
    ax.axvline(len(ROWS) - 4.5, color="0.3", ls=":", lw=0.9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[m] for m in keys], rotation=32, ha="right")
    ax.set_ylabel("$|\\Delta|$ vs gen 0 (%, log scale)")
    ax.set_title("Collapse signals move 12-375%; every benchmark movement "
                 "outside the most severe cell stays under 2.2%")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(os.path.join(out, "fig3_effect_size.pdf"))
    plt.close(fig)


# --------------------------------------------------------------- fig 7 ------
def abruptness(arms, arm, m):
    t = traj(arms, arm, m)
    if not t:
        return None
    _, v = t
    tot = abs(v[-1] - v[0])
    if tot == 0:
        return None
    return 100 * max(abs(v[i + 1] - v[i]) for i in range(len(v) - 1)) / tot


def fig_abruptness(arms, out):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    keys = ROWS + ["lambada_openai_ppl"]
    w, n = 0.2, len(ARMS)
    for j, arm in enumerate(ARMS):
        vals, pos = [], []
        for i, m in enumerate(keys):
            a = abruptness(arms, arm, m)
            if a is None:
                continue
            vals.append(a); pos.append(i + (j - (n - 1) / 2) * w)
        colour, filled = ARM_STYLE[arm]
        ax.bar(pos, vals, width=w, color=colour,
               alpha=1.0 if filled else 0.45, label=ARM_LABEL[arm])
    ax.set_yscale("symlog", linthresh=100)   # keeps 16-45% legible next to 640%
    ax.axhline(100, color="0.3", ls="--", lw=0.9)
    ax.text(0.1, 108, "100% = non-monotonic", fontsize=6.5, color="0.35")
    ax.set_xticks(list(range(len(keys))))
    ax.set_xticklabels([LABEL[m] for m in keys], rotation=32, ha="right")
    ax.set_ylabel("abruptness (max per-gen $|\\Delta|$ / total $|\\Delta|$, %)")
    ax.set_title("Leading signals have gradual onset; benchmark movement is "
                 "non-monotonic")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.savefig(os.path.join(out, "fig7_abruptness.pdf"))
    plt.close(fig)


# --------------------------------------------------------------- main -------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", default="figures")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    arms, p0b = load(a.datadir)
    fig_first_signal(arms, a.out);   print("wrote fig2_first_signal.pdf")
    fig_transient_arc(arms, a.out);  print("wrote fig1_transient_arc.pdf")
    fig_calibration_gap(arms, a.out); print("wrote fig4_calibration_gap.pdf")
    fig_effect_size(arms, a.out);    print("wrote fig3_effect_size.pdf")
    fig_abruptness(arms, a.out);     print("wrote fig7_abruptness.pdf")


if __name__ == "__main__":
    main()
