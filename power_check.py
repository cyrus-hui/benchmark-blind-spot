#!/usr/bin/env python3
"""
Power check for model collapse pilot (seeds 42, 43, 44 x gens 0-7).
Computes per-metric cross-seed mean, std, delta, and SNR (delta/sigma).
Also prints a lag-readiness table: which metrics are likely resolvable
at p<0.05 with 3 seeds vs needing 5.
"""

import json
import os
import glob
import numpy as np
from pathlib import Path

BASE = Path(os.environ.get("SCRATCH", "/scratch/cyrh")) / "collapse/pilot_gpt2m"
SEEDS = [42, 43, 44]
GENS = list(range(int(__import__("os").environ.get("MAXGEN", "7")) + 1))

# ── 1. Load all metrics ───────────────────────────────────────────────────────

def load_metrics(seed, gen):
    m_path = BASE / f"seed{seed}/gen{gen}/metrics.json"
    b_glob = str(BASE / f"seed{seed}/gen{gen}/benchmarks/*/*results*.json")
    
    data = {}
    
    if m_path.exists():
        with open(m_path) as f:
            m = json.load(f)
        data.update({
            "perplexity":   m.get("perplexity"),
            "distinct2":    m.get("distinct2"),
            "emb_variance": m.get("emb_variance"),
            "kl_unigram":   m.get("kl_unigram"),
            "mauve":        m.get("mauve"),
        })
    
    b_files = glob.glob(b_glob)
    if b_files:
        with open(b_files[0]) as f:
            b = json.load(f)
        results = b.get("results", {})
        hs = results.get("hellaswag", {})
        la = results.get("lambada_openai", {})
        data.update({
            "hellaswag_acc_norm": hs.get("acc_norm,none"),
            "lambada_acc":        la.get("acc,none"),
            "lambada_ppl":        la.get("perplexity,none"),
        })
    
    return data if data else None

# higher_is_better: False means collapse = rising value
METRIC_DIRECTION = {
    "perplexity":        "higher",   # rising = collapse
    "distinct2":         "lower",    # falling = collapse
    "emb_variance":      "lower",    # falling = collapse
    "kl_unigram":        "higher",   # rising = collapse
    "mauve":             "lower",    # falling = collapse
    "hellaswag_acc_norm":"lower",    # falling = collapse
    "lambada_acc":       "lower",    # falling = collapse
    "lambada_ppl":       "higher",   # rising = collapse
}

# Build matrix: metrics x seeds x gens
all_metrics = list(METRIC_DIRECTION.keys())
data = {m: {s: {} for s in SEEDS} for m in all_metrics}

missing = []
for seed in SEEDS:
    for gen in GENS:
        row = load_metrics(seed, gen)
        if row is None:
            missing.append((seed, gen))
            continue
        for m in all_metrics:
            if row.get(m) is not None:
                data[m][seed][gen] = row[m]

if missing:
    print(f"WARNING: missing data for (seed, gen): {missing}\n")

# ── 2. Cross-seed stats per generation ───────────────────────────────────────

print("=" * 72)
print("CROSS-SEED STATS (mean ± std across seeds 42/43/44)")
print("=" * 72)

stats = {}   # metric -> gen -> {mean, std, n}

for m in all_metrics:
    stats[m] = {}
    direction = METRIC_DIRECTION[m]
    print(f"\n{'─'*60}")
    print(f"  {m}  ({'↑ collapse' if direction == 'higher' else '↓ collapse'})")
    print(f"  {'Gen':>4}  {'Mean':>10}  {'Std':>10}  {'N':>3}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*3}")
    for gen in GENS:
        vals = [data[m][s][gen] for s in SEEDS if gen in data[m][s]]
        if len(vals) < 2:
            print(f"  {gen:>4}  {'N/A':>10}  {'N/A':>10}  {len(vals):>3}")
            stats[m][gen] = None
            continue
        mn, sd = np.mean(vals), np.std(vals, ddof=1)
        stats[m][gen] = {"mean": mn, "std": sd, "n": len(vals)}
        print(f"  {gen:>4}  {mn:>10.4f}  {sd:>10.4f}  {len(vals):>3}")

# ── 3. SNR table: delta/sigma per consecutive generation ─────────────────────

print("\n\n" + "=" * 72)
print("SIGNAL-TO-NOISE RATIO  (|Δ mean| / σ  per consecutive generation)")
print("Higher = easier to detect with few seeds")
print("Rule of thumb: SNR > 2 at gen2 → 3 seeds likely sufficient")
print("               SNR < 1 at gen2 → consider 5 seeds")
print("=" * 72)

header = f"  {'Metric':<22}" + "".join(f"  {'0→1':>6}  {'1→2':>6}  {'2→3':>6}  {'3→4':>6}")
print(header)
print("  " + "─" * 68)

snr_at_2 = {}
for m in all_metrics:
    row_str = f"  {m:<22}"
    snrs = []
    for g0, g1 in [(0,1),(1,2),(2,3),(3,4)]:
        s0 = stats[m].get(g0)
        s1 = stats[m].get(g1)
        if s0 and s1 and s0["std"] > 0:
            delta = abs(s1["mean"] - s0["mean"])
            sigma = (s0["std"] + s1["std"]) / 2   # avg sigma across the two gens
            snr = delta / sigma
            snrs.append(snr)
            row_str += f"  {snr:>6.2f}"
        else:
            snrs.append(None)
            row_str += f"  {'N/A':>6}"
    print(row_str)
    # SNR at gen2 = the 1→2 transition (index 1)
    snr_at_2[m] = snrs[1] if len(snrs) > 1 else None

# ── 4. Power check summary ───────────────────────────────────────────────────

print("\n\n" + "=" * 72)
print("POWER CHECK SUMMARY  (based on gen 1→2 SNR)")
print("=" * 72)
print(f"  {'Metric':<22}  {'SNR@gen2':>10}  {'Verdict':>30}")
print("  " + "─" * 66)

problem_metrics = []
for m in all_metrics:
    snr = snr_at_2.get(m)
    if snr is None:
        verdict = "insufficient data"
    elif snr > 2.0:
        verdict = "✓  3 seeds likely sufficient"
    elif snr > 1.0:
        verdict = "⚠  borderline — 5 seeds safer"
        problem_metrics.append(m)
    else:
        verdict = "✗  5 seeds needed"
        problem_metrics.append(m)
    snr_str = f"{snr:.2f}" if snr is not None else "N/A"
    print(f"  {m:<22}  {snr_str:>10}  {verdict:>30}")

print()
if not problem_metrics:
    print("  → DECISION: 3 seeds is sufficient for the main condition.")
    print("    Proceed with Phase 3 config as planned.")
else:
    print(f"  → Problem metrics: {problem_metrics}")
    print("    Consider 5 seeds on the main condition.")
    print("    If compute is tight: 5 seeds for top-k main; 3 seeds for ablations.")

# ── 5. Total collapse magnitude (gen0 → gen7) ────────────────────────────────

print("\n\n" + "=" * 72)
print("TOTAL COLLAPSE MAGNITUDE  (gen 0 → gen 7, mean across seeds)")
print("=" * 72)
print(f"  {'Metric':<22}  {'Gen0 mean':>10}  {'Gen7 mean':>10}  {'% change':>10}  {'Direction':>12}")
print("  " + "─" * 68)

for m in all_metrics:
    s0 = stats[m].get(0)
    s7 = stats[m].get(7)
    if s0 and s7:
        pct = (s7["mean"] - s0["mean"]) / abs(s0["mean"]) * 100
        direction = METRIC_DIRECTION[m]
        expected_sign = "+" if direction == "higher" else "-"
        actual_sign = "+" if pct > 0 else "-"
        ok = "✓" if expected_sign == actual_sign else "✗ WRONG DIR"
        print(f"  {m:<22}  {s0['mean']:>10.4f}  {s7['mean']:>10.4f}  {pct:>+10.1f}%  {ok:>12}")
    else:
        print(f"  {m:<22}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}")

print("\nDone.")
