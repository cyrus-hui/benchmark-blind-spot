"""Shared logic for the Phase 5.4 detector evaluation.

Every rule implemented here is pre-registered in decisions.md, Sept 17 (c):
  - detectors: dcal, pinc, pslope, sigma_sub, sigma_full
  - thresholds: max over control chains (single tau, or one tau per window for pslope)
  - statistics: AUC (Mann-Whitney, ties 0.5), mean censored latency, TPR at tau, FPR at TPR = 1
  - "beats": lexicographic AUC -> latency -> FPR@TPR1, a difference counting only if its
    sign survives deletion of any single chain (jackknife-stable); else tie at that level
Imported by compute_pplslope_thresholds.py and detector_eval_5p4.py. No script
re-implements any of it.
"""
import json, math, pathlib, sys

GENS = list(range(8))
CENSOR = 8                      # latency assigned to "not flagged by gen 7"
EVAL_SEEDS = [43, 44, 45]       # the chains every detector is compared on
RATIOS = [0, 25, 50, 75, 100]
DECODERS = ["topk", "greedy"]

# perplexity / entropy come from the model weights; r = 0 is ONE set of models
WEIGHTS_ARM = {
    "topk":   {0: "control_gpt2m", 25: "mix25_topk", 50: "mix50_topk", 75: "mix75_topk", 100: "pilot_gpt2m"},
    "greedy": {0: "control_gpt2m", 25: "mix25_greedy", 50: "mix50_greedy", 75: "mix75_greedy", 100: "gpt2m_greedy"},
}
# distinct2 / kl_unigram / SIGMA come from eval completions, which depend on the decoder
COMPLETIONS_ARM = {
    "topk":   dict(WEIGHTS_ARM["topk"]),
    "greedy": {**WEIGHTS_ARM["greedy"], 0: "control_gpt2m_greedyeval"},
}
# threshold construction sets (decisions.md Sept 15 (b), Sept 17, Sept 17 (c))
WEIGHTS_TAU = ("control_gpt2m", [43, 44, 45, 46])          # dcal, pinc, pslope: decoder-independent
COMPLETIONS_TAU = {"topk": ("control_gpt2m", [43, 44, 45, 46]),
                   "greedy": ("control_gpt2m_greedyeval", [43, 44, 45])}


def is_control(arm):
    return arm.startswith("control_")


class Reader:
    """All disk access. An exposed arm cannot be opened unless allow_exposed is True
    (the authorised grid read) or names that arm explicitly (arms already read and published)."""

    def __init__(self, root, allow_exposed=False):
        self.root = pathlib.Path(root)
        self.allow_exposed = allow_exposed
        self.opened = set()

    def _guard(self, arm):
        ok = self.allow_exposed is True or (isinstance(self.allow_exposed, (set, frozenset)) and arm in self.allow_exposed)
        if not is_control(arm) and not ok:
            sys.exit(f"REFUSED: {arm} is not a control arm and the grid read is not authorised")
        self.opened.add(arm)

    def metrics(self, arm, seed, keys):
        self._guard(arm)
        out = []
        for g in GENS:
            p = self.root / arm / f"seed{seed}" / f"gen{g}" / "metrics.json"
            if not p.is_file():
                sys.exit(f"STOP: missing {p}")
            d = json.load(open(p))
            for k in keys:
                if k not in d:
                    sys.exit(f"STOP: key {k} absent in {p}; keys = {sorted(d)}")
            out.append({k: float(d[k]) for k in keys})
        return out

    def sigma(self, arm, seed):
        self._guard(arm)
        p = self.root / "sigma" / arm / f"seed{seed}.json"
        if not p.is_file():
            sys.exit(f"STOP: missing SIGMA record {p}")
        r = json.load(open(p))
        if r.get("n_k") != 1000 or len(r.get("gens", {})) != 8:
            sys.exit(f"STOP: SIGMA record shape wrong in {p}")
        for g in GENS:
            for k in ("dU_sub", "dU_full"):
                if k not in r["gens"][str(g)]:
                    sys.exit(f"STOP: key {k} absent at gen {g} in {p}; keys = {sorted(r['gens'][str(g)])}")
        return r


# ---------------------------------------------------------------- series (index = generation)

def dcal_series(m):
    gap = [math.log(x["perplexity"]) - x["entropy"] for x in m]
    return [v - gap[0] for v in gap]


def pinc_series(m):
    lp = [math.log(x["perplexity"]) for x in m]
    return [v - lp[0] for v in lp]


def ols_slope(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def pslope_series(m):
    """b(g) = least-squares slope of log PPL on generation over gens 1..g; defined for g >= 2.
    Gen 0 is excluded: it is fitted to a different corpus from every later generation of every
    arm that contains real data, and including it hands the control a step, not a trend."""
    lp = [math.log(x["perplexity"]) for x in m]
    return [None, None] + [ols_slope(list(range(1, g + 1)), lp[1:g + 1]) for g in range(2, 8)]


def sigma_series(rec, key):
    """Collapse direction positive: -dU(g)."""
    return [-float(rec["gens"][str(g)][key]) for g in GENS]


# ---------------------------------------------------------------- thresholds

def tau_single(series_list):
    return max(s[g] for s in series_list for g in range(1, 8) if s[g] is not None)


def tau_window(series_list):
    return {g: max(s[g] for s in series_list) for g in range(2, 8)}


def first_flag(series, tau):
    for g in range(1, 8):
        if series[g] is None:
            continue
        t = tau[g] if isinstance(tau, dict) else tau
        if series[g] > t:
            return g
    return CENSOR


# ---------------------------------------------------------------- statistics

def auc(pos, neg):
    tot = sum(1.0 if p > n else (0.5 if p == n else 0.0) for p in pos for n in neg)
    return tot / (len(pos) * len(neg))


def stats(det, pos_ids, neg_ids):
    """det = {"score": {chain: s7}, "lat": {chain: latency}}"""
    ps = [det["score"][c] for c in pos_ids]
    ns = [det["score"][c] for c in neg_ids]
    lat = [det["lat"][c] for c in pos_ids]
    thr = min(ps)
    return {
        "auc": auc(ps, ns),
        "mean_lat": sum(lat) / len(lat),
        "tpr": sum(l <= 7 for l in lat) / len(lat),
        "n_flag": sum(l <= 7 for l in lat),
        "fpr_at_tpr1": sum(n >= thr for n in ns) / len(ns),
        "neg_flagged": sum(det["lat"][c] <= 7 for c in neg_ids),
    }


LEVELS = [("auc", +1), ("mean_lat", -1), ("fpr_at_tpr1", -1)]   # sign: +1 higher is better
EPS = 1e-12


def _sgn(x):
    return 0 if abs(x) < EPS else (1 if x > 0 else -1)


def compare(a, b, pos_ids, neg_ids):
    """Registered 'beats' rule. Returns (verdict for a, level, note).
    verdict in WIN / LOSS / TIE / CEILING."""
    sa, sb = stats(a, pos_ids, neg_ids), stats(b, pos_ids, neg_ids)
    if sa["auc"] == 1 and sb["auc"] == 1 and sa["mean_lat"] == 1 and sb["mean_lat"] == 1:
        return "CEILING", "-", "both at AUC 1.0 and latency 1.0 on every positive chain"
    notes = []
    for key, sign in LEVELS:
        d = _sgn(sign * (sa[key] - sb[key]))
        if d == 0:
            notes.append(f"{key}: equal")
            continue
        stable = True
        for drop in list(pos_ids) + list(neg_ids):
            p2 = [c for c in pos_ids if c != drop]
            n2 = [c for c in neg_ids if c != drop]
            if not p2 or not n2:
                continue
            d2 = _sgn(sign * (stats(a, p2, n2)[key] - stats(b, p2, n2)[key]))
            if d2 != d:
                stable = False
                notes.append(f"{key}: difference not jackknife-stable (dropping {drop})")
                break
        if stable:
            return ("WIN" if d > 0 else "LOSS"), key, "; ".join(notes + [f"{key}: decided, jackknife-stable"])
    return "TIE", "-", "; ".join(notes)


def better_of(variants, default, pos_ids, neg_ids):
    """Rank a baseline family on its better variant; a tie goes to the default."""
    names = list(variants)
    assert len(names) == 2 and default in names
    other = [n for n in names if n != default][0]
    v, _, _ = compare(variants[other], variants[default], pos_ids, neg_ids)
    return other if v == "WIN" else default
