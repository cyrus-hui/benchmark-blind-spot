#!/usr/bin/env python3
"""
rate_shape.py  --  RQ2: does collapse accelerate, decelerate, or stay linear?

Per-metric trajectory shape over generations 0..7. For each metric we ask three
things, in increasing order of strength:

  (1) model-free:  is the per-generation increment |Delta| growing (accelerating),
                   shrinking (decelerating), or flat (linear)? -> early/late ratio
  (2) curvature:   sign of the quadratic term, and does quadratic beat linear by
                   AICc (> 2)?  -> is the curvature real or noise?
  (3) functional:  best fit among {linear, saturating-exp, exp-growth} by AICc.

For PPL we ALSO fit log(PPL) = cross-entropy loss, because A Tale of Tails predicts
loss linear in generation count n; PPL is exp(loss), so the linearity claim must be
tested in log space.

Usage:
    python rate_shape.py smollm2        # per-seed, loads metrics.json from $SCRATCH
    python rate_shape.py gpt2           # per-seed
    python rate_shape.py --demo {smollm2|gpt2}   # uses embedded cross-seed means (no cluster)

Read-only. Prints a table; writes nothing.
"""
import sys, os, json, glob, warnings
import numpy as np
from scipy.optimize import curve_fit

warnings.simplefilter("ignore")  # curve_fit covariance / overflow chatter on 8 pts

# ----------------------------------------------------------------------------- #
# Embedded cross-seed MEANS (gens 0..7) for --demo and sanity-checking.
# These are the published tables in results.md; the per-seed loader supersedes them.
# ----------------------------------------------------------------------------- #
MEANS = {
    "gpt2": {
        "n_seeds": 4,
        "perplexity":   [23.7262,25.7434,27.3023,28.4217,29.2690,29.9302,30.4694,30.8962],
        "distinct2":    [0.5816,0.5704,0.5634,0.5674,0.5683,0.5676,0.5643,0.5652],
        "emb_variance": [0.9452,0.9475,0.9474,0.9484,0.9476,0.9485,0.9488,0.9486],
        "kl_unigram":   [1.8132,1.9087,1.9562,1.9689,1.9836,2.0079,2.0096,2.0249],
        "mauve":        [0.9246,0.9271,0.9150,0.9184,0.8936,0.9023,0.9041,0.8916],
    },
    "smollm2": {
        "n_seeds": 5,
        "perplexity":   [12.0136,12.4217,12.7106,12.9252,13.0994,13.2482,13.3817,13.5092],
        "distinct2":    [0.5463,0.5203,0.5064,0.4961,0.4890,0.4825,0.4758,0.4717],
        "emb_variance": [0.9579,0.9591,0.9593,0.9593,0.9595,0.9595,0.9601,0.9597],
        "kl_unigram":   [1.0631,1.1360,1.1862,1.2186,1.2422,1.2653,1.2810,1.2948],
        "mauve":        [0.8922,0.8896,0.8711,0.8402,0.8386,0.8076,0.8017,0.7920],
    },
}

# Metrics whose rate-shape adjudicates A Tale of Tails (linear) vs Entropy Collapse
# (abrupt) -- these are the signalling metrics. EmbVar/benchmarks are flat/noise.
SIGNAL_METRICS = ["perplexity", "distinct2", "kl_unigram", "mauve"]

MODEL_DIR = {"gpt2": "pilot_gpt2m",   "smollm2": "smollm2_1.7b",
             "greedy": "gpt2m_greedy", "nucleus": "gpt2m_nucleus"}
SEEDS     = {"gpt2": [43,44,45,46],   "smollm2": [43,44,45,46,47],
             "greedy": [43,44,45],    "nucleus": [43,44,45]}
assert set(MODEL_DIR) == set(SEEDS), "MODEL_DIR / SEEDS key mismatch"
GENS      = list(range(int(__import__("os").environ.get("MAXGEN", "7")) + 1))
print(f"[rate_shape] gens {GENS[0]}-{GENS[-1]}")

# ----------------------------------------------------------------------------- #
# Fit models
# ----------------------------------------------------------------------------- #
def f_lin(x, a, b):              return a + b*x
def f_quad(x, a, b, c):          return a + b*x + c*x*x
def f_sat(x, a, b, c):           return a + b*(1.0 - np.exp(-c*x))   # decelerating
def f_exp(x, a, b, c):           return a + b*np.exp(c*x)            # accelerating if c>0

def aicc(y, yhat, k):
    n = len(y)
    rss = float(np.sum((y - yhat)**2))
    if rss <= 0: rss = 1e-12
    aic = n*np.log(rss/n) + 2*k
    denom = (n - k - 1)
    return aic + (2*k*(k+1)/denom if denom > 0 else 1e9)

def r2(y, yhat):
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y-np.mean(y))**2))
    return 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

def try_fit(func, x, y, p0):
    try:
        popt, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)
        yhat = func(x, *popt)
        return popt, yhat, aicc(y, yhat, len(popt)), r2(y, yhat)
    except Exception:
        return None, None, 1e18, -1e18

def shape_of_series(y):
    """Return dict of diagnostics for one trajectory (length-8 array)."""
    x = np.asarray(GENS, float); y = np.asarray(y, float)
    d1 = np.diff(y)                          # increments
    # (1) model-free accel/decel: |Delta| early (g0-3) vs late (g4-7)
    early = np.mean(np.abs(d1[:len(d1)//2])); late = np.mean(np.abs(d1[len(d1)//2:]))
    ratio = late/early if early > 0 else np.nan   # <1 decel, >1 accel
    # (2) curvature via quad
    span = max(np.ptp(y), 1e-9)
    _, _, a_lin,  r_lin  = try_fit(f_lin,  x, y, [y[0], (y[-1]-y[0])/max(len(x)-1,1)])
    pq, _, a_quad, r_quad = try_fit(f_quad, x, y, [y[0], (y[-1]-y[0])/max(len(x)-1,1), 0.0])
    quad_c = pq[2] if pq is not None else np.nan
    quad_beats = (a_lin - a_quad) > 2.0
    # (3) functional family
    rng = y[-1]-y[0]
    _, _, a_sat, r_sat = try_fit(f_sat, x, y, [y[0], rng if rng!=0 else 1.0, 0.5])
    p_exp, _, a_exp, r_exp = try_fit(f_exp, x, y, [y[0], 0.01*np.sign(rng) if rng!=0 else 0.01, 0.2])
    fam = min([("linear",a_lin),("saturating-exp",min(a_sat,a_exp))], key=lambda t:t[1])[0]
    if fam == "exp-growth" and p_exp is not None:
        print("    [exp fit] b=%+.4g c=%+.4g -> %s" % (p_exp[1], p_exp[2], "saturating-shaped (c<0)" if p_exp[2] < 0 else "ACCELERATING (c>0)"))
    # verdict
    if not quad_beats:
        verdict = "linear"
    else:
        # curvature relative to trend direction: same sign as slope => accelerating
        slope = rng
        verdict = "accelerating" if (quad_c*slope > 0) else "decelerating"
    return dict(ratio=ratio, r_lin=r_lin, quad_c=quad_c, quad_beats=quad_beats,
                fam=fam, r_sat=r_sat, r_exp=r_exp, verdict=verdict)

# ----------------------------------------------------------------------------- #
# Per-seed loader (real run on Narval)
# ----------------------------------------------------------------------------- #
def load_per_seed(model):
    scratch = os.environ.get("SCRATCH")
    if not scratch:
        sys.exit("SCRATCH not set -- run inside the Narval env, or use --demo.")
    base = os.path.join(scratch, "collapse", MODEL_DIR[model])
    data = {m: [] for m in SIGNAL_METRICS}   # data[metric] -> list of per-seed length-8 arrays
    for s in SEEDS[model]:
        series = {m: [] for m in SIGNAL_METRICS}
        for g in GENS:
            mp = os.path.join(base, f"seed{s}", f"gen{g}", "metrics.json")
            with open(mp) as fh: j = json.load(fh)
            for m in SIGNAL_METRICS:
                series[m].append(float(j[m]))
        for m in SIGNAL_METRICS:
            data[m].append(np.asarray(series[m], float))
    return data

def aggregate_per_seed(model, data):
    """Fit each seed separately; report seed-agreement + mean R^2."""
    print(f"\n=== {model}  per-seed rate-shape (n={len(SEEDS[model])} seeds) ===")
    hdr = f"{'metric':14} {'verdict (modal)':16} {'agree':6} {'late/early Δ':12} {'R2_lin':7} {'best family':12}"
    print(hdr); print("-"*len(hdr))
    for m in SIGNAL_METRICS:
        diags = [shape_of_series(arr) for arr in data[m]]
        verdicts = [d["verdict"] for d in diags]
        modal = max(set(verdicts), key=verdicts.count)
        agree = f"{verdicts.count(modal)}/{len(verdicts)}"
        ratio = np.nanmean([d["ratio"] for d in diags])
        rlin  = np.mean([d["r_lin"] for d in diags])
        fams  = [d["fam"] for d in diags]
        fam   = max(set(fams), key=fams.count)
        print(f"{m:14} {modal:16} {agree:6} {ratio:<12.3f} {rlin:<7.3f} {fam:12}")
    # log-PPL test (loss linearity, A Tale of Tails)
    print("\n-- log(PPL) = loss, tested for linearity in n (A Tale of Tails) --")
    for arr in data["perplexity"]:
        pass
    logs = [np.log(arr) for arr in data["perplexity"]]
    dl = [shape_of_series(a) for a in logs]
    v = [d["verdict"] for d in dl]; modal = max(set(v), key=v.count)
    ratio = np.nanmean([d["ratio"] for d in dl]); rlin = np.mean([d["r_lin"] for d in dl])
    print(f"{'log-PPL':14} {modal:16} {v.count(modal)}/{len(v):<4} late/early={ratio:.3f}  R2_lin={rlin:.3f}")

def demo(model):
    print(f"\n=== {model}  rate-shape on CROSS-SEED MEANS (demo; n={MEANS[model]['n_seeds']}) ===")
    hdr = f"{'metric':14} {'verdict':14} {'late/early Δ':12} {'R2_lin':7} {'quad_c':11} {'beats_lin':9} {'best family':12}"
    print(hdr); print("-"*len(hdr))
    for m in SIGNAL_METRICS:
        d = shape_of_series(MEANS[model][m])
        print(f"{m:14} {d['verdict']:14} {d['ratio']:<12.3f} {d['r_lin']:<7.3f} "
              f"{d['quad_c']:<+11.5f} {str(d['quad_beats']):9} {d['fam']:12}")
    d = shape_of_series(np.log(MEANS[model]["perplexity"]))
    print(f"\n{'log-PPL':14} {d['verdict']:14} {d['ratio']:<12.3f} {d['r_lin']:<7.3f} "
          f"{d['quad_c']:<+11.5f} {str(d['quad_beats']):9} {d['fam']:12}")
    print("\nlate/early Δ < 1.0 => decelerating ; ~1.0 => linear ; > 1.0 => accelerating")

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--demo" in args:
        args.remove("--demo")
        model = args[0] if args else "smollm2"
        demo(model)
    else:
        model = args[0] if args else "smollm2"
        aggregate_per_seed(model, load_per_seed(model))
