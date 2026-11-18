#!/usr/bin/env python3
r"""7.5 early forecasting: the registered read. Written blind, before any held-out chain or value exists.

REGISTRATION. decisions.md sec. 73, frozen Nov 14 2026. Project notes are not in git, so this docstring is the
git-timestamped copy: committed and pushed with the pipeline code before the first 7.5 job, run once unmodified
after every chain is complete, gated and scored, its log committed unedited.

  --selftest  arithmetic and verdict logic on synthetic numbers. No disk.
  --fit       ALREADY-READ arms only (fit set, controls, greedy-grid gen-1 values): the pinned map, the noise
              scales and the predictions on record for C. Run before the first job; commit its log.
  --check     validate every input and the canaries; print counts only, NO held-out value.
  (default)   canaries first (stop on any failure), then 73.8 on the primary, then the rest, in order.
  --rehearse  mock tree only (refuses a root without "mock" in its path; skips the git and gate checks).

The read refuses unless: this script, src/detectors.py, src/generate.py and fc75_conditions.json are committed and
unmodified; HEAD is on origin; logs/fc75/gate_g0.json, gate_all.json (18 keys) and, if any C condition is
registered, gate_topkeval.json are PASS.

73.1 Question. Does Delta(1), the calibration-gap rise after ONE generation, forecast gen-7 damage on conditions
  the map never saw, better than (a) knowing the synthetic fraction r and (b) the same map built on P-inc(1)?

73.2 Quantities, per chain (seed s), all against the chain's own gen 0:
  x_D = Delta_cal(1) - Delta_cal(0)   in-domain, src/detectors.dcal_series      the candidate
  x_P = ln PPL(1) - ln PPL(0)          src/detectors.pinc_series                 baseline (b)
  x_r = nominal synthetic fraction, percent                                      baseline (a)
  y_K = KL-unigram(7) - KL-unigram(0)  PRIMARY damage
  y_D = Distinct-2(0) - Distinct-2(7)  secondary
  y_L = ln PPL(7) - ln PPL(0)          secondary; same instrument as baseline (b), flagged
  Eval completions are ALWAYS top-k 50 at temperature 1.0: the instrument is fixed. Gen 0 is pilot_gpt2m's gen 0
  for every chain (identical weights by construction; canaried on perplexity and entropy).
  Unit = the CONDITION: the mean over seeds 43, 44, 45. Seeds share real slices and are not replicates.

73.3 Fit set (already read; the fit is exploratory, only the held-out test is registered): top-k corpus, T = 1,
  r in {0, 25, 50, 75, 100} = control_gpt2m, mix25/50/75_topk, pilot_gpt2m. Top-k only: damage is measured on
  eval completions, so greedy-completion damage is on another scale, and under greedy the gen-7 damage is
  saturated at every exposed ratio (Distinct-2 fall 0.092 / 0.092 / 0.078 / 0.088 at r = 25 / 50 / 75 / 100).

73.4 Predictor (pinned). For input k in {D, r, P}: the piecewise-linear interpolant through the five fit
  condition means (xbar_k, ybar), ordered by xbar_k. Outside the fitted x-range the nearest end segment is
  continued linearly and the condition is OUT OF RANGE. No free parameter. The candidate is scored under this
  form only. Each baseline is scored under whichever of this form and the least-squares line through the same
  five means has the lower leave-one-ratio-out error on the fit set for the primary (already read, so the choice
  is fixed now): interpolation for r-only (0.020 vs 0.030), the line for P-inc (0.022 vs 0.030). The other form
  is printed as a sensitivity and never ranked.

73.5 Held-out conditions (n = 3 chains each, gens 1-7):
  A  new ratio, T = 1:            mix10_topk (r = 10), mix90_topk (r = 90)
  B  sampled corpus temperature:  fc75_r25_t075, fc75_r25_t050, fc75_r50_t075, fc75_r50_t050
     (top-k 50; the temperature applies to the TRAINING corpus only)
  C  T -> 0 (greedy corpus):      the existing mix25/50/75_greedy and gpt2m_greedy chains, with gen-7 damage
     re-measured on top-k completions (eval-only pass into *_topkeval). The set is the conditions whose three
     gen-7 checkpoints exist at prepare time, pinned in fc75_conditions.json before the pass runs.
  P  scope probe, reported only:  fc75_r100_t125 (T = 1.25)
  B and C are the TEMPERATURE-TYPE conditions: severity changes that r does not see.

73.6 In range. A held-out condition is IN RANGE iff its xbar_D lies inside the fitted range of xbar_D. This uses
  gen-1 values only. Out-of-range conditions are reported under the continuation rule and never ranked.

73.7 Error and noise scale. e_k(c) = |prediction_k(c) - ybar(c)|, in the damage's own units. The noise scale N is
  the largest error the map makes on real-data control conditions: over gens 1-7 and every 3-seed subset of
  control_gpt2m seeds 43-46 (28 values), max |mean control shift - the map's r = 0 anchor|. Controls only.
  Expected N_K 0.0497, N_D 0.0068, N_L 0.0048.

73.8 Verdict, on the primary y_K with the candidate Delta(1).
  (i)  ACCURATE iff e_D(c) <= N at EVERY in-range held-out condition of types A, B and C.
  (ii) Against baseline b: the CONTESTED set R_b = in-range temperature-type conditions where the two
       predictions differ by more than N (known at gen 1, before the damage exists).
       UNCONTESTED iff |R_b| < 2.
       WIN  iff e_D < e_b at every condition of R_b and mean(e_b - e_D) over R_b >= N.
       LOSS iff e_D > e_b at every condition of R_b and mean(e_D - e_b) over R_b >= N.
       TIE otherwise.
  PASS           iff ACCURATE and WIN against r-only and WIN against P-inc.
  FAIL           iff not ACCURATE, or LOSS or TIE against either baseline.
  NOT EVALUABLE  otherwise (accurate, nothing lost or tied, but a baseline was never contested).
  A-conditions count toward (i) only: within a decoder Delta(1) is a smooth function of r, so new ratios cannot
  separate the predictors (leave-one-ratio-out on the fit set: errors 0.017 vs 0.020, both inside N_K).

73.9 Registered alternative: S-ROBUST (sharpness-robust at interior ratios). It is what the r-only predictor
  asserts. HOLDS iff |ybar(c) - ybar_fit(r_c)| <= N at every temperature-type condition with r <= 50.
  Reported on all three measures, whatever 73.8 reads. Its complement at r >= 75 is descriptive.

73.10 Secondary, reported whatever they show, never overriding 73.8:
  (a) the same components on y_D and y_L. For y_L the C values are ALREADY READ (perplexity is decoder-
      independent) and are shown but excluded from the y_L components.
  (b) a second candidate, Delta(2) (inputs at gen 2, baseline (b) at gen 2), on all three measures.
  (c) out-of-range conditions and the probe P.
  (d) Spearman rank correlation of predicted and observed across in-range held-out conditions (descriptive).

73.11 Stated in advance.
  - Predictions on record for C on the primary (KL rise), from Delta(1) values already read:
      mix25_greedy  Delta(1) 0.500  in range      candidate 0.162   r-only 0.017
      mix50_greedy  Delta(1) 0.809  out of range  candidate 0.343   r-only 0.060
      mix75_greedy  Delta(1) 0.958  out of range  candidate 0.430   r-only 0.100
      gpt2m_greedy  Delta(1) 2.285  out of range  candidate 1.207   r-only 0.201
  - ALREADY KNOWN, on y_L: at r = 25 and 50 the decoder does not move gen-7 perplexity damage (top-k vs greedy:
    0.0810 vs 0.0772; 0.0894 vs 0.0888) although Delta(1) differs 3-4x (0.13 vs 0.50; 0.25 vs 0.81). There the
    candidate map misses by 0.128 and 0.393, the r-only map by 0.004 and 0.001. At r = 75 and 100 the decoder
    does move it (0.110 vs 0.146; 0.264 vs 1.252).
  - Named mechanism: under greedy at interior r, Delta FALLS after gen 1 (r = 25: 0.50 -> 0.14 by gen 7) while
    under top-k it rises (0.13 -> 0.18). Gen 1 trains on text from the WikiText-2-tuned gen-0 model; later
    generations do not. The map may OVER-predict damage for sharper generators.
  - Forecast on record: FAIL on 73.8 with S-ROBUST holding is more likely than PASS (PASS about 10-15%).
  - Expected directions: lower T raises Delta(1) at fixed r; A cannot separate the predictors.
  - Probe P (T = 1.25): Delta_cal measures sharpening. Expectation: Delta(1) below the T = 1 value at r = 100 and
    the candidate UNDER-predicts the KL rise. Outside the verdict: the monitor was validated for T <= 1 only.
  - Baseline (b) is weak by construction here: x_P is not monotone in r on the fit set (0.071, 0.022, 0.032,
    0.051, 0.082), because the real slices carry a +0.07 register offset that synthetic text lacks.
  - Known deviation: P reaches r = 100 through the mixing hook at ratio 1.00 (a permutation of the synthetic
    corpus), not through the driver's default path.

73.12 Scale and corpus (closes 70.7.7). What carries to 6.1 (8B) and 6.2 (second corpus) unchanged is the form,
  the held-out types, the error, the noise-scale construction, the verdict rule and this code. The NUMERIC map is
  rebuilt from that setting's own T = 1 fit set (at least r = 0, one interior ratio and 100). The 345M map
  applied unchanged is reported as a secondary and is expected to fail (a threshold built at 1.7B already
  failed to transfer to 345M). The exact cells are registered before any chain at that setting.

73.13 Must not appear. The fit is never out-of-sample. Delta_cal never scores the forecast. An out-of-range or
  probe condition never counts toward the verdict. "Beats perplexity" is never written without 73.11's note on
  baseline (b). The y_L values at C are never presented as held-out. No 7.5 result is claimed before the read.
"""
import argparse, hashlib, itertools, json, math, os, subprocess, sys, time

SEEDS, CTL_SEEDS, GENS = (43, 44, 45), (43, 44, 45, 46), range(1, 8)
FIT = [(0, "control_gpt2m"), (25, "mix25_topk"), (50, "mix50_topk"), (75, "mix75_topk"), (100, "pilot_gpt2m")]
PILOT, CTL = "pilot_gpt2m", "control_gpt2m"
CHAINS = [("mix10_topk", "A", 10, None), ("mix90_topk", "A", 90, None),            # must equal fc75_conditions.json
          ("fc75_r25_t075", "B", 25, 0.75), ("fc75_r25_t050", "B", 25, 0.50),
          ("fc75_r50_t075", "B", 50, 0.75), ("fc75_r50_t050", "B", 50, 0.50),
          ("fc75_r100_t125", "P", 100, 1.25)]
C_ALLOWED = {"mix25_greedy": 25, "mix50_greedy": 50, "mix75_greedy": 75, "gpt2m_greedy": 100}
TARGETS = [("K", "KL-unigram rise at gen 7"), ("D", "Distinct-2 fall at gen 7"), ("L", "ln PPL rise at gen 7")]
# canaries: fit condition means recomputed from the public job logs on Nov 14 2026 (x_D from logs/5p4_gen1_margins.log)
FIT_CANARY = {  # r: (x_D(1), x_P(1), y_K, y_D, y_L)
    0: (0.061257, 0.07117311, 0.02756677, -0.00426479, 0.07328679),
    25: (0.131910, 0.02161087, 0.01741755, -0.00235514, 0.08099321),
    50: (0.253113, 0.03179528, 0.06034171, -0.00110547, 0.08937730),
    75: (0.393950, 0.05070582, 0.09996400, 0.00295209, 0.10981112),
    100: (0.565867, 0.08154782, 0.20066923, 0.01506547, 0.26392415)}
N_CANARY = {"K": 0.049699, "D": 0.006812, "L": 0.004834}
TAU_CANARY = {"K": 0.111117, "D": 0.010500}      # registered tau_K / tau_D: max |control shift| per chain
TOL_X, TOL_Y, TOL_ID = 2e-5, 2e-6, 1e-6
BASE_FORM = {"r": "interp", "P": "line"}        # 73.4: fixed from the fit set; the other form is a sensitivity


def die(msg):
    print(f"STOP: {msg}"); sys.exit(2)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def mean(v):
    v = list(v)
    if not v:
        die("mean of an empty list (n = 0)")
    return sum(v) / len(v)


# ------------------------------------------------------------------ the pinned predictor (73.4)

def interp(pts, x):
    """Piecewise-linear through pts ordered by x; end segments continued. Returns (y, in_range)."""
    p = sorted(pts)
    if len(p) < 2 or len({a for a, _ in p}) != len(p):
        die(f"interpolant needs >= 2 distinct x values, got {[a for a, _ in p]}")
    if x <= p[0][0]:
        (x0, y0), (x1, y1) = p[0], p[1]
    elif x >= p[-1][0]:
        (x0, y0), (x1, y1) = p[-2], p[-1]
    else:
        (x0, y0), (x1, y1) = next((a, b) for a, b in zip(p, p[1:]) if a[0] <= x <= b[0])
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0), p[0][0] <= x <= p[-1][0]


def line(pts, x):
    """Least-squares line through pts, evaluated at x."""
    mx, my = mean(a for a, _ in pts), mean(b for _, b in pts)
    sxx = sum((a - mx) ** 2 for a, _ in pts)
    if sxx == 0:
        die("least-squares line: zero variance in x")
    return my + sum((a - mx) * (b - my) for a, b in pts) / sxx * (x - mx)


def versus(rows, base, form, N):
    """73.8 (ii): the candidate against one baseline under one form, on the contested set."""
    R = [r for r in rows if r["ranked"] and abs(r["pD"] - r[f"p{base}_{form}"]) > N]
    if len(R) < 2:
        return "UNCONTESTED", [r["name"] for r in R], None
    gaps = [abs(r[f"p{base}_{form}"] - r["y"]) - r["eD"] for r in R]      # e_b - e_D
    g = mean(gaps)
    v = "WIN" if all(x > 0 for x in gaps) and g >= N else "LOSS" if all(x < 0 for x in gaps) and -g >= N else "TIE"
    return v, [r["name"] for r in R], g


def evaluate(fit, held, N, tgt, gen_in, skip_c=False):
    """All 73.8-73.10 components for one damage measure and one candidate generation."""
    kD, kP = f"xD{gen_in}", f"xP{gen_in}"
    ptsD = [(fit[r][kD], fit[r][tgt]) for r in fit]
    ptsR = [(float(r), fit[r][tgt]) for r in fit]
    ptsP = [(fit[r][kP], fit[r][tgt]) for r in fit]
    rows = []
    for c in held:
        pD, inr = interp(ptsD, c[kD])
        row = dict(name=c["name"], type=c["type"], r=c["r"], T=c["T"], xD=c[kD], xP=c[kP], y=c[tgt], pD=pD,
                   in_range=inr, eD=abs(pD - c[tgt]), already_read=bool(skip_c and c["type"] == "C"),
                   pr_interp=interp(ptsR, float(c["r"]))[0], pr_line=line(ptsR, float(c["r"])),
                   pP_interp=interp(ptsP, c[kP])[0], pP_line=line(ptsP, c[kP]))
        row["counted"] = inr and c["type"] in "ABC" and not row["already_read"]          # for (i)
        row["ranked"] = inr and c["type"] in "BC" and not row["already_read"]            # for (ii)
        rows.append(row)
    counted = [r for r in rows if r["counted"]]
    acc = bool(counted) and all(r["eD"] <= N for r in counted)
    vs = {}
    for b in ("r", "P"):
        res = {f: versus(rows, b, f, N) for f in ("interp", "line")}
        reg = BASE_FORM[b]
        vs[b] = dict(verdict=res[reg][0], form=reg, contested=res[reg][1], gap=res[reg][2],
                     other_form={f: res[f][0] for f in res if f != reg})
    if not counted:
        verdict = "NOT EVALUABLE"
    elif acc and vs["r"]["verdict"] == "WIN" and vs["P"]["verdict"] == "WIN":
        verdict = "PASS"
    elif not acc or any(vs[b]["verdict"] in ("LOSS", "TIE") for b in vs):
        verdict = "FAIL"
    else:
        verdict = "NOT EVALUABLE"
    # 73.9 S-ROBUST: temperature-type conditions against the fit value at the same ratio
    rob = [dict(name=c["name"], r=c["r"], move=abs(c[tgt] - fit[c["r"]][tgt]), already_read=bool(skip_c and c["type"] == "C"))
           for c in held if c["type"] in "BC" and c["r"] in fit]
    low = [x for x in rob if x["r"] <= 50]
    srob = "NOT EVALUABLE" if not low else "HOLDS" if all(x["move"] <= N for x in low) else "DOES NOT HOLD"
    return dict(rows=rows, accurate=acc, n_counted=len(counted), versus=vs, verdict=verdict, s_robust=srob, robust=rob)


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = float(k)
        return r
    if len(a) < 3:
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = mean(ra), mean(rb)
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den if den else None


# ------------------------------------------------------------------ loading

class Loader:
    def __init__(self, root, allowed):
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))
        import detectors
        self.D, self.root, self.allowed = detectors, root, set(allowed)
        self.R = detectors.Reader(root, allow_exposed=self.allowed)
        self.pilot = {}

    def g0(self, s):
        """pilot_gpt2m gen 0: top-k completions and scores of the gen-0 weights every chain shares."""
        if s not in self.pilot:
            self.pilot[s] = self.cell(PILOT, s, 0, ("perplexity", "entropy", "distinct2", "kl_unigram"))
        return self.pilot[s]

    def cell(self, arm, s, g, keys):
        if arm != CTL and arm not in self.allowed:
            die(f"REFUSED: {arm} is not open in this mode")
        p = os.path.join(self.root, arm, f"seed{s}", f"gen{g}", "metrics.json")
        if not os.path.isfile(p):
            die(f"missing {p}")
        d = json.load(open(p))
        for k in keys:
            if k not in d:
                die(f"key {k} absent in {p}; keys = {sorted(d)}")
        return {k: float(d[k]) for k in keys}

    def chain(self, warm, carm, s, gens=(7,)):
        """One chain: gen-1/2 inputs from the weights arm, damage at `gens` from the completions arm."""
        m = self.R.metrics(warm, s, ["perplexity", "entropy"])
        g0 = self.g0(s)
        if abs(m[0]["perplexity"] - g0["perplexity"]) > TOL_ID or abs(m[0]["entropy"] - g0["entropy"]) > TOL_ID:
            die(f"gen-0 identity canary: {warm} seed {s} does not share pilot_gpt2m's gen-0 weights")
        dc, pi = self.D.dcal_series(m), self.D.pinc_series(m)
        out = dict(xD1=dc[1], xD2=dc[2], xP1=pi[1], xP2=pi[2])
        for g in gens:
            c = self.cell(carm, s, g, ("distinct2", "kl_unigram"))
            out[g] = dict(K=c["kl_unigram"] - g0["kl_unigram"], D=g0["distinct2"] - c["distinct2"], L=pi[g])
        return out


def condition(L, name, typ, r, T, warm, carm):
    ch = [L.chain(warm, carm, s) for s in SEEDS]
    c = dict(name=name, type=typ, r=r, T=T, per_seed=ch)
    for k in ("xD1", "xD2", "xP1", "xP2"):
        c[k] = mean(x[k] for x in ch)
    for k in "KDL":
        c[k] = mean(x[7][k] for x in ch)
    return c


def build_fit(L):
    fit = {}
    for r, arm in FIT:
        fit[r] = condition(L, arm, "fit", r, 1.0, arm, arm)
    for r, (xd, xp, yk, yd, yl) in FIT_CANARY.items():
        got = (fit[r]["xD1"], fit[r]["xP1"], fit[r]["K"], fit[r]["D"], fit[r]["L"])
        for name, g, w, tol in zip(("x_D", "x_P", "y_K", "y_D", "y_L"), got, (xd, xp, yk, yd, yl), (TOL_X, TOL_Y, TOL_Y, TOL_Y, TOL_Y)):
            if abs(g - w) > tol:
                die(f"fit canary r={r} {name}: {g:.8f} != recorded {w:.8f}")
    return fit


def noise(L, fit):
    """73.7: controls only. Also reproduces the registered tau_K / tau_D as a canary on the control data."""
    ctl = {s: L.chain(CTL, CTL, s, gens=GENS) for s in CTL_SEEDS}
    N = {}
    for k in "KDL":
        vals = [abs(mean(ctl[s][g][k] for s in tri) - fit[0][k]) for g in GENS for tri in itertools.combinations(CTL_SEEDS, 3)]
        if len(vals) != 28:
            die(f"noise scale built from {len(vals)} values, expected 28")
        N[k] = max(vals)
        if abs(N[k] - N_CANARY[k]) > 2e-5:
            die(f"noise canary N_{k}: {N[k]:.6f} != recorded {N_CANARY[k]:.6f}")
    for k, w in TAU_CANARY.items():
        tau = max(abs(ctl[s][g][k]) for s in CTL_SEEDS for g in GENS)
        if abs(tau - w) > 2e-5:
            die(f"tau canary {k}: max |control shift| {tau:.6f} != registered {w:.6f}")
    return N


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True)


def pins():
    for f in ("scripts/fc75_analysis.py", "src/detectors.py", "src/generate.py", "fc75_conditions.json"):
        if git("ls-files", "--error-unmatch", f).returncode or git("diff", "--quiet", "HEAD", "--", f).returncode:
            die(f"{f} is not committed and unmodified")
    if "origin/" not in git("branch", "-r", "--contains", "HEAD").stdout:
        die("HEAD is not on origin: push before the read (sec. 73)")


def gate(name, keys=None):
    p = f"logs/fc75/gate_{name}.json"
    if not os.path.isfile(p):
        die(f"{p} missing: run scripts/fc75_gate.py --{name}")
    g = json.load(open(p))
    if g.get("verdict") != "PASS" or g.get("checked", 0) <= 0 or (keys and g.get("expect_keys") != keys):
        die(f"{p} is not a PASS" + (f" at {keys} keys" if keys else ""))


def registered(cond):
    got = [(c["arm"], c["type"], c["r"], c["corpus_temperature"]) for c in cond["chains"]]
    if got != CHAINS:
        die("fc75_conditions.json chains differ from the registered list in this script")
    for c in cond["C"]:
        if C_ALLOWED.get(c["arm"]) != c["r"] or c["completions_arm"] != c["arm"] + "_topkeval":
            die(f"fc75_conditions.json: unexpected C condition {c}")
    return cond["C"]


# ------------------------------------------------------------------ printing

def f4(x):
    return "    --" if x is None else f"{x:+.4f}"


def print_fit(fit, N):
    print("\nFIT SET (already read; exploratory). Condition means over seeds 43-45.")
    print(f"  {'r':>4} {'arm':15} {'Delta(1)':>9} {'Delta(2)':>9} {'P-inc(1)':>9} {'y_K':>8} {'y_D':>8} {'y_L':>8}")
    for r, arm in FIT:
        c = fit[r]
        print(f"  {r:>4} {arm:15} {c['xD1']:>9.4f} {c['xD2']:>9.4f} {c['xP1']:>9.4f} {f4(c['K']):>8} {f4(c['D']):>8} {f4(c['L']):>8}")
    print(f"  fitted Delta(1) range [{min(fit[r]['xD1'] for r in fit):.4f}, {max(fit[r]['xD1'] for r in fit):.4f}]")
    print(f"NOISE SCALE (controls only, 28 three-seed means): N_K {N['K']:.4f}  N_D {N['D']:.4f}  N_L {N['L']:.4f}")


def print_eval(title, ev, N, show_rows=True):
    print(f"\n{title}   (N = {N:.4f})")
    if show_rows:
        print(f"  {'condition':16} {'type':4} {'r':>4} {'T':>5} {'x_D':>7} {'range':5} {'observed':>9} {'cand':>8} {'e_D':>7}"
              f" {'r-only':>8} {'e_r':>7} {'P-inc':>8} {'e_P':>7}  note")
        for r in ev["rows"]:
            note = "ALREADY READ" if r["already_read"] else "probe, reported" if r["type"] == "P" else \
                   "" if r["in_range"] else "out of range, reported"
            pr, pP = r["pr_" + BASE_FORM["r"]], r["pP_" + BASE_FORM["P"]]
            print(f"  {r['name']:16} {r['type']:4} {r['r']:>4} {r['T']:>5} {r['xD']:>7.4f} {'in' if r['in_range'] else 'OUT':5}"
                  f" {f4(r['y']):>9} {f4(r['pD']):>8} {r['eD']:>7.4f} {f4(pr):>8} {abs(pr - r['y']):>7.4f}"
                  f" {f4(pP):>8} {abs(pP - r['y']):>7.4f}  {note}")
        print(f"  (baselines in their registered forms: r-only {BASE_FORM['r']}, P-inc {BASE_FORM['P']})")
    print(f"  (i)  ACCURATE: {ev['accurate']}  over {ev['n_counted']} in-range held-out conditions")
    for b, lab in (("r", "r-only"), ("P", "P-inc")):
        v = ev["versus"][b]
        gap = "" if v["gap"] is None else f", mean(e_b - e_D) {v['gap']:+.4f}"
        print(f"  (ii) vs {lab:6}: {v['verdict']:11} (contested {v['contested']}{gap}; sensitivity, other form: {v['other_form']})")
    low = [x for x in ev["robust"] if x["r"] <= 50]
    hi = [x for x in ev["robust"] if x["r"] > 50]
    print(f"  S-ROBUST (r <= 50): {ev['s_robust']}   " + "  ".join(
        f"{x['name']} {x['move']:.4f}{'*' if x['already_read'] else ''}" for x in low))
    if hi:
        print("  at r >= 75 (descriptive): " + "  ".join(f"{x['name']} {x['move']:.4f}{'*' if x['already_read'] else ''}" for x in hi))
    if any(x["already_read"] for x in ev["robust"]):
        print("  * already read: not a held-out value")


def selftest():
    n = 0
    def ok(c, m):
        nonlocal n; n += 1
        if not c:
            print(f"SELFTEST FAIL: {m}"); sys.exit(1)
    pts = [(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)]
    ok(interp(pts, 0.5) == (0.5, True), "interp interior")
    ok(interp(pts, 1.0)[0] == 1.0, "interp node")
    ok(interp(pts, 3.0) == (7.0, False), "interp continuation above")
    ok(interp(pts, -1.0) == (-1.0, False), "interp continuation below")
    ok(interp(list(reversed(pts)), 1.5)[0] == 2.5, "interp sorts")
    ok(abs(line([(0, 1), (1, 3), (2, 5)], 3) - 7) < 1e-12, "line")
    ok(abs(spearman([1, 2, 3, 4], [2, 4, 6, 9]) - 1) < 1e-12 and spearman([1, 2], [1, 2]) is None, "spearman")
    fit = {r: dict(xD1=x, xD2=x, xP1=p, xP2=p, K=y, D=y, L=y) for r, x, p, y in
           [(0, 0.0, 0.5, 0.0), (25, 1.0, 0.1, 0.1), (50, 2.0, 0.2, 0.2), (75, 3.0, 0.3, 0.3), (100, 4.0, 0.6, 0.4)]}
    def held(spec):
        return [dict(name=nm, type=t, r=r, T="t", xD1=x, xD2=x, xP1=p, xP2=p, K=y, D=y, L=y) for nm, t, r, x, p, y in spec]
    N = 0.05
    # damage follows Delta: the candidate must PASS
    h = held([("a", "A", 10, 0.4, 0.3, 0.04), ("b1", "B", 25, 2.5, 0.12, 0.25), ("b2", "B", 50, 3.5, 0.22, 0.35), ("c", "C", 25, 3.0, 0.15, 0.30)])
    e = evaluate(fit, h, N, "K", 1)
    ok(e["verdict"] == "PASS" and e["accurate"] and e["versus"]["r"]["verdict"] == "WIN", f"PASS branch: {e['verdict']} {e['versus']}")
    ok(e["s_robust"] == "DOES NOT HOLD", "S-ROBUST must not hold when damage follows Delta")
    # damage follows r: the candidate must FAIL and S-ROBUST must hold
    h = held([("a", "A", 10, 0.4, 0.3, 0.04), ("b1", "B", 25, 2.5, 0.12, 0.10), ("b2", "B", 50, 3.5, 0.22, 0.20), ("c", "C", 25, 3.0, 0.15, 0.10)])
    e = evaluate(fit, h, N, "K", 1)
    ok(e["verdict"] == "FAIL" and not e["accurate"] and e["versus"]["r"]["verdict"] == "LOSS", f"FAIL branch: {e['verdict']} {e['versus']}")
    ok(e["s_robust"] == "HOLDS", "S-ROBUST must hold when damage follows r")
    # temperature moved nothing: predictions never separate
    h = held([("a", "A", 10, 0.4, 0.3, 0.04), ("b1", "B", 25, 1.0, 0.10, 0.10), ("b2", "B", 50, 2.0, 0.20, 0.20)])
    e = evaluate(fit, h, N, "K", 1)
    ok(e["verdict"] == "NOT EVALUABLE" and e["versus"]["r"]["verdict"] == "UNCONTESTED", f"NOT EVALUABLE branch: {e['verdict']}")
    # out-of-range and probe conditions never count
    h = held([("a", "A", 10, 0.4, 0.3, 0.04), ("b1", "B", 25, 9.0, 0.12, 5.0), ("p", "P", 100, 2.0, 0.2, 9.0)])
    e = evaluate(fit, h, N, "K", 1)
    ok(e["n_counted"] == 1 and e["accurate"], "out-of-range / probe excluded from (i)")
    ok(not [r for r in e["rows"] if r["ranked"]], "out-of-range / probe excluded from (ii)")
    # already-read C values are excluded when skip_c is set
    h = held([("a", "A", 10, 0.4, 0.3, 0.04), ("c", "C", 25, 3.0, 0.15, 9.0)])
    ok(evaluate(fit, h, N, "L", 1, skip_c=True)["accurate"] and not evaluate(fit, h, N, "L", 1)["accurate"], "skip_c")
    # mixed signs on the contested set are a TIE, and a TIE fails
    h = held([("b1", "B", 25, 2.5, 0.12, 0.25), ("b2", "B", 50, 3.5, 0.22, 0.20)])
    e = evaluate(fit, h, N, "K", 1)
    ok(e["versus"]["r"]["verdict"] == "TIE" and e["verdict"] == "FAIL", f"TIE branch: {e['versus']['r']} {e['verdict']}")
    ok(evaluate(fit, [], N, "K", 1)["verdict"] == "NOT EVALUABLE", "n = 0 is never a pass")
    print(f"SELFTEST PASS ({n} checks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("SCRATCH", ""), "collapse"))
    for f in ("selftest", "fit", "check", "rehearse"):
        ap.add_argument("--" + f, action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not os.path.isdir(a.root):
        die(f"collapse root {a.root!r} not found")
    if a.rehearse and "mock" not in os.path.abspath(a.root):
        die("--rehearse refuses a root without 'mock' in its path")
    cond = json.load(open("fc75_conditions.json")) if os.path.isfile("fc75_conditions.json") else None
    if cond is None and not a.fit:
        die("fc75_conditions.json missing: run prepare_7p5.py")
    fit_arms = [arm for _, arm in FIT]
    c_arms = registered(cond) if cond else [dict(arm=k, r=v, completions_arm=k + "_topkeval") for k, v in C_ALLOWED.items()]

    if a.fit:
        L = Loader(a.root, fit_arms + [c["arm"] for c in c_arms])
        fit = build_fit(L); N = noise(L, fit)
        print("7.5 --fit: already-read arms only. Canaries: fit means 25/25, noise 3/3, registered tau 2/2.")
        print_fit(fit, N)
        print("\nPREDICTIONS ON RECORD for C (inputs already read; the top-k damage does not exist yet).")
        print(f"  {'arm':14} {'r':>4} {'Delta(1)':>9} {'range':5}  " + "  ".join(f"cand {k:1} / r-only" for k, _ in TARGETS))
        for c in c_arms:
            ch = [L.R.metrics(c["arm"], s, ["perplexity", "entropy"]) for s in SEEDS]
            xd = mean(L.D.dcal_series(m)[1] for m in ch)
            cells = []
            for k, _ in TARGETS:
                pd, inr = interp([(fit[r]["xD1"], fit[r][k]) for r in fit], xd)
                cells.append(f"{pd:+.4f} / {fit[c['r']][k]:+.4f}")
            print(f"  {c['arm']:14} {c['r']:>4} {xd:>9.4f} {'in' if inr else 'OUT':5}  " + "   ".join(cells))
        print("\nALREADY READ, y_L at C (perplexity is decoder-independent): observed against both maps.")
        for c in c_arms:
            ch = [L.R.metrics(c["arm"], s, ["perplexity", "entropy"]) for s in SEEDS]
            xd, yl = mean(L.D.dcal_series(m)[1] for m in ch), mean(L.D.pinc_series(m)[7] for m in ch)
            pd = interp([(fit[r]["xD1"], fit[r]["L"]) for r in fit], xd)[0]
            print(f"  {c['arm']:14} observed {yl:+.4f}   candidate {pd:+.4f} (error {abs(pd - yl):.4f})   r-only {fit[c['r']]['L']:+.4f} (error {abs(fit[c['r']]['L'] - yl):.4f})")
        print("\nRESULT OK (fit only; no held-out arm was opened)")
        return

    if not a.rehearse:
        pins(); gate("g0"); gate("all", 18)
        if c_arms:
            gate("topkeval")
    held_spec = [(n, t, r, "1.0" if T is None else f"{T:.2f}", n, n) for n, t, r, T in CHAINS] + \
                [(c["arm"], "C", c["r"], "->0", c["arm"], c["completions_arm"]) for c in c_arms]
    L = Loader(a.root, fit_arms + [w for *_, w, _ in held_spec] + [c for *_, c in held_spec])
    fit = build_fit(L); N = noise(L, fit)
    held = [condition(L, n, t, r, T, w, c) for n, t, r, T, w, c in held_spec]
    order = {"A": 0, "B": 1, "C": 2, "P": 3}
    held.sort(key=lambda c: (order[c["type"]], c["r"], c["T"]))
    print(f"canaries: fit means 25/25, noise 3/3, registered tau 2/2, gen-0 identity {3 * (len(held) + len(FIT))} chains, "
          f"held-out conditions {len(held)} ({sum(c['type'] == 'C' for c in held)} of type C)")
    if a.check:
        print("--check: inputs valid. No held-out value printed.")
        return
    print_fit(fit, N)
    out = {"N": N, "fit": {r: {k: fit[r][k] for k in ("xD1", "xD2", "xP1", "xP2", "K", "D", "L")} for r in fit}, "evals": {}}

    print("\n" + "=" * 118)
    ev = evaluate(fit, held, N["K"], "K", 1)
    out["evals"]["K1"] = ev
    print_eval("73.8 PRIMARY: KL-unigram rise at gen 7, candidate Delta(1)", ev, N["K"])
    print(f"  VERDICT: {ev['verdict']}")
    print("=" * 118)

    print("\n73.10(a) secondary measures, candidate Delta(1). Never override 73.8.")
    for k, lab in TARGETS[1:]:
        ev2 = evaluate(fit, held, N[k], k, 1, skip_c=(k == "L"))
        out["evals"][k + "1"] = ev2
        print_eval(f"{lab}", ev2, N[k])
        print(f"  components read: {ev2['verdict']}")

    print("\n73.10(b) second candidate, Delta(2) (inputs at gen 2). Never overrides 73.8.")
    for k, lab in TARGETS:
        ev2 = evaluate(fit, held, N[k], k, 2, skip_c=(k == "L"))
        out["evals"][k + "2"] = ev2
        print_eval(f"{lab}", ev2, N[k], show_rows=(k == "K"))
        print(f"  components read: {ev2['verdict']}")

    print("\n73.10(c) probe P (T = 1.25), reported only. Expectation: Delta(1) below the T = 1 value at r = 100; candidate UNDER-predicts.")
    for r in out["evals"]["K1"]["rows"]:
        if r["type"] == "P":
            print(f"  Delta(1) {r['xD']:.4f} vs {fit[100]['xD1']:.4f} at T = 1: {'below (as expected)' if r['xD'] < fit[100]['xD1'] else 'NOT below'}")
            print(f"  KL rise observed {r['y']:+.4f}, candidate {r['pD']:+.4f}: {'under-predicts (as expected)' if r['pD'] < r['y'] else 'does NOT under-predict'}"
                  f"; r-only {r['pr_' + BASE_FORM['r']]:+.4f}")

    print("\n73.10(d) Spearman, predicted vs observed, in-range held-out conditions (descriptive, not a tested outcome)")
    for key, lab in (("K1", "KL, Delta(1)"), ("K2", "KL, Delta(2)")):
        rs = [r for r in out["evals"][key]["rows"] if r["counted"]]
        y = [r["y"] for r in rs]
        print(f"  {lab}: n {len(rs)}  candidate {f4(spearman([r['pD'] for r in rs], y))}  r-only {f4(spearman([r['pr_' + BASE_FORM['r']] for r in rs], y))}"
              f"  P-inc {f4(spearman([r['pP_' + BASE_FORM['P']] for r in rs], y))}")

    print("\nper-seed held-out values (x_D(1), y_K, y_D, y_L), seeds 43 / 44 / 45")
    for c in held:
        print(f"  {c['name']:16} " + "   ".join(f"{x['xD1']:.4f} {x[7]['K']:+.4f} {x[7]['D']:+.4f} {x[7]['L']:+.4f}" for x in c["per_seed"]))

    out["meta"] = {"head": git("rev-parse", "HEAD").stdout.strip(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "script_md5": md5(os.path.abspath(__file__)), "rehearse": a.rehearse, "verdict": out["evals"]["K1"]["verdict"]}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nRESULT OK  head {out['meta']['head'][:7]}  script {out['meta']['script_md5']}  primary verdict {out['meta']['verdict']}")


if __name__ == "__main__":
    main()
