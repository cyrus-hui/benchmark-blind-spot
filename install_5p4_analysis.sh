# install_5p4_analysis.sh - Phase 5.4 analysis tooling (Sept 17 (c)).
# Paste the whole file into a Narval login shell, or: bash install_5p4_analysis.sh
# Runs in a subshell, refuses to overwrite, md5-verifies every payload IN THE DESTINATION,
# runs the self-test, and reads no file under $SCRATCH. It creates registered_thresholds.json
# with the values already recorded in decisions.md and null for everything still pending.
(
cd ~/modelcollapse || { echo "REFUSE: ~/modelcollapse not found"; exit 1; }
for f in src/detectors.py thresholds_registry.py compute_pplslope_thresholds.py detector_eval_5p4.py scripts/read_greedy_canary.py registered_thresholds.json; do
  if [ -e "$f" ]; then echo "REFUSE: $f exists"; exit 1; fi
done
mkdir -p src scripts
cat > src/detectors.py <<'PAYLOAD_0_EOF'
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
PAYLOAD_0_EOF
cat > thresholds_registry.py <<'PAYLOAD_1_EOF'
#!/usr/bin/env python
"""Registered-threshold file for the Phase 5.4 read (registered_thresholds.json).

The file holds the ROUNDED values exactly as recorded in decisions.md. The analysis
recomputes every threshold at full precision from control arms and asserts that it
rounds to the registered value, so this file is a pre-registration, not an input.

  python thresholds_registry.py init                       # refuses if the file exists
  python thresholds_registry.py set sigma.greedy.sub 1.234 # refuses to overwrite a non-null value
  python thresholds_registry.py show                       # prints values, pending keys, and md5
"""
import hashlib, json, pathlib, sys

PATH = pathlib.Path(__file__).resolve().parent / "registered_thresholds.json"
INITIAL = {
    "dcal":    {"tau": 0.0635},
    "pinc":    {"tau": None},
    "pslope":  {"tau_g2": None, "tau_g3": None, "tau_g4": None, "tau_g5": None, "tau_g6": None, "tau_g7": None},
    "sigma":   {"topk": {"sub": 7.248, "full": 6.148}, "greedy": {"sub": None, "full": None}},
    "outcome": {"topk": {"tau_D": 0.0105, "tau_K": 0.1111}, "greedy": {"tau_D": None, "tau_K": None}},
}
DECIMALS = {"dcal": 4, "pinc": 4, "pslope": 5, "sigma": 3, "outcome": 4}


def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, pre + k + ".")
        else:
            yield pre + k, v


def load():
    if not PATH.is_file():
        sys.exit(f"STOP: {PATH} does not exist; run init")
    return json.load(open(PATH))


def save(d):
    PATH.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")


def md5():
    return hashlib.md5(PATH.read_bytes()).hexdigest()


def pending(d):
    return [k for k, v in flat(d) if v is None]


def setkey(d, dotted, value):
    node, parts = d, dotted.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            sys.exit(f"REFUSE: no such key {dotted}")
        node = node[p]
    leaf = parts[-1]
    if leaf not in node or isinstance(node[leaf], dict):
        sys.exit(f"REFUSE: no such key {dotted}")
    if node[leaf] is not None:
        sys.exit(f"REFUSE: {dotted} is already registered as {node[leaf]}; a registered value is never overwritten")
    node[leaf] = round(float(value), DECIMALS[parts[0]])


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd = sys.argv[1]
    if cmd == "init":
        if PATH.exists():
            sys.exit(f"REFUSE: {PATH} exists")
        save(INITIAL)
    elif cmd == "set":
        if len(sys.argv) != 4:
            sys.exit("usage: set <dotted.key> <value>")
        d = load()
        setkey(d, sys.argv[2], sys.argv[3])
        save(d)
    elif cmd != "show":
        sys.exit(__doc__)
    d = load()
    for k, v in flat(d):
        print(f"  {k:22s} {v}")
    p = pending(d)
    print(f"pending: {len(p)}  {p}")
    print(f"md5: {md5()}")


if __name__ == "__main__":
    main()
PAYLOAD_1_EOF
cat > compute_pplslope_thresholds.py <<'PAYLOAD_2_EOF'
#!/usr/bin/env python
"""Perplexity baseline thresholds, from CONTROL arms only.

Specification pre-registered in decisions.md, Sept 17 (c):
  P-inc   s(g) = log PPL(g) - log PPL(0), in-domain;  tau = max over control chains, gens 1-7
  P-slope b(g) = OLS slope of log PPL on g over gens 1..g (g >= 2);
          tau(g) = max over control chains of b(g), one threshold per window
Both must be recorded in decisions.md BEFORE any 5.3 value is read. This script refuses
any arm that is not a control arm.

  python compute_pplslope_thresholds.py                 # control_gpt2m, seeds 43-46, canary on
  python compute_pplslope_thresholds.py --register      # also writes pinc/pslope into the registry
  python compute_pplslope_thresholds.py --arm control_smollm2 --seeds 43,44,45,46,47   # transfer report
"""
import argparse, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
import detectors as D  # noqa: E402

# results.md, Phase 5.2: per-seed max over gens 1-7 of the log-PPL term, and max Delta
CANARY = {
    "control_gpt2m":   {"pinc": {43: 0.0731, 44: 0.0741, 45: 0.0731, 46: 0.0736}, "dcal_max": 0.0635},
    "control_smollm2": {"pinc": {43: 0.0338, 44: 0.0344, 45: 0.0343, 46: 0.0346, 47: 0.0345}, "dcal_max": 0.0307},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--arm", default="control_gpt2m")
    ap.add_argument("--seeds", default="43,44,45,46")
    ap.add_argument("--register", action="store_true")
    a = ap.parse_args()
    if not D.is_control(a.arm):
        sys.exit("REFUSED: this script reads control arms only")
    seeds = [int(s) for s in a.seeds.split(",")]
    rd = D.Reader(a.root)                       # allow_exposed stays False
    m = {s: rd.metrics(a.arm, s, ["perplexity", "entropy"]) for s in seeds}
    pinc = {s: D.pinc_series(m[s]) for s in seeds}
    psl = {s: D.pslope_series(m[s]) for s in seeds}
    dcal = {s: D.dcal_series(m[s]) for s in seeds}

    fails = 0
    ref = CANARY.get(a.arm)
    if ref is None:
        sys.exit(f"STOP: no canary recorded for {a.arm}")
    for s in seeds:
        if s not in ref["pinc"]:
            sys.exit(f"STOP: no canary recorded for seed {s}")
        got = max(pinc[s][1:])
        ok = abs(got - ref["pinc"][s]) < 5e-5
        fails += not ok
        print(f"canary s{s} max log-PPL increment {got:.6f} vs recorded {ref['pinc'][s]:.4f}  {'OK' if ok else 'FAIL'}")
    got = max(max(dcal[s][1:]) for s in seeds)
    ok = abs(got - ref["dcal_max"]) < 5e-5
    fails += not ok
    print(f"canary max Delta_cal increment      {got:.6f} vs recorded {ref['dcal_max']:.4f}  {'OK' if ok else 'FAIL'}")
    if fails:
        sys.exit("CANARY FAIL - stop; do not record thresholds")

    print(f"\narm={a.arm} seeds={seeds}")
    print("\nP-inc  s(g) = log PPL(g) - log PPL(0), nats")
    for s in seeds:
        print(f"  s{s}: " + " ".join(f"{x:+.4f}" for x in pinc[s][1:]))
    t_inc = D.tau_single(list(pinc.values()))
    print(f"  tau_P-inc = {t_inc:.4f}")
    print("\nP-slope  b(g) = OLS slope of log PPL over gens 1..g, nats/gen, g = 2..7")
    for s in seeds:
        print(f"  s{s}: " + " ".join(f"{x:+.5f}" for x in psl[s][2:]))
    t_sl = D.tau_window(list(psl.values()))
    print("  tau_P-slope(g): " + " ".join(f"g{g}={t_sl[g]:+.5f}" for g in range(2, 8)))
    print("\nRECORD IN decisions.md: tau_P-inc = %.4f ; tau_P-slope(g2..g7) = %s"
          % (t_inc, " ".join(f"{t_sl[g]:+.5f}" for g in range(2, 8))))

    if a.register:
        if a.arm != "control_gpt2m" or seeds != [43, 44, 45, 46]:
            sys.exit("REFUSE: only control_gpt2m seeds 43-46 is registered")
        import thresholds_registry as R
        d = R.load()
        for key, val in [("pinc.tau", t_inc)] + [(f"pslope.tau_g{g}", t_sl[g]) for g in range(2, 8)]:
            R.setkey(d, key, val)               # exits before anything is saved if any key is non-null
        R.save(d)
        for k, v in R.flat(d):
            print(f"  {k:22s} {v}")
        print(f"pending: {R.pending(d)}")
        print(f"md5: {R.md5()}")

if __name__ == "__main__":
    main()
PAYLOAD_2_EOF
cat > detector_eval_5p4.py <<'PAYLOAD_3_EOF'
#!/usr/bin/env python
"""Phase 5.4 detector evaluation: Delta_cal vs perplexity baseline vs SIGMA-UB.

Every rule here is pre-registered in decisions.md Sept 15 (b), Sept 17 and Sept 17 (c).
Unit of analysis: the chain. Per decoder, 12 exposed chains (r in 25/50/75/100 x seeds 43-45)
against 3 control chains (r = 0, seeds 43-45). Two labels x two decoders = four readings.

  python detector_eval_5p4.py --self-test
  python detector_eval_5p4.py                       # controls only: verifies every registered tau
  python detector_eval_5p4.py --transfer            # 1.7B arms (already read) at the 345M thresholds
  python detector_eval_5p4.py --authorise-grid-read <md5 of registered_thresholds.json>

The grid read is refused unless the registry has no pending value and its md5 matches the
one given on the command line (the md5 recorded in decisions.md).
"""
import argparse, os, pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE))
import detectors as D            # noqa: E402
import thresholds_registry as R  # noqa: E402

WKEYS, CKEYS = ["perplexity", "entropy"], ["distinct2", "kl_unigram"]
fails = 0


def check(name, ok, detail=""):
    global fails
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ------------------------------------------------------------------ thresholds from controls

def control_thresholds(rd, decoders):
    """Full-precision thresholds, recomputed from control arms only."""
    arm, seeds = D.WEIGHTS_TAU
    wm = [rd.metrics(arm, s, WKEYS) for s in seeds]
    tau = {"dcal": D.tau_single([D.dcal_series(m) for m in wm]),
           "pinc": D.tau_single([D.pinc_series(m) for m in wm]),
           "pslope": D.tau_window([D.pslope_series(m) for m in wm]),
           "sigma": {}, "outcome": {}}
    for dec in decoders:
        arm, seeds = D.COMPLETIONS_TAU[dec]
        recs = [rd.sigma(arm, s) for s in seeds]
        tau["sigma"][dec] = {"sub": D.tau_single([D.sigma_series(r, "dU_sub") for r in recs]),
                             "full": D.tau_single([D.sigma_series(r, "dU_full") for r in recs])}
        cm = [rd.metrics(arm, s, CKEYS) for s in seeds]
        tau["outcome"][dec] = {
            "tau_D": max(abs(m[g]["distinct2"] - m[0]["distinct2"]) for m in cm for g in range(1, 8)),
            "tau_K": max(abs(m[g]["kl_unigram"] - m[0]["kl_unigram"]) for m in cm for g in range(1, 8))}
    return tau


def verify_against_registry(tau, reg, decoders):
    def one(label, exact, registered, nd):
        if registered is None:
            print(f"PEND  {label}: recomputed {exact:+.{nd + 2}f}; nothing registered yet")
            return
        check(f"tau {label}", round(exact, nd) == registered, f"recomputed {exact:+.{nd + 2}f} vs registered {registered}")
    one("dcal", tau["dcal"], reg["dcal"]["tau"], 4)
    one("pinc", tau["pinc"], reg["pinc"]["tau"], 4)
    for g in range(2, 8):
        one(f"pslope g{g}", tau["pslope"][g], reg["pslope"][f"tau_g{g}"], 5)
    for dec in decoders:
        for k in ("sub", "full"):
            one(f"sigma {dec} {k}", tau["sigma"][dec][k], reg["sigma"][dec][k], 3)
        for k in ("tau_D", "tau_K"):
            one(f"outcome {dec} {k}", tau["outcome"][dec][k], reg["outcome"][dec][k], 4)


# ------------------------------------------------------------------ one decoder

def build(rd, dec, tau, ratios):
    """Returns detectors {name: {"score": {cid: s7}, "lat": {cid: L}}}, collapsed {cid: bool}."""
    dets = {n: {"score": {}, "lat": {}} for n in ("dcal", "pinc", "pslope", "sigma_sub", "sigma_full")}
    collapsed = {}
    for r in ratios:
        for s in D.EVAL_SEEDS:
            cid = (r, s)
            wm = rd.metrics(D.WEIGHTS_ARM[dec][r], s, WKEYS)
            cm = rd.metrics(D.COMPLETIONS_ARM[dec][r], s, CKEYS)
            rec = rd.sigma(D.COMPLETIONS_ARM[dec][r], s)
            series = {"dcal": (D.dcal_series(wm), tau["dcal"]),
                      "pinc": (D.pinc_series(wm), tau["pinc"]),
                      "pslope": (D.pslope_series(wm), tau["pslope"]),
                      "sigma_sub": (D.sigma_series(rec, "dU_sub"), tau["sigma"][dec]["sub"]),
                      "sigma_full": (D.sigma_series(rec, "dU_full"), tau["sigma"][dec]["full"])}
            for n, (ser, t) in series.items():
                dets[n]["score"][cid] = ser[7]
                dets[n]["lat"][cid] = D.first_flag(ser, t)
            o = tau["outcome"][dec]
            collapsed[cid] = (cm[7]["distinct2"] - cm[0]["distinct2"] < -o["tau_D"]
                              and cm[7]["kl_unigram"] - cm[0]["kl_unigram"] > o["tau_K"])
    return dets, collapsed


def fmt_stats(n, st, npos):
    return (f"  {n:11s} AUC {st['auc']:.4f}  mean latency {st['mean_lat']:.3f}  "
            f"TPR@tau {st['n_flag']}/{npos}  FPR@TPR=1 {st['fpr_at_tpr1']:.3f} (resolution 1/3)  "
            f"controls flagged at tau {st['neg_flagged']}/3 (0 by construction)")


def reading(dets, pos, neg, title):
    """One label x decoder reading. Returns dict of verdicts, or None when not evaluable."""
    print(f"\n--- {title}: {len(pos)} positive vs {len(neg)} negative chains")
    if len(pos) < 3:
        print("  NOT EVALUABLE: fewer than 3 positive chains; set aside, counts neither way")
        return None
    for n, d in dets.items():
        print(fmt_stats(n, D.stats(d, pos, neg), len(pos)))
    sig = D.better_of({"sigma_sub": dets["sigma_sub"], "sigma_full": dets["sigma_full"]}, "sigma_sub", pos, neg)
    ppl = D.better_of({"pinc": dets["pinc"], "pslope": dets["pslope"]}, "pinc", pos, neg)
    out = {"separates": D.stats(dets["dcal"], pos, neg)["tpr"] == 1.0}
    print(f"  Delta_cal separates at the registered tau: {'YES' if out['separates'] else 'NO'}")
    for fam, best in (("SIGMA", sig), ("PPL", ppl)):
        v, lvl, note = D.compare(dets["dcal"], dets[best], pos, neg)
        out[fam] = v
        print(f"  Delta_cal vs {fam} (ranked on {best}): {v}  [level: {lvl}]  {note}")
    return out


def exit_verdict(readings):
    ev = {k: v for k, v in readings.items() if v is not None}
    print("\n=== Phase 5 exit criterion, over the evaluable readings:", ", ".join(ev) or "none")
    if not any(k.startswith("design") for k in ev):
        print("STOP: no design-label reading is evaluable"); return
    sep = all(v["separates"] for v in ev.values())
    print(f"separates in every reading: {sep}")
    beats = {}
    for fam in ("SIGMA", "PPL"):
        vs = [v[fam] for v in ev.values()]
        beats[fam] = all(x in ("WIN", "CEILING") for x in vs) and "WIN" in vs
        print(f"beats {fam}: {beats[fam]}   readings: {vs}")
    print("EXIT", "(i) separates and beats both baselines" if sep and all(beats.values())
          else "(ii) criterion not met; weaker verdict is the outcome")
    print("Every FPR above is 0 at the registered tau BY CONSTRUCTION (tau is the control maximum).")
    print("The only out-of-sample false-positive evidence is cross-scale: run --transfer.")


def grid(rd, reg):
    tau = control_thresholds(rd, D.DECODERS)
    verify_against_registry(tau, reg, D.DECODERS)
    if fails:
        sys.exit("STOP: a recomputed threshold does not round to its registered value")
    readings, pooled = {}, {"score": {}, "lat": {}}
    for dec in D.DECODERS:
        dets, collapsed = build(rd, dec, tau, D.RATIOS)
        ids = list(dets["dcal"]["score"])
        neg = [c for c in ids if c[0] == 0]
        exposed = [c for c in ids if c[0] > 0]
        check(f"{dec}: no control chain is outcome-collapsed (by construction)", not any(collapsed[c] for c in neg))
        print(f"\n##### decoder = {dec}")
        print("outcome label, collapsed chains by r: " + "  ".join(
            f"r={r}: {sum(collapsed[(r, s)] for s in D.EVAL_SEEDS)}/3" for r in D.RATIOS if r))
        for n in dets:
            print(f"latency {n:11s} " + "  ".join(
                f"r={r}:" + ",".join(str(dets[n]['lat'][(r, s)]) for s in D.EVAL_SEEDS) for r in D.RATIOS))
        readings[f"design/{dec}"] = reading(dets, exposed, neg, f"design label, {dec}")
        pos_o = [c for c in exposed if collapsed[c]]
        aside = [c for c in exposed if not collapsed[c]]
        readings[f"outcome/{dec}"] = reading(dets, pos_o, neg, f"outcome label, {dec}")
        print(f"  exposed-not-collapsed chains (set aside, own row): {len(aside)}")
        for n in dets:
            print(f"    {n:11s} flags {sum(dets[n]['lat'][c] <= 7 for c in aside)}/{len(aside)}")
        for c in ids:
            if c[0] > 0 or dec == "topk":               # r = 0 counted ONCE in the pooled ROC
                pooled["score"][(dec,) + c] = dets["dcal"]["score"][c]
                pooled["lat"][(dec,) + c] = dets["dcal"]["lat"][c]
    pp = [c for c in pooled["score"] if c[1] > 0]
    pn = [c for c in pooled["score"] if c[1] == 0]
    st = D.stats(pooled, pp, pn)
    print(f"\nDelta_cal pooled ROC as pre-registered Sept 15 (b): {len(pp)} exposed vs {len(pn)} controls, "
          f"AUC {st['auc']:.4f}, TPR@tau {st['n_flag']}/{len(pp)}")
    exit_verdict(readings)


def controls_only(rd, reg):
    have = [d for d in D.DECODERS if (rd.root / "sigma" / D.COMPLETIONS_TAU[d][0]).is_dir()
            and (rd.root / D.COMPLETIONS_TAU[d][0] / "seed43" / "gen7" / "metrics.json").is_file()]
    print("decoders with control completions and SIGMA records on disk:", have)
    tau = control_thresholds(rd, have)
    verify_against_registry(tau, reg, have)
    print("arms opened:", sorted(rd.opened))
    check("only control arms were opened", all(D.is_control(a) for a in rd.opened))


def transfer(root, reg):
    """345M thresholds, unchanged, on the 1.7B arms. All four arms below were read on Sept 12-15."""
    arms = {"control_smollm2": [43, 44, 45, 46, 47], "smollm2_1.7b": [43, 44, 45, 46, 47], "smollm2_greedy": [43, 44, 45]}
    rd = D.Reader(root, allow_exposed=frozenset(arms))
    t345 = control_thresholds(D.Reader(root), [])
    verify_against_registry(t345, reg, [])
    print("\nfirst flagged generation at the 345M threshold (8 = not flagged by gen 7)")
    for arm, seeds in arms.items():
        for n, fn in (("dcal", D.dcal_series), ("pinc", D.pinc_series), ("pslope", D.pslope_series)):
            lat = [D.first_flag(fn(rd.metrics(arm, s, WKEYS)), t345[n]) for s in seeds]
            print(f"  {arm:16s} {n:7s} " + " ".join(str(x) for x in lat))
    print("SIGMA is not evaluated at 1.7B (n_k = 2000 there; the registered index set is 500 of 1000).")


# ------------------------------------------------------------------ self-test

def self_test():
    import json, tempfile
    # pure functions
    check("auc perfect", D.auc([3, 4], [1, 2]) == 1.0)
    check("auc ties 0.5", D.auc([1], [1]) == 0.5)
    check("auc one reversal of 36", abs(D.auc([5] * 11 + [0.5], [1, 1, 1]) - 33 / 36) < 1e-12)
    check("ols slope", abs(D.ols_slope([1, 2, 3], [2.0, 4.0, 6.0]) - 2.0) < 1e-12)
    check("first_flag single", D.first_flag([0, .1, .2, .9, 0, 0, 0, 0], 0.5) == 3)
    check("first_flag censored", D.first_flag([0] * 8, 0.5) == D.CENSOR)
    check("first_flag window skips None", D.first_flag([None, None, 1, 1, 1, 1, 1, 1], {g: 0.5 for g in range(2, 8)}) == 2)
    pos, neg = [(r, s) for r in (25, 50, 75, 100) for s in (43, 44, 45)], [(0, s) for s in (43, 44, 45)]

    def det(score_pos, lat_pos, score_neg=0.0):
        d = {"score": {c: score_neg for c in neg}, "lat": {c: D.CENSOR for c in neg}}
        for c in pos:
            d["score"][c] = score_pos(c); d["lat"][c] = lat_pos(c)
        return d
    a = det(lambda c: 1.0, lambda c: 1)
    b = det(lambda c: 1.0, lambda c: 1)
    check("compare ceiling", D.compare(a, b, pos, neg)[0] == "CEILING")
    b = det(lambda c: 1.0, lambda c: 3 if c[0] == 25 else 1)
    check("compare latency win, stable", D.compare(a, b, pos, neg)[:2] == ("WIN", "mean_lat"))
    check("compare is antisymmetric", D.compare(b, a, pos, neg)[0] == "LOSS")
    b = det(lambda c: 1.0, lambda c: 2 if c == (25, 43) else 1)
    v = D.compare(a, b, pos, neg)
    check("one-chain latency edge is NOT a win", v[0] == "TIE", v[2])
    b = det(lambda c: -1.0 if c == (25, 43) else 1.0, lambda c: 1)
    v = D.compare(a, b, pos, neg)
    check("one-chain AUC edge is NOT a win", v[0] == "TIE", v[2])
    b = det(lambda c: -1.0 if c[0] == 25 else 1.0, lambda c: 1)
    check("three-chain AUC edge is a win", D.compare(a, b, pos, neg)[:2] == ("WIN", "auc"))
    check("better_of default on tie", D.better_of({"x": a, "y": det(lambda c: 1.0, lambda c: 1)}, "x", pos, neg) == "x")
    check("better_of picks the better", D.better_of({"x": b, "y": a}, "x", pos, neg) == "y")

    # fake tree: loader, guard, thresholds, end-to-end verdict
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)

        def write(arm, s, lp, ent, d2, kl, du):
            for g in D.GENS:
                p = root / arm / f"seed{s}" / f"gen{g}"
                p.mkdir(parents=True, exist_ok=True)
                import math
                json.dump({"perplexity": math.exp(3.0 + lp[g]), "entropy": 3.0 + ent[g],
                           "distinct2": 0.5 + d2[g], "kl_unigram": 1.8 + kl[g]}, open(p / "metrics.json", "w"))
            p = root / "sigma" / arm
            p.mkdir(parents=True, exist_ok=True)
            json.dump({"arm": arm, "seed": s, "n_k": 1000,
                       "gens": {str(g): {"dU_sub": du[g], "dU_full": du[g] / 2} for g in D.GENS}},
                      open(p / f"seed{s}.json", "w"))
        z = [0.0] * 8
        step = [0.0] + [0.07] * 7                                  # control: a step, no trend
        for s in (43, 44, 45, 46):
            write("control_gpt2m", s, step, [0.0] + [0.01] * 7, z, z, [0, 1, -1, 1, -1, 1, -1, 1])
        for s in (43, 44, 45):
            write("control_gpt2m_greedyeval", s, step, [0.0] + [0.01] * 7, z, z, [0, 1, -1, 1, -1, 1, -1, 1])
        for dec in D.DECODERS:
            for r in (25, 50, 75, 100):
                f = r / 100
                lp = [0.0] + [0.07 + 0.002 * f * g for g in range(1, 8)]          # PPL: slow trend
                ent = [0.0] + [0.01 - 0.5 * f] * 7                                # entropy falls at gen 1
                d2 = [0.0] + ([-0.05] * 7 if r >= 75 else [0.0] * 7)              # only r >= 75 outcome-collapsed
                kl = [0.0] + ([0.5] * 7 if r >= 75 else [0.0] * 7)
                du = [0, 1, -1, 1, -1, 1, -1, 1]                                  # SIGMA: control-like noise
                for s in D.EVAL_SEEDS:
                    write(D.WEIGHTS_ARM[dec][r], s, lp, ent, d2, kl, du)
        try:
            D.Reader(root).metrics("mix25_topk", 43, WKEYS)
            check("guard refuses an exposed arm", False)
        except SystemExit as e:
            check("guard refuses an exposed arm", "REFUSED" in str(e))
        rd = D.Reader(root, allow_exposed=True)
        tau = control_thresholds(rd, D.DECODERS)
        check("tau dcal = 0.07 - 0.01", abs(tau["dcal"] - 0.06) < 1e-9, f"{tau['dcal']:.6f}")
        check("tau pinc = 0.07", abs(tau["pinc"] - 0.07) < 1e-9)
        check("tau pslope = 0 on a step", all(abs(tau["pslope"][g]) < 1e-12 for g in range(2, 8)))
        check("tau sigma sub = 1", tau["sigma"]["topk"]["sub"] == 1.0)
        dets, col = build(rd, "topk", tau, D.RATIOS)
        ids = list(dets["dcal"]["score"]); ng = [c for c in ids if c[0] == 0]; ex = [c for c in ids if c[0] > 0]
        check("dcal latency 1 on all exposed", all(dets["dcal"]["lat"][c] == 1 for c in ex))
        check("pinc latency 1 on all exposed", all(dets["pinc"]["lat"][c] == 1 for c in ex))
        check("pslope latency 2 on all exposed", all(dets["pslope"]["lat"][c] == 2 for c in ex))
        check("sigma never flags", all(dets["sigma_sub"]["lat"][c] == D.CENSOR for c in ex))
        check("outcome label: 6 of 12 collapsed", sum(col[c] for c in ex) == 6)
        check("dcal vs pinc at ceiling", D.compare(dets["dcal"], dets["pinc"], ex, ng)[0] == "CEILING")
        check("dcal beats sigma on AUC", D.compare(dets["dcal"], dets["sigma_sub"], ex, ng)[:2] == ("WIN", "auc"))
        check("better-of PPL is pinc", D.better_of({"pinc": dets["pinc"], "pslope": dets["pslope"]}, "pinc", ex, ng) == "pinc")
    print("SELF-TEST", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--authorise-grid-read", metavar="MD5")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    reg = R.load()
    print(f"registry md5 {R.md5()}  pending {R.pending(reg)}")
    if a.transfer:
        transfer(a.root, reg)
    elif a.authorise_grid_read:
        if R.pending(reg):
            sys.exit(f"REFUSED: thresholds still pending: {R.pending(reg)}")
        if a.authorise_grid_read != R.md5():
            sys.exit("REFUSED: the md5 given is not the registry's md5; use the one recorded in decisions.md")
        grid(D.Reader(a.root, allow_exposed=True), reg)
    else:
        controls_only(D.Reader(a.root), reg)
    print("RESULT", "OK" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
PAYLOAD_3_EOF
cat > scripts/read_greedy_canary.py <<'PAYLOAD_4_EOF'
#!/usr/bin/env python
"""Gen-0 canary for the greedy outcome-label tau pass.

Tolerance pre-registered in decisions.md, Sept 17 ("Greedy outcome-label tau"):
  perplexity            |diff| <= 1e-6            PASS, else STOP (wrong model)
  distinct2, kl_unigram diff == 0                 PASS
                        0 < |diff| <= 0.002       ACCEPT-AND-RECORD (bf16 batch padding)
                        |diff| > 0.002            STOP
  mauve                 informational
  eval rows             exactly 1000

Shadow  : $SCRATCH/collapse/control_gpt2m_greedyeval/seed{S}/gen0
Against : $SCRATCH/collapse/gpt2m_greedy/seed{S}/gen0   (md5-identical weights)

Read-only. Reads gen 0 only. Refuses any arm containing "mix".

  python scripts/read_greedy_canary.py --seeds 43            # after the pilot 3267692
  python scripts/read_greedy_canary.py --seeds 43,44,45      # after the array; also prints the
                                                             # greedy gen-0 reference means
  python scripts/read_greedy_canary.py --seeds 43 --inverse  # must end GATE STOP on distinct2
"""
import argparse, json, os, pathlib, sys

SHADOW, REF, INVERSE_REF = "control_gpt2m_greedyeval", "gpt2m_greedy", "control_gpt2m"
PPL_TOL, DIST_TOL, N_ROWS = 1e-6, 0.002, 1000


def load(root, arm, seed):
    if "mix" in arm:
        sys.exit("REFUSED: this script never reads a 5.3 grid arm")
    p = root / arm / f"seed{seed}" / "gen0" / "metrics.json"
    if not p.is_file():
        sys.exit(f"STOP: missing {p}")
    d = json.load(open(p))
    for k in ("perplexity", "distinct2", "kl_unigram", "mauve"):
        if k not in d:
            sys.exit(f"STOP: key {k} absent in {p}; keys = {sorted(d)}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--seeds", default="43")
    ap.add_argument("--inverse", action="store_true",
                    help="compare against control_gpt2m gen 0 (top-k completions); must STOP")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    seeds = [int(s) for s in a.seeds.split(",")]
    ref_arm = INVERSE_REF if a.inverse else REF
    print(f"shadow={SHADOW}  against={ref_arm}  seeds={seeds}  gen=0"
          + ("   [INVERSE CANARY: expect GATE STOP]" if a.inverse else ""))
    stop = record = 0
    for s in seeds:
        sh, rf = load(root, SHADOW, s), load(root, ref_arm, s)
        rows_p = root / SHADOW / f"seed{s}" / "gen0" / "eval_completions.jsonl"
        rows = sum(1 for _ in open(rows_p)) if rows_p.is_file() else -1
        v = "PASS" if rows == N_ROWS else "STOP"
        stop += v == "STOP"
        print(f"s{s} eval rows        {rows:>14d}   expect {N_ROWS}                 {v}")
        d = float(sh["perplexity"]) - float(rf["perplexity"])
        v = "PASS" if abs(d) <= PPL_TOL else "STOP"
        stop += v == "STOP"
        print(f"s{s} perplexity  diff {d:+14.3e}   tol {PPL_TOL:.0e}                {v}")
        for k in ("distinct2", "kl_unigram"):
            d = float(sh[k]) - float(rf[k])
            v = "PASS" if d == 0 else ("ACCEPT-AND-RECORD" if abs(d) <= DIST_TOL else "STOP")
            stop += v == "STOP"
            record += v == "ACCEPT-AND-RECORD"
            print(f"s{s} {k:11s} diff {d:+14.6f}   exact | <= {DIST_TOL} | STOP   {v}")
        d = float(sh["mauve"]) - float(rf["mauve"])
        print(f"s{s} mauve       diff {d:+14.6f}   informational")
    if not a.inverse and len(seeds) > 1:
        print("\ngreedy gen-0 reference means (gpt2m_greedy) over seeds", seeds)
        for k in ("distinct2", "kl_unigram", "mauve"):
            m = sum(float(load(root, REF, s)[k]) for s in seeds) / len(seeds)
            print(f"  {k:11s} {m:.6f}")
    print(f"\nrecord-lines={record}")
    print("GATE", "STOP" if stop else "PASS", f"({stop} stop lines)")
    sys.exit(1 if stop else 0)


if __name__ == "__main__":
    main()
PAYLOAD_4_EOF
md5sum -c - <<'MD5_EOF' || { echo "REFUSE: md5 mismatch - delete the five files and re-paste"; exit 1; }
a060b18d3d4035e1711e0cb0647323ca  src/detectors.py
5009726826d5c94efb864893f728ae21  thresholds_registry.py
0d3080de9f711af0d2dbde9432dcbd7f  compute_pplslope_thresholds.py
95187fa95fac2a80e4c7b6a41d694034  detector_eval_5p4.py
b347a0a959d69a0196b1d36946ab3841  scripts/read_greedy_canary.py
MD5_EOF
python -m py_compile src/detectors.py thresholds_registry.py compute_pplslope_thresholds.py detector_eval_5p4.py scripts/read_greedy_canary.py || { echo "REFUSE: syntax"; exit 1; }
python detector_eval_5p4.py --self-test | grep -c '^PASS'
python detector_eval_5p4.py --self-test | grep -E '^(FAIL|SELF-TEST)'
python detector_eval_5p4.py --self-test >/dev/null || { echo "REFUSE: self-test failed"; exit 1; }
python thresholds_registry.py init || exit 1
)
echo "install exit status: $?"
