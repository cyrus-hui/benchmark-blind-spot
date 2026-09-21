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
