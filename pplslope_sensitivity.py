#!/usr/bin/env python3
"""pplslope_sensitivity.py -- robustness of the P-slope baseline (decisions.md Sept 17 (c) Sec 6).

REPORTED, NEVER RANKED.  The ranked perplexity baseline stays exactly what Sec 6
registered: better-of {P-inc, P-slope(registered)}.  This script varies the three
discretionary choices inside P-slope and the input corpus, and reports the WEAKEST
verdict Delta_cal earns across the in-domain variants.

One-directional by construction: a worse verdict here is a stated limitation in the
paper; a better verdict changes nothing.  No variant is ever promoted to the ranked
baseline, before or after the read.

It cannot write registered_thresholds.json (opened 'r' only; md5 asserted unchanged
across every run) so it cannot disturb the md5 that authorises the 5.4 grid read.
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("MC_ROOT", Path.home() / "modelcollapse"))
COLLAPSE = Path(os.environ.get("MC_COLLAPSE", Path(os.environ.get("SCRATCH", "/tmp")) / "collapse"))
REGISTRY = ROOT / "registered_thresholds.json"
SENS = ROOT / "sensitivity_thresholds.json"

CONTROL_ARMS = ("control_gpt2m", "control_gpt2m_greedyeval")
TAU_ARM = "control_gpt2m"          # Sec 6: thresholds from control_gpt2m 43-46 only
TAU_SEEDS = (43, 44, 45, 46)
GENS = tuple(range(0, 8))

# Sec 6 canary constants (already on the record, Sept 14 / Sept 17 (c)).
PINC_MAX_BY_SEED = {43: 0.0731, 44: 0.0741, 45: 0.0731, 46: 0.0736}
PINC_TOL = 1e-4
DCAL_MAX_INC = 0.0635            # tau_Delta, recorded Sept 14; max over all 28 control cells
DCAL_TOL = 1e-4

# "dcal" is not a corpus: it is Delta_cal = log PPL - entropy, in-domain, carried
# through the SAME three axes as P-slope so the baseline's differencing transform
# is applied to the candidate too.  Reported, never ranked -- Sec 3 fixes the
# ranked Delta_cal detector as a level against tau_Delta = 0.0635.
CORPORA = {
    "in": "perplexity",
    "dcal": None,
    "ood_wiki": "perplexity_ood_wiki",
    "ood_c4": "perplexity_ood_c4",
    "ood_ccnews": "perplexity_ood_ccnews",
}

# The three discretionary axes.  ("excl", "expand", "perwin") is the Sec 6 rule.
AXES = {
    "gen0": ("excl", "incl"),
    "window": ("expand", "trail3"),
    "tau": ("perwin", "single"),
}
REGISTERED = ("excl", "expand", "perwin")


def variant_names(corpus):
    for g0 in AXES["gen0"]:
        for win in AXES["window"]:
            for tau in AXES["tau"]:
                yield (corpus, g0, win, tau)


def vkey(v):
    return ".".join(v)


# ---------------------------------------------------------------- arithmetic

def ols_slope(pts):
    """OLS slope of y on x.  pts = [(x, y), ...].  None if < 2 points."""
    n = len(pts)
    if n < 2:
        return None
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    return sxy / sxx


def window_gens(g, gen0, window):
    lo = 0 if gen0 == "incl" else 1
    if window == "trail3":
        lo = max(lo, g - 2)
    return [k for k in range(lo, g + 1)]


def slope_series(logppl, gen0, window):
    """b(g) for g = 1..7; None where the window has < 2 points."""
    out = {}
    for g in range(1, 8):
        pts = [(k, logppl[k]) for k in window_gens(g, gen0, window) if k in logppl]
        out[g] = ols_slope(pts)
    return out


def pinc_series(logppl):
    return {g: logppl[g] - logppl[0] for g in logppl if g != 0}


# ---------------------------------------------------------------- disk

class MissingKey(Exception):
    pass


def md5_of(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def read_cell(arm, seed, gen, authorised):
    if arm not in CONTROL_ARMS and not authorised:
        sys.exit("REFUSE: non-control arm %r without --authorise-grid-read" % arm)
    p = COLLAPSE / arm / ("seed%d" % seed) / ("gen%d" % gen) / "metrics.json"
    if not p.exists():
        sys.exit("REFUSE: missing %s" % p)
    return json.loads(p.read_text())


def logppl_chain(arm, seed, key, authorised=False, inverse=False):
    out = {}
    for g in GENS:
        m = read_cell(arm, seed, g, authorised)
        if key not in m:
            raise MissingKey("key %r absent at %s seed%d gen%d (%d keys present)"
                             % (key, arm, seed, g, len(m)))
        v = float(m[key])
        if v <= 0:
            sys.exit("REFUSE: non-positive perplexity %r at %s seed%d gen%d" % (v, arm, seed, g))
        out[g] = math.log(v)
    if inverse:
        out = {g: -v for g, v in out.items()}
    return out


# ---------------------------------------------------------------- thresholds

def control_logppl(key, inverse=False):
    return {s: logppl_chain(TAU_ARM, s, key, authorised=False, inverse=inverse)
            for s in TAU_SEEDS}


def dcal_chain(arm, seed, authorised=False, inverse=False):
    """Delta_cal(g) = log PPL(g) - entropy(g), in-domain."""
    lp = logppl_chain(arm, seed, "perplexity", authorised=authorised)
    out = {}
    for g in GENS:
        m = read_cell(arm, seed, g, authorised)
        if "entropy" not in m:
            raise MissingKey("key 'entropy' absent at %s seed%d gen%d (%d keys present)"
                             % (arm, seed, g, len(m)))
        out[g] = lp[g] - float(m["entropy"])
    if inverse:
        out = {g: -v for g, v in out.items()}
    return out


def control_dcal(inverse=False):
    return {s: dcal_chain(TAU_ARM, s, authorised=False, inverse=inverse) for s in TAU_SEEDS}


def canary_dcal(chains):
    got = max(max(pinc_series(c).values()) for c in chains.values())
    return None if abs(got - DCAL_MAX_INC) <= DCAL_TOL else got


def taus_for_variant(chains, v):
    _, gen0, window, tau = v
    per = {}
    for g in range(1, 8):
        vals = [slope_series(c, gen0, window)[g] for c in chains.values()]
        vals = [x for x in vals if x is not None]
        per[g] = max(vals) if vals else None
    if tau == "single":
        allv = [x for x in per.values() if x is not None]
        one = max(allv) if allv else None
        return {g: (one if per[g] is not None else None) for g in per}
    return per


def canary_pinc(chains):
    bad = []
    for s, c in chains.items():
        got = max(pinc_series(c).values())
        want = PINC_MAX_BY_SEED[s]
        if abs(got - want) > PINC_TOL:
            bad.append((s, got, want))
    return bad


# ---------------------------------------------------------------- self-test

def self_test():
    n = p = 0

    def chk(name, cond):
        nonlocal n, p
        n += 1
        p += bool(cond)
        print("  %-58s %s" % (name, "PASS" if cond else "FAIL"))

    # 1-2. exact linear recovery, both windows, both gen-0 settings
    lin = {g: 3.0 + 0.25 * g for g in GENS}
    for gen0 in AXES["gen0"]:
        for win in AXES["window"]:
            b = slope_series(lin, gen0, win)
            ok = all(b[g] is None or abs(b[g] - 0.25) < 1e-12 for g in b)
            chk("linear series -> 0.25 (%s,%s)" % (gen0, win), ok)

    # 3. constant series -> 0
    const = {g: 2.5 for g in GENS}
    b = slope_series(const, "excl", "expand")
    chk("constant series -> 0", all(b[g] is None or abs(b[g]) < 1e-15 for g in b))

    # 4. THE Sec 6 CLAIM, made mechanical: a pure register step is not a trend.
    step = {0: 0.0}
    step.update({g: 0.07 for g in range(1, 8)})
    bx = slope_series(step, "excl", "expand")
    bi = slope_series(step, "incl", "expand")
    chk("register step: gen0 excluded -> slope 0", abs(bx[7]) < 1e-15)
    chk("register step: gen0 included -> slope > 0", bi[7] > 0)
    chk("including gen 0 raises the control slope", bi[7] > bx[7])

    # 5. trail3 definedness
    bt = slope_series(lin, "excl", "trail3")
    chk("trail3 defined from g=2 (excl)", bt[1] is None and bt[2] is not None)
    chk("trail3 window is 3 wide at g=7", window_gens(7, "excl", "trail3") == [5, 6, 7])
    chk("expand window is 1..g at g=7", window_gens(7, "excl", "expand") == [1, 2, 3, 4, 5, 6, 7])
    chk("incl window starts at gen 0", window_gens(3, "incl", "expand") == [0, 1, 2, 3])

    # 6. single tau >= per-window tau(g) for every g -- structural
    fake = {s: {g: 3.0 + 0.01 * s * g + 0.001 * g * g for g in GENS} for s in TAU_SEEDS}
    for gen0 in AXES["gen0"]:
        for win in AXES["window"]:
            pw = taus_for_variant(fake, ("in", gen0, win, "perwin"))
            sg = taus_for_variant(fake, ("in", gen0, win, "single"))
            ok = all(pw[g] is None or sg[g] >= pw[g] - 1e-15 for g in pw)
            chk("single tau >= per-window tau (%s,%s)" % (gen0, win), ok)

    # 7. tau is a maximum over control chains -> no control chain exceeds it
    for v in variant_names("in"):
        t = taus_for_variant(fake, v)
        ok = True
        for c in fake.values():
            b = slope_series(c, v[1], v[2])
            for g in range(1, 8):
                if b[g] is not None and t[g] is not None and b[g] > t[g] + 1e-15:
                    ok = False
        chk("no control chain exceeds tau [%s]" % vkey(v), ok)

    # 7b. Delta-slope sees the same axes as P-slope
    chk("dcal is enumerated over all 8 axis settings",
        len(list(variant_names("dcal"))) == len(list(variant_names("in"))) == 8)
    dc = {0: 0.30}
    dc.update({g: 0.30 + 0.0635 for g in range(1, 8)})
    chk("Delta_cal canary wiring", canary_dcal({43: dc, 44: dc, 45: dc, 46: dc}) is None)
    chk("Delta_cal canary catches a wrong series",
        canary_dcal({43: {g: 0.5 * g for g in GENS}}) is not None)

    # 8. the registered variant is in the enumeration exactly once
    reg = [v for v in variant_names("in") if v[1:] == REGISTERED]
    chk("registered variant enumerated exactly once", len(reg) == 1)

    # 9. P-inc canary wiring, on synthetic data with a known answer
    c = {0: 0.0}
    c.update({g: 0.0731 for g in range(1, 8)})
    chk("P-inc max recovered", abs(max(pinc_series(c).values()) - 0.0731) < 1e-12)

    # 10. the disk guard refuses an exposed arm
    try:
        read_cell("mix25_topk", 43, 1, authorised=False)
        chk("exposed arm refused without authorisation", False)
    except SystemExit:
        chk("exposed arm refused without authorisation", True)

    # 11. inverse: negating the series must break the P-inc canary
    neg = {g: -v for g, v in c.items()}
    chk("inverse breaks the P-inc canary", abs(max(pinc_series(neg).values()) - 0.0731) > PINC_TOL)

    print("\nself-test: %d / %d PASS" % (p, n))
    return 0 if p == n else 1


# ---------------------------------------------------------------- registration

def do_register(dry, inverse):
    if SENS.exists() and not dry:
        sys.exit("REFUSE: %s exists; it is never overwritten (md5 %s)" % (SENS, md5_of(SENS)))
    if not REGISTRY.exists():
        sys.exit("REFUSE: %s absent -- install the Sept 17 (c) tooling first" % REGISTRY)
    before = md5_of(REGISTRY)
    print("registered_thresholds.json md5 before: %s" % before)

    out = {"_note": "P-slope sensitivity variants. Reported, never ranked. "
                    "decisions.md Sept 17 (c) Sec 6 + the sensitivity rule.",
           "_tau_arm": TAU_ARM, "_tau_seeds": list(TAU_SEEDS), "variants": {}}

    for cname, key in CORPORA.items():
        try:
            chains = (control_dcal(inverse=inverse) if cname == "dcal"
                      else control_logppl(key, inverse=inverse))
        except MissingKey as e:
            print("SKIP series %s: %s" % (cname, e))
            continue

        if cname == "dcal":
            got = canary_dcal(chains)
            if inverse:
                if got is not None:
                    print("\nGATE STOP: inverse run broke the Delta_cal canary, as required "
                          "(%.6f vs %.4f)." % (got, DCAL_MAX_INC))
                    return 1
                print("\nCANARY NOT WIRED: inverse run passed the Delta_cal canary.")
                return 1
            if got is not None:
                print("\nGATE STOP: Delta_cal canary failed: %.6f vs %.4f"
                      % (got, DCAL_MAX_INC))
                return 1
            print("Delta_cal canary OK against %.4f" % DCAL_MAX_INC)

        if cname == "in":
            bad = canary_pinc(chains)
            if inverse:
                if bad:
                    print("\nGATE STOP: inverse run broke the P-inc canary, as required.")
                    for s, got, want in bad:
                        print("  seed %d: %.4f vs %.4f" % (s, got, want))
                    return 1
                print("\nCANARY NOT WIRED: inverse run passed the P-inc canary.")
                return 1
            if bad:
                print("\nGATE STOP: P-inc canary failed on %s." % TAU_ARM)
                for s, got, want in bad:
                    print("  seed %d: got %.6f want %.4f" % (s, got, want))
                return 1
            print("P-inc canary 4/4 OK against %s" %
                  " / ".join("%.4f" % PINC_MAX_BY_SEED[s] for s in TAU_SEEDS))

        for v in variant_names(cname):
            t = taus_for_variant(chains, v)
            out["variants"][vkey(v)] = {
                "ranked": False,   # nothing here is ever ranked; see the header
                "mirrors_registered_pslope": bool(cname == "in" and v[1:] == REGISTERED),
                "tau": {str(g): (None if t[g] is None else repr(t[g])) for g in t},
                "tau_rounded": {str(g): (None if t[g] is None else round(t[g], 6)) for g in t},
            }

    reg = out["variants"].get(vkey(("in",) + REGISTERED))
    if reg is None:
        sys.exit("REFUSE: the registered variant did not compute -- refusing to write a partial file")
    if out["variants"].get(vkey(("dcal",) + REGISTERED)) is None:
        sys.exit("REFUSE: Delta-slope did not compute -- refusing to write an asymmetric file")

    print("\n%-30s %s" % ("variant", " ".join("%9s" % ("tau(g%d)" % g) for g in range(1, 8))))
    for k, rec in out["variants"].items():
        if not (k.startswith("in.") or k.startswith("dcal.")):
            continue
        row = " ".join("%9s" % "--" if rec["tau_rounded"][str(g)] is None
                       else "%+9.6f" % rec["tau_rounded"][str(g)] for g in range(1, 8))
        print("%-30s %s%s" % (k, row,
              "   <- mirrors the ranked P-slope" if rec["mirrors_registered_pslope"] else ""))

    if dry:
        print("\n--dry-run: nothing written.")
    else:
        SENS.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print("\nwrote %s" % SENS)
        print("sensitivity_thresholds.json md5: %s" % md5_of(SENS))

    after = md5_of(REGISTRY)
    print("registered_thresholds.json md5 after:  %s   %s"
          % (after, "UNCHANGED" if after == before else "*** CHANGED -- STOP ***"))
    return 0 if after == before else 1


# ---------------------------------------------------------------- grid gate

def gate_grid(md5_given, ranked_verdicts):
    if not REGISTRY.exists():
        sys.exit("REFUSE: %s absent" % REGISTRY)
    reg = json.loads(REGISTRY.read_text())
    pending = [k for k, v in reg.items() if v is None]
    if pending:
        sys.exit("REFUSE: %d threshold(s) still pending: %s" % (len(pending), ", ".join(sorted(pending))))
    actual = md5_of(REGISTRY)
    if md5_given != actual:
        sys.exit("REFUSE: registry md5 %s does not match --authorise-grid-read %s" % (actual, md5_given))
    if not SENS.exists():
        sys.exit("REFUSE: run --register first")
    if not ranked_verdicts or not Path(ranked_verdicts).exists():
        sys.exit("REFUSE: --ranked-verdicts must point at the completed 5.4 output; "
                 "the sensitivity is read after the registered verdict, never instead of it")
    print("gate OK: registry complete, md5 %s, ranked verdicts on file" % actual)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--probe", action="store_true",
                    help="print src/detectors.py's public surface; reads no data")
    ap.add_argument("--register", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inverse", action="store_true",
                    help="negate the control series; the P-inc canary must then fail (GATE STOP)")
    ap.add_argument("--run", action="store_true", help="the grid half; gated")
    ap.add_argument("--authorise-grid-read", default=None)
    ap.add_argument("--ranked-verdicts", default=None)
    a = ap.parse_args()

    if a.probe:
        sys.path.insert(0, str(ROOT / "src"))
        import detectors
        print("detectors from %s" % detectors.__file__)
        for n in sorted(x for x in dir(detectors) if not x.startswith("_")):
            print("  %s" % n)
        return 0
    if a.self_test:
        return self_test()
    if a.register or a.dry_run:
        return do_register(a.dry_run, a.inverse)
    if a.run:
        gate_grid(a.authorise_grid_read, a.ranked_verdicts)
        print("NOT IMPLEMENTED: the head-to-head half is written after src/detectors.py's "
              "surface is probed on Narval.  Thresholds are complete; nothing was read.")
        return 1
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
