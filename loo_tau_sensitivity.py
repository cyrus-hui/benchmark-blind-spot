#!/usr/bin/env python3
"""loo_tau_sensitivity.py -- leave-one-out threshold sensitivity (decisions.md Sept 21 Sec 22.2).

REPORTED, NEVER RANKED.  Every registered tau is recomputed four times from the control
arms, dropping one of s43-46; the verdict table is recomputed under each fold with
detector_eval_5p4.py's own build(); the weakest verdict per reading is reported beside
the registered one.  A worse verdict is a stated limitation; a better one changes nothing.
LOO FPR is not reported (1/n by construction for a max rule).

Seed sets.  Weights taus (dcal, pinc, pslope) and top-k completion taus: 43-46.  Greedy
completion taus (SIGMA, outcome label): 43-46 per the Sept 20 (b) Sec 13 amendment,
although src/detectors.py's COMPLETIONS_TAU still lists 43-45.  Canary B shows the
amended set reproduces the registry and the read-log verdicts before any fold runs.
Negatives stay s43-45 at both decoders (unchanged); in each fold the dropped seed is a
held-out control, so "separates" is reported as its TPR half only -- the other half is
the LOO FPR this analysis does not report.

  python loo_tau_sensitivity.py --self-test
  python loo_tau_sensitivity.py --authorise-grid-read <md5> --ranked-verdicts logs/5p4_read.log
"""
import argparse
import contextlib
import os
import sys

import sens_common as C
from sens_common import D, E

FOLDS = (43, 44, 45, 46)
AMENDED = {"weights": [43, 44, 45, 46], "topk": [43, 44, 45, 46], "greedy": [43, 44, 45, 46]}


@contextlib.contextmanager
def tau_seeds(sets):
    """Temporarily point detectors' threshold seed lists at `sets`; restored on exit."""
    saved = (D.WEIGHTS_TAU, dict(D.COMPLETIONS_TAU))
    try:
        D.WEIGHTS_TAU = (saved[0][0], list(sets["weights"]))
        D.COMPLETIONS_TAU = {dec: (saved[1][dec][0], list(sets[dec])) for dec in D.DECODERS}
        yield
    finally:
        D.WEIGHTS_TAU, D.COMPLETIONS_TAU = saved
    if (D.WEIGHTS_TAU, D.COMPLETIONS_TAU) != saved:
        sys.exit("STOP: threshold seed lists were not restored")


def drop(k):
    return {name: [s for s in seeds if s != k] for name, seeds in AMENDED.items()}


def flat(tau):
    out = {"dcal": tau["dcal"], "pinc": tau["pinc"]}
    out.update({f"pslope g{g}": tau["pslope"][g] for g in range(2, 8)})
    for dec in D.DECODERS:
        out.update({f"sigma {dec} {k}": tau["sigma"][dec][k] for k in ("sub", "full")})
        out.update({f"outcome {dec} {k}": tau["outcome"][dec][k] for k in ("tau_D", "tau_K")})
    return out


def fold_table(rd, tau):
    sets, dets_by = C.chain_sets(rd, tau)
    return C.table_of(sets, dets_by)


def run(root, md5_given, log_path):
    reg = C.gate(md5_given)
    reg_md5 = C.R.md5()
    rd = D.Reader(root, allow_exposed=True)

    # Canary A: the code's own seed sets rebuild the read log exactly.
    tau_code, _, _, ranked = C.canary_ranked(rd, reg, log_path)

    # Canary B: the amended seed sets reproduce the registry and the verdict table.
    with tau_seeds(AMENDED):
        before = E.fails
        tau_amd = E.control_thresholds(rd, D.DECODERS)
        E.verify_against_registry(tau_amd, reg, D.DECODERS)
        if E.fails != before:
            sys.exit("STOP: the Sec 13 amended seed sets do not reproduce the registry")
        t_amd = fold_table(rd, tau_amd)
    d = C.diff_tables(t_amd, ranked)
    if d:
        sys.exit("STOP: the amended seed sets change the ranked verdict table:\n  " + "\n  ".join(d))
    fc, fa = flat(tau_code), flat(tau_amd)
    moved = [k for k in fc if fc[k] != fa[k]]
    print("CANARY PASS: Sec 13 amended seed sets (greedy 43-46) reproduce the registry and the "
          "read-log verdicts")
    print("  full-precision tau that differ between code (greedy 43-45) and amendment (43-46): "
          + (", ".join(moved) if moved else "none"))

    order = list(ranked)
    reg_flat = flat(tau_amd)
    folds = {}
    for k in FOLDS:
        with tau_seeds(drop(k)):
            tau = E.control_thresholds(rd, D.DECODERS)
        folds[k] = (flat(tau), fold_table(rd, tau))

    print("\n=== thresholds by fold (full set = amended, 43-46) ===")
    print(f"  {'tau':20s} {'full set':>11s} " + " ".join(f"{'drop s%d' % k:>11s}" for k in FOLDS))
    for name in reg_flat:
        print(f"  {name:20s} {reg_flat[name]:+11.5f} " + " ".join(f"{folds[k][0][name]:+11.5f}" for k in FOLDS))

    print("\n=== verdict table by fold ('separates' = TPR half; see header) ===")
    print("  registered:")
    for r in order:
        print("  " + C.fmt_row(r, ranked[r]))
    print(f"    exit {C.exit_of(ranked)['exit']}")
    for k in FOLDS:
        tab = folds[k][1]
        print(f"  drop s{k}:")
        for r in order:
            print("  " + C.fmt_row(r, tab.get(r)))
        e = C.exit_of(tab)
        print(f"    exit {e['exit']}   beats SIGMA {e['beats'].get('SIGMA')}   beats PPL {e['beats'].get('PPL')}")
        d = C.diff_tables(tab, ranked)
        print("    differs from registered: " + ("; ".join(d) if d else "no"))

    print("\n=== weakest across the four folds, beside the registered verdict ===")
    worse = []
    for r in order:
        evals = [folds[k][1][r] for k in FOLDS if folds[k][1].get(r) is not None]
        n_e = len(evals)
        if ranked[r] is None:
            note = (f"not evaluable registered; evaluable in {n_e}/4 folds"
                    + (": " + ", ".join(f"{f} {C.weakest([v[f][1] for v in evals])}" for f in C.FAMS) if n_e else ""))
            print(f"  {r:15s} {note}")
            if n_e and any(C.RANK[C.weakest([v[f][1] for v in evals])] < C.RANK["CEILING"] for f in C.FAMS):
                worse.append(f"{r} becomes evaluable with a TIE/LOSS")
            continue
        cells = []
        for f in C.FAMS:
            w = C.weakest([v[f][1] for v in evals]) if evals else "-"
            reg_v = ranked[r][f][1]
            flag = " <- WEAKER" if evals and C.RANK[w] < C.RANK[reg_v] else ""
            if flag:
                worse.append(f"{r} {f}")
            cells.append(f"{f} registered {reg_v:7s} weakest {w:7s}{flag}")
        sep = all(v["separates"] for v in evals)
        if not sep:
            worse.append(f"{r} separates (TPR half)")
        cells.append(f"TPR half holds {sum(v['separates'] for v in evals)}/{n_e}")
        if n_e < 4:
            cells.append(f"not evaluable in {4 - n_e} fold(s)")
        print(f"  {r:15s} " + "   ".join(cells))
    exits = {k: C.exit_of(folds[k][1])["exit"] for k in FOLDS}
    print("  exit by fold: " + "  ".join(f"s{k}: {v}" for k, v in exits.items()))
    print("  WEAKER THAN REGISTERED (stated limitations): " + ("; ".join(worse) if worse else "none"))

    if C.R.md5() != reg_md5:
        sys.exit("STOP: the registry changed during the run")
    print(f"\nregistry md5 unchanged {reg_md5}")
    print("RESULT OK")
    return 0


def self_test():
    n = p = 0

    def chk(name, ok):
        nonlocal n, p
        n += 1
        p += bool(ok)
        print(f"  {name:62s} {'PASS' if ok else 'FAIL'}")

    saved = (D.WEIGHTS_TAU, dict(D.COMPLETIONS_TAU))
    with tau_seeds(drop(44)):
        chk("inside: weights seeds exclude 44", D.WEIGHTS_TAU[1] == [43, 45, 46])
        chk("inside: greedy uses the amended set minus 44", D.COMPLETIONS_TAU["greedy"][1] == [43, 45, 46])
        chk("inside: arms unchanged", D.WEIGHTS_TAU[0] == saved[0][0]
            and all(D.COMPLETIONS_TAU[d][0] == saved[1][d][0] for d in D.DECODERS))
    chk("restored after the context", (D.WEIGHTS_TAU, D.COMPLETIONS_TAU) == saved)
    try:
        with tau_seeds(drop(43)):
            raise RuntimeError("x")
    except RuntimeError:
        pass
    chk("restored after an exception", (D.WEIGHTS_TAU, D.COMPLETIONS_TAU) == saved)
    chk("fold 46 leaves greedy at 43-45", drop(46)["greedy"] == [43, 44, 45])
    chk("every fold has 3 seeds for every tau", all(len(v) == 3 for k in FOLDS for v in drop(k).values()))
    chk("E.control_thresholds reads D at call time", "D.WEIGHTS_TAU" in open(E.__file__).read())
    print(f"\nloo self-test: {p} / {n} PASS")
    return 0 if p == n else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--authorise-grid-read", metavar="MD5")
    ap.add_argument("--ranked-verdicts")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.authorise_grid_read or not a.ranked_verdicts:
        ap.error("--authorise-grid-read and --ranked-verdicts are both required")
    return run(a.root, a.authorise_grid_read, a.ranked_verdicts)


if __name__ == "__main__":
    sys.exit(main())
