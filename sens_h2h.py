#!/usr/bin/env python3
"""sens_h2h.py -- the head-to-head half of pplslope_sensitivity.py --run.

REPORTED, NEVER RANKED (decisions.md Sept 17 (c) Sec 6; Sept 20 Sec 4).  Called only
from pplslope_sensitivity.py after its gate_grid() has passed.

  A. P-slope sensitivity.  For each of the 8 in-domain P-slope variants, the ranked
     Delta_cal (a level against tau_Delta) is compared with better-of {P-inc, P-slope
     variant}, exactly as the registered baseline was ranked.  The WEAKEST verdict per
     reading across the 8 is reported beside the registered one.  A worse verdict is a
     stated limitation; a better one changes nothing.
  B. OOD P-slope variants: Delta_cal vs the variant alone.  Reported, not in the weakest.
  C. Delta-slope mirror: each Delta-slope variant vs the P-slope variant with the SAME
     three axis settings (the differencing transform given to both).  Reported only.

Every threshold is read from sensitivity_thresholds.json (registered before the read)
and must first be reproduced EXACTLY from controls by pplslope_sensitivity.py's own code.
"""
import json
import sys

import sens_common as C
from sens_common import D

SENS_MD5 = "dd756a9628e124f402ecdd50dd187d0c"   # decisions.md Sept 20; plan.md 5.4


def stored_tau(rec):
    return {g: (None if rec["tau"][str(g)] is None else float(rec["tau"][str(g)])) for g in range(1, 8)}


def canary_sens(PS, sens):
    """Every stored variant tau reproduced from controls with the registering code, to repr."""
    n = 0
    for cname, key in PS.CORPORA.items():
        chains = PS.control_dcal() if cname == "dcal" else PS.control_logppl(key)
        if cname == "dcal" and PS.canary_dcal(chains) is not None:
            sys.exit("STOP: Delta_cal canary failed on the controls")
        if cname == "in" and PS.canary_pinc(chains):
            sys.exit("STOP: P-inc canary failed on the controls")
        for v in PS.variant_names(cname):
            rec = sens["variants"].get(PS.vkey(v))
            if rec is None:
                sys.exit(f"STOP: {PS.vkey(v)} absent from {PS.SENS}")
            t = PS.taus_for_variant(chains, v)
            got = {str(g): (None if t[g] is None else repr(t[g])) for g in t}
            if got != rec["tau"]:
                sys.exit(f"STOP: {PS.vkey(v)} does not reproduce from controls:\n  {got}\n  {rec['tau']}")
            n += 1
    print(f"CANARY PASS: {n}/{len(sens['variants'])} sensitivity threshold sets reproduce "
          f"exactly from {PS.TAU_ARM} {list(PS.TAU_SEEDS)}")
    if n != len(sens["variants"]):
        sys.exit("STOP: the sensitivity file holds variants this code does not enumerate")


class Chains:
    """Cache of grid chains read with pplslope_sensitivity.py's own reader."""

    def __init__(self, PS):
        self.PS, self.c = PS, {}

    def get(self, cname, key, arm, seed):
        k = (cname, arm, seed)
        if k not in self.c:
            self.c[k] = (self.PS.dcal_chain(arm, seed, authorised=True) if cname == "dcal"
                         else self.PS.logppl_chain(arm, seed, key, authorised=True))
        return self.c[k]


def variant_det(PS, ch, dec, cname, key, v, tau):
    det = {"score": {}, "lat": {}}
    for r in D.RATIOS:
        for s in D.EVAL_SEEDS:
            b = PS.slope_series(ch.get(cname, key, D.WEIGHTS_ARM[dec][r], s), v[1], v[2])
            ser = [None] + [b[g] for g in range(1, 8)]
            for g in range(1, 8):
                if (ser[g] is None) != (tau[g] is None):
                    sys.exit(f"STOP: series/threshold definedness differ at g{g} for {PS.vkey(v)}")
            det["score"][(r, s)] = ser[7]
            det["lat"][(r, s)] = D.first_flag(ser, tau)
    return det


def run(PS, log_path):
    reg = C.R.load()
    reg_md5 = C.R.md5()
    if C.md5(PS.SENS) != SENS_MD5:
        sys.exit(f"REFUSED: {PS.SENS} md5 {C.md5(PS.SENS)} is not the registered {SENS_MD5}")
    sens = json.loads(PS.SENS.read_text())

    rd = D.Reader(PS.COLLAPSE, allow_exposed=True)
    tau, sets, dets_by, ranked = C.canary_ranked(rd, reg, log_path)
    canary_sens(PS, sens)

    ch = Chains(PS)
    dets = {}                       # (cname, axes) -> {dec: det}
    skipped = []
    for cname, key in PS.CORPORA.items():
        for v in PS.variant_names(cname):
            t = stored_tau(sens["variants"][PS.vkey(v)])
            try:
                dets[v] = {dec: variant_det(PS, ch, dec, cname, key, v, t) for dec in D.DECODERS}
            except PS.MissingKey as e:
                if cname in ("in", "dcal"):
                    sys.exit(f"STOP: in-domain series missing on the grid: {e}")
                skipped.append(f"{PS.vkey(v)}: {e}")

    # canary: the registered variant, computed by this path, IS the ranked P-slope
    regv = ("in",) + PS.REGISTERED
    for dec in D.DECODERS:
        a, b = dets[regv][dec], dets_by[dec]["pslope"]
        if a["lat"] != b["lat"] or any(abs(a["score"][c] - b["score"][c]) > 1e-12 for c in b["score"]):
            sys.exit(f"STOP: {PS.vkey(regv)} does not reproduce the ranked P-slope under {dec}")
    print(f"CANARY PASS: {PS.vkey(regv)} reproduces the ranked P-slope (latency exact, score < 1e-12), both decoders")

    order = [k for k in ranked]
    print("\n=== registered (ranked) verdicts, rebuilt ===")
    for k in order:
        print(C.fmt_row(k, ranked[k]))
    ex = C.exit_of(ranked)
    print(f"  exit {ex['exit']}   beats PPL {ex['beats'].get('PPL')}")

    # A
    print("\n=== A. P-slope sensitivity: Delta_cal vs better-of {P-inc, P-slope variant}, in-domain ===")
    print("    reported, never ranked; the weakest verdict per reading is the sensitivity result")
    per_reading = {k: [] for k in order}
    exits = []
    for v in PS.variant_names("in"):
        tab = C.table_of(sets, dets_by, override={dec: {"pslope": dets[v][dec]} for dec in D.DECODERS})
        e = C.exit_of(tab)
        exits.append((PS.vkey(v), e))
        cells = []
        for k in order:
            if tab[k] is None:
                cells.append(f"{k}: n/e")
                continue
            per_reading[k].append(tab[k]["PPL"][1])
            cells.append(f"{k}: {tab[k]['PPL'][1]} ({tab[k]['PPL'][0]})")
        mark = "  <- registered" if v == regv else ""
        print(f"  {PS.vkey(v):28s} beats PPL {str(e['beats']['PPL']):5s}  " + "  ".join(cells) + mark)
    print("\n  weakest across the 8 in-domain variants, beside the registered verdict:")
    for k in order:
        if ranked[k] is None:
            print(f"    {k:15s} not evaluable")
            continue
        w = C.weakest(per_reading[k])
        flag = "" if C.RANK[w] >= C.RANK[ranked[k]["PPL"][1]] else "   <- WEAKER: a stated limitation"
        print(f"    {k:15s} registered {ranked[k]['PPL'][1]:7s} weakest {w:7s}{flag}")
    held = all(e["beats"]["PPL"] for _, e in exits)
    print(f"  'beats perplexity' holds under all 8 in-domain variants: {held}")
    if not held:
        print("    fails under: " + ", ".join(n for n, e in exits if not e["beats"]["PPL"]))

    # B
    print("\n=== B. OOD P-slope variants: Delta_cal vs the variant alone (reported, not in the weakest) ===")
    for cname in ("ood_wiki", "ood_c4", "ood_ccnews"):
        for v in PS.variant_names(cname):
            if v not in dets:
                continue
            cells = []
            for k in order:
                dec, pos, neg = sets[k]
                if len(pos) < 3:
                    cells.append(f"{k}: n/e")
                    continue
                cells.append(f"{k}: {D.compare(dets_by[dec]['dcal'], dets[v][dec], pos, neg)[0]}")
            print(f"  {PS.vkey(v):34s} " + "  ".join(cells))
    for s in skipped:
        print(f"  SKIP {s}")

    # C
    print("\n=== C. Delta-slope mirror: Delta-slope(axes) vs P-slope(same axes), in-domain (reported only) ===")
    for v in PS.variant_names("dcal"):
        pv = ("in",) + v[1:]
        cells = []
        for k in order:
            dec, pos, neg = sets[k]
            if len(pos) < 3:
                cells.append(f"{k}: n/e")
                continue
            st = D.stats(dets[v][dec], pos, neg)
            vd = D.compare(dets[v][dec], dets[pv][dec], pos, neg)[0]
            cells.append(f"{k}: {vd} (AUC {st['auc']:.3f}, lat {st['mean_lat']:.3f})")
        print(f"  {PS.vkey(v):28s} " + "  ".join(cells))
    print("  AUC resolution per reading is 1/(n_pos x 3); FPR at every variant tau is 0 on the")
    print("  controls by construction (each tau is the control maximum).")

    if C.R.md5() != reg_md5 or C.md5(PS.SENS) != SENS_MD5:
        sys.exit("STOP: a threshold file changed during the run")
    print(f"\nregistry md5 unchanged {reg_md5}; sensitivity md5 unchanged {SENS_MD5}")
    print("arms opened by the ranked rebuild:", sorted(rd.opened))
    print("RESULT OK")
    return 0
