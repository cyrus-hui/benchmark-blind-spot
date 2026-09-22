#!/usr/bin/env python3
"""gen1_margins.py -- descriptive gen-1 margins (decisions.md Sept 21 Sec 24.3, Sec 27.3).

DESCRIPTIVE, NOT TESTED, NOT RANKED.  Per decoder x ratio x seed: Delta(1)/tau_Delta and
s(1)/tau_P-inc, where Delta = dcal_series and s = pinc_series from src/detectors.py and both
tau are recomputed at full precision from controls and verified against the registry (the
same values the read used).  Seeds share slices, so the three seeds are not independent
replicates; the replication is across ratios.  r = 0 uses control_gpt2m weights under both
decoders, so its rows are identical by construction.

  python gen1_margins.py --authorise-grid-read <md5> --ranked-verdicts logs/5p4_read.log
"""
import argparse
import os
import sys

import sens_common as C
from sens_common import D, E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--authorise-grid-read", metavar="MD5", required=True)
    ap.add_argument("--ranked-verdicts", required=True)
    a = ap.parse_args()
    reg = C.gate(a.authorise_grid_read)
    rd = D.Reader(a.root, allow_exposed=True)
    tau, _, dets_by, _ = C.canary_ranked(rd, reg, a.ranked_verdicts)
    td, tp = tau["dcal"], tau["pinc"]
    print(f"\ntau_Delta {td:+.6f} (registered {reg['dcal']['tau']})   tau_P-inc {tp:+.6f} (registered {reg['pinc']['tau']})")
    print("DESCRIPTIVE ONLY.  Seeds share slices: three seeds are not three replicates.\n")
    print(f"  {'decoder':7s} {'r':>4s} {'seed':>4s}   {'Delta(1)':>9s} {'/tau_D':>7s}   {'s(1)':>9s} {'/tau_P':>7s}   lat D  lat P-inc")
    for dec in D.DECODERS:
        for r in D.RATIOS:
            md, mp = [], []
            for s in D.EVAL_SEEDS:
                wm = rd.metrics(D.WEIGHTS_ARM[dec][r], s, E.WKEYS)
                d1, s1 = D.dcal_series(wm)[1], D.pinc_series(wm)[1]
                ld, lp = dets_by[dec]["dcal"]["lat"][(r, s)], dets_by[dec]["pinc"]["lat"][(r, s)]
                if (d1 > td) != (ld == 1) or (s1 > tp) != (lp == 1):
                    sys.exit(f"STOP: gen-1 margin disagrees with the read's latency at {dec} r={r} s{s}")
                md.append(d1 / td)
                mp.append(s1 / tp)
                print(f"  {dec:7s} {r:4d} {s:4d}   {d1:+9.5f} {d1 / td:7.2f}   {s1:+9.5f} {s1 / tp:7.2f}   "
                      f"{ld:5d}  {lp:5d}")
            print(f"  {'':7s} {r:4d} {'min':>4s}   {'':9s} {min(md):7.2f}   {'':9s} {min(mp):7.2f}")
    print("\nCANARY PASS: every gen-1 margin agrees with the read's latency-1 decision")
    print("RESULT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
