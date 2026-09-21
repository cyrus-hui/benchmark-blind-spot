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
