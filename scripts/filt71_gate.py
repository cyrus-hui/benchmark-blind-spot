#!/usr/bin/env python3
"""Phase 7.1 gate (decisions.md §62.8-§62.9, §64). Run from ~/modelcollapse on a login node.

  python scripts/filt71_gate.py --pilot           # greedy F(0.25) s43 g1 -> logs/filt71/gate_pilot.json
  python scripts/filt71_gate.py --all             # every completed cell  -> logs/filt71/gate_all.json
  python scripts/filt71_gate.py --census          # counts only

Opens ONLY gate quantities: manifest counts and md5s, log lines, step counts, metrics.json KEY NAMES.
It never reads a metric value (§62.9.2). Per cell:
  G1 filter manifest: >= 1 row for (arm,q,decoder,seed,gen); all rows agree on k and out_md5;
     kept = n_total - k; K-F1 PASS at gen 1; K-F3/K-F4 PASS.
  G2 training corpus on disk: md5 = manifest out_md5, lines = kept; the log names that md5.
  G3 R only: k = the paired F row's k; gen 1: identical unscorable set (md5) (§64.1-§64.2).
  G4 log: '7.1 filter OK', 'prompt source: default (18686 rows)', '=== gen G complete', no Traceback.
  G5 scripts/check_cell.sh: corpus md5 in the manifest (column 9), step parity = 3*ceil(kept/32),
     metrics.json key count (default 9: before the scoring pass), no extra keys vs the U twin.
Verdict PASS only if every checked cell passes, n > 0, and (for --all) all 252 cells are present.
"""
import argparse, glob, hashlib, json, math, os, re, subprocess, sys, time

SCR = os.environ.get("SCRATCH", "")
ROOT = os.environ.get("FILT71_ROOT", os.path.join(SCR, "collapse"))
QS = {10: "0.1", 25: "0.25", 50: "0.5"}
DECS = ("greedy", "topk")
U = {"greedy": "gpt2m_greedy", "topk": "pilot_gpt2m"}
SEEDS, GENS, N = (43, 44, 45), range(1, 8), 18686
HEADER = None  # read from each manifest


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(arm_root):
    p = os.path.join(arm_root, "filter_manifest.tsv")
    if not os.path.isfile(p):
        return []
    L = open(p).read().splitlines()
    if not L:
        return []
    hdr = L[0].split("\t")
    return [dict(zip(hdr, l.split("\t"))) for l in L[1:] if l.strip()]


def newest_log(name, s, g):
    cands = sorted(glob.glob(f"logs/filt71/{name}-s{s}-g{g}-*.out"), key=os.path.getmtime, reverse=True)
    for c in cands:
        t = open(c, errors="replace").read().replace("\r", "\n")
        if f"=== gen {g} complete" in t:
            return c, t
    return (cands[0], open(cands[0], errors="replace").read().replace("\r", "\n")) if cands else (None, "")


def check_cell(arm, qi, dec, s, g, keys):
    nm = f"filt71_{arm}{qi}_{dec}"
    rr = os.path.join(ROOT, nm)
    cell = os.path.join(rr, f"seed{s}", f"gen{g}")
    P, F = [], []
    ok = lambda c, m: (P if c else F).append(m)
    rows = [r for r in manifest(rr) if r["arm"] == arm and r["q"] == QS[qi] and r["decoder"] == dec
            and r["seed"] == str(s) and r["gen"] == str(g)]
    ok(len(rows) >= 1, f"G1 manifest rows: {len(rows)}")
    if not rows:
        return nm, F, P
    ks, outs = {r["k"] for r in rows}, {r["out_md5"] for r in rows}
    ok(len(ks) == 1 and len(outs) == 1, f"G1 rows agree: k {sorted(ks)} out_md5 {len(outs)} distinct")
    r = rows[-1]
    k, kept, nt = int(r["k"]), int(r["kept"]), int(r["n_total"])
    ok(nt == N and kept == nt - k and k > 0, f"G1 counts n_total={nt} k={k} kept={kept}")
    ok(r["kf1"].startswith("PASS") if g == 1 else r["kf1"].startswith("n/a"), f"G1 K-F1 {r['kf1']}")
    ok(r["kf3"] == "PASS" and r["kf4"] == "PASS", f"G1 K-F3 {r['kf3']} K-F4 {r['kf4']}")
    fc = os.path.join(rr, "filtered", f"seed{s}", f"gen{g}.jsonl")
    if os.path.isfile(fc):
        m = md5(fc); nl = sum(1 for _ in open(fc, "rb"))
        ok(m == r["out_md5"] and nl == kept, f"G2 corpus md5 {'==' if m == r['out_md5'] else '!='} manifest, lines {nl}/{kept}")
    else:
        ok(False, f"G2 missing {fc}")
    if arm == "R":
        fr = [x for x in manifest(os.path.join(ROOT, f"filt71_F{qi}_{dec}")) if x["arm"] == "F"
              and x["q"] == QS[qi] and x["decoder"] == dec and x["seed"] == str(s) and x["gen"] == str(g)]
        ok(bool(fr) and {x["k"] for x in fr} == {r["k"]}, f"G3 R k={r['k']} vs F k={sorted({x['k'] for x in fr})}")
        if g == 1:
            ok(bool(fr) and {x["unscorable_md5"] for x in fr} == {r["unscorable_md5"]}, "G3 gen-1 unscorable set = F's")
    log, t = newest_log(nm, s, g)
    ok(log is not None, f"G4 log {log}")
    if log:
        ok("7.1 filter OK" in t, "G4 '7.1 filter OK' in log")
        ok(f"md5 {r['out_md5']}" in t, "G4 log names the training-corpus md5")
        ok(f"prompt source: default ({N} rows)" in t, f"G4 prompt source default ({N} rows)")
        ok(f"=== gen {g} complete" in t, f"G4 '=== gen {g} complete'")
        ok("Traceback" not in t, "G4 no Traceback")
        twin = os.path.join(ROOT, U[dec], f"seed{s}", f"gen{g}")
        cc = subprocess.run(["bash", "scripts/check_cell.sh", "--cell", cell, "--log", log, "--corpus", fc,
                             "--manifest", os.path.join(rr, "filter_manifest.tsv"),
                             "--expect-steps", str(3 * math.ceil(kept / 32)), "--expect-keys", str(keys),
                             "--twin", twin], capture_output=True, text=True)
        ok(cc.returncode == 0, f"G5 check_cell.sh rc={cc.returncode} steps={3 * math.ceil(kept / 32)}"
           + ("" if cc.returncode == 0 else " :: " + " | ".join(l for l in cc.stdout.splitlines() if l.startswith("FAIL"))))
    return nm, F, P


def all_cells():
    for qi in QS:
        for dec in DECS:
            for arm in ("F", "R"):
                for s in SEEDS:
                    for g in GENS:
                        yield arm, qi, dec, s, g


def done(arm, qi, dec, s, g):
    return os.path.isfile(os.path.join(ROOT, f"filt71_{arm}{qi}_{dec}", f"seed{s}", f"gen{g}", "metrics.json"))


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--pilot", action="store_true"); m.add_argument("--all", action="store_true")
    m.add_argument("--census", action="store_true")
    ap.add_argument("--expect-keys", type=int, default=9)
    a = ap.parse_args()
    if not ROOT or not os.path.isdir(ROOT):
        sys.exit(f"STOP: collapse root {ROOT!r} not found")
    complete = [c for c in all_cells() if done(*c)]
    if a.census:
        print(f"complete cells: {len(complete)}/252")
        for qi in QS:
            for dec in DECS:
                for arm in ("F", "R"):
                    print(f"  filt71_{arm}{qi}_{dec}: " + " ".join(
                        f"s{s}:{sum(done(arm, qi, dec, s, g) for g in GENS)}/7" for s in SEEDS))
        return
    cells = [("F", 25, "greedy", 43, 1)] if a.pilot else complete
    if a.pilot and not done(*cells[0]):
        sys.exit("STOP: pilot cell has no metrics.json yet")
    fails, n = [], 0
    for c in cells:
        nm, F, P = check_cell(*c, keys=a.expect_keys); n += 1
        tag = f"{nm} s{c[3]} g{c[4]}"
        if F:
            fails.append({"cell": tag, "fail": F})
            print(f"FAIL  {tag}"); [print(f"        {x}") for x in F]
        else:
            print(f"PASS  {tag}  ({len(P)} checks)")
    verdict = "PASS" if n > 0 and not fails and (a.pilot or len(complete) == 252) else \
              ("FAIL" if fails or n == 0 else "PARTIAL")
    out = {"verdict": verdict, "mode": "pilot" if a.pilot else "all", "cells_checked": n,
           "complete": len(complete), "failures": fails, "expect_keys": a.expect_keys,
           "head": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "script_md5": md5(os.path.abspath(__file__))}
    os.makedirs("logs/filt71", exist_ok=True)
    p = f"logs/filt71/gate_{out['mode']}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"--- {n} cell(s) checked, {len(fails)} failed, {len(complete)}/252 complete -> {verdict}  ({p})")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
