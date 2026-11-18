#!/usr/bin/env python3
"""Phase 7.5 gate (decisions.md sec. 73). Run from ~/modelcollapse on a login node.

  python scripts/fc75_gate.py --g0                 # 9 gen-0 corpora          -> logs/fc75/gate_g0.json
  python scripts/fc75_gate.py --topkeval           # canary + type-C cells    -> logs/fc75/gate_topkeval.json
  python scripts/fc75_gate.py --topkeval --inverse # canary against greedy completions: must end GATE STOP
  python scripts/fc75_gate.py --pilot              # fc75_r50_t075 s43 g1     -> logs/fc75/gate_pilot.json
  python scripts/fc75_gate.py --all [--expect-keys 18]   # all 147 cells      -> logs/fc75/gate_all.json
  python scripts/fc75_gate.py --census             # counts only

Opens ONLY gate quantities: manifest counts and md5s, log lines, step counts, metrics.json KEY NAMES and
row counts. It prints no held-out metric value. Two value comparisons exist and print PASS/STOP only:
  - the top-k-eval CANARY, on gen 0 (values already read): top-k completions of a greedy arm's gen-0 weights
    against pilot_gpt2m gen 0. Perplexity |diff| <= 1e-6. Distinct-2 and KL-unigram are SAMPLED, so the
    tolerance is the registered control noise (tau_D 0.0105, tau_K 0.1111): exact -> PASS, inside ->
    ACCEPT-AND-RECORD, outside -> STOP. A decoder mix-up is 0.39 / 2.25 (the Sept 20 inverse canary).
  - weight identity at gen 7: the shadow's perplexity equals the weights arm's (teacher-forced), and its
    Distinct-2 differs from the greedy-completions value (the decoder really changed).
Per chain cell: mixing manifest row (ratio, counts), corpus md5 and lines, lineage (the synthetic source is this
chain's previous generation, unchanged since), log lines (mix, prompt source, eval and corpus decoding, complete,
no Traceback), scripts/check_cell.sh (md5 in manifest, 1752 steps, key count, no extra keys vs the twin), the
next corpus has 18,686 lines, and gen0 links to the right gen-0 corpus.
Verdict PASS only if every checked cell passes and n > 0 (for --all: all 147 cells are present).
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys, time

SCR = os.environ.get("SCRATCH", "")
ROOT = os.environ.get("FC75_ROOT", os.path.join(SCR, "collapse"))
SEEDS, GENS, N, STEPS = (43, 44, 45), range(1, 8), 18686, 1752
PILOT = ("fc75_r50_t075", 43, 1)
TAU_D, TAU_K, PPL_TOL, ROWS = 0.0105, 0.1111, 1e-6, 1000


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nlines(p):
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def read(p):
    return open(p, errors="replace").read().replace("\r", "\n")


def newest_log(pattern, must):
    """Newest log matching the glob that contains every string in `must`; else the newest; else None."""
    cands = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    for c in cands:
        t = read(c)
        if all(m in t for m in must):
            return c, t
    return (cands[0], read(cands[0])) if cands else (None, "")


def cond():
    c = json.load(open("fc75_conditions.json"))
    if not c["chains"]:
        sys.exit("STOP: fc75_conditions.json lists no chain (n = 0)")
    return c


def fl(x):
    return str(float(x))


# ---------------------------------------------------------------- gen-0 corpora

def gate_g0(c):
    fails, n = [], 0
    for t, arm in sorted(c["gen0_corpus_roots"].items()):
        for s in SEEDS:
            n += 1
            F = []
            d = os.path.join(ROOT, arm, f"seed{s}", "gen0")
            src = os.path.join(ROOT, "pilot_gpt2m", f"seed{s}", "gen0")
            corp = os.path.join(d, "synthetic_corpus.jsonl")
            if not os.path.isfile(corp):
                fails.append(f"{arm} s{s}: missing {corp}"); continue
            m = md5(corp)
            if nlines(corp) != N: F.append(f"lines {nlines(corp)} != {N}")
            if m == md5(os.path.join(src, "synthetic_corpus.jsonl")): F.append("corpus is byte-identical to the T = 1 corpus")
            if os.path.islink(d) or not os.path.islink(os.path.join(d, "model")): F.append("gen0 must be a real dir with a model symlink")
            if os.path.realpath(os.path.join(d, "model")) != os.path.realpath(os.path.join(src, "model")): F.append("model does not resolve to the pilot gen-0 weights")
            mj = os.path.join(d, "metrics.json")
            if os.path.islink(mj) or not os.path.isfile(mj) or md5(mj) != md5(os.path.join(src, "metrics.json")): F.append("metrics.json is not an unchanged copy of the pilot's gen 0")
            log, txt = newest_log("logs/fc75/fc75_g0-*.out", [f"arm={arm} seed={s} ", "=== g0 corpus done"])
            for want in (f"arm={arm} seed={s} ", f"decoding[corpus] strategy=top_k temperature={fl(t)} top_k=50",
                         f"prompt source: default ({N} rows)", f"lines {N} md5 {m}", "=== g0 corpus done"):
                if want not in txt: F.append(f"log {log}: no '{want}'")
            if "Traceback" in txt: F.append(f"log {log}: Traceback")
            if F: fails.append(f"{arm} s{s}: " + "; ".join(F))
            else: print(f"PASS  {arm} s{s}  {N} lines  md5 {m}  T={fl(t)}")
    return fails, n, 9


# ---------------------------------------------------------------- top-k eval of greedy-corpus checkpoints

def mload(arm, s, g):
    p = os.path.join(ROOT, arm, f"seed{s}", f"gen{g}", "metrics.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def gate_topkeval(c, inverse):
    if not c["C"] or not c["canary"]:
        sys.exit("STOP: no C condition is registered in fc75_conditions.json (n = 0)")
    can = c["canary"]
    ref_arm = "gpt2m_greedy" if inverse else "pilot_gpt2m"
    sh, rf = mload(can["arm"], can["seed"], can["gen"]), mload(ref_arm, can["seed"], can["gen"])
    if sh is None or rf is None:
        sys.exit(f"STOP: canary metrics.json missing ({can['arm']} or {ref_arm})")
    print(f"canary: {can['arm']} s{can['seed']} g{can['gen']} (top-k completions) against {ref_arm}"
          + ("   [INVERSE: expect GATE STOP]" if inverse else ""))
    stop = rec = 0
    rows = nlines(os.path.join(ROOT, can["arm"], f"seed{can['seed']}", f"gen{can['gen']}", "eval_completions.jsonl"))
    v = "PASS" if rows == ROWS else "STOP"; stop += v == "STOP"
    print(f"  eval rows         {rows:>12d}   expect {ROWS}        {v}")
    d = float(sh["perplexity"]) - float(rf["perplexity"])
    v = "PASS" if abs(d) <= PPL_TOL else "STOP"; stop += v == "STOP"
    print(f"  perplexity  diff  {d:+12.3e}   tol {PPL_TOL:.0e}       {v}")
    for k, tol in (("distinct2", TAU_D), ("kl_unigram", TAU_K)):
        d = float(sh[k]) - float(rf[k])
        v = "PASS" if d == 0 else ("ACCEPT-AND-RECORD" if abs(d) <= tol else "STOP")
        stop += v == "STOP"; rec += v == "ACCEPT-AND-RECORD"
        print(f"  {k:11s} diff  {d:+12.6f}   exact | <= {tol} | STOP   {v}")
    print(f"record-lines={rec}")
    print("GATE", "STOP" if stop else "PASS", f"({stop} stop lines)")
    if inverse:
        sys.exit(0 if stop else 1)          # the inverse canary succeeds by stopping; it writes no gate file
    fails = [f"canary: {stop} stop line(s)"] if stop else []
    n = 0
    for cc in c["C"]:
        for s in SEEDS:
            n += 1
            F = []
            sh, wt = mload(cc["completions_arm"], s, 7), mload(cc["arm"], s, 7)
            if sh is None or wt is None:
                fails.append(f"{cc['completions_arm']} s{s}: metrics.json missing"); continue
            if len(sh) != 9: F.append(f"{len(sh)} keys != 9")
            cell = os.path.join(ROOT, cc["completions_arm"], f"seed{s}", "gen7")
            if nlines(os.path.join(cell, "eval_completions.jsonl")) != ROWS: F.append("eval rows != 1000")
            if abs(float(sh["perplexity"]) - float(wt["perplexity"])) > PPL_TOL: F.append("perplexity differs from the weights arm: wrong model")
            if float(sh["distinct2"]) == float(wt["distinct2"]): F.append("Distinct-2 equals the greedy-completions value: decoder did not change")
            if os.path.realpath(os.path.join(cell, "model")) != os.path.realpath(os.path.join(ROOT, cc["arm"], f"seed{s}", "gen7", "model")): F.append("model symlink does not resolve to the weights arm")
            log, txt = newest_log("logs/fc75/fc75_topkeval-*.out", [f"arm={cc['completions_arm']} seed={s} gen=7 ", "=== topkeval done"])
            for want in (f"arm={cc['completions_arm']} seed={s} gen=7 ", "decoding[eval] strategy=top_k temperature=1.0 top_k=50", "=== topkeval done"):
                if want not in txt: F.append(f"log {log}: no '{want}'")
            if "Traceback" in txt: F.append(f"log {log}: Traceback")
            if F: fails.append(f"{cc['completions_arm']} s{s}: " + "; ".join(F))
            else: print(f"PASS  {cc['completions_arm']} s{s} g7  (weights identical, decoder changed, 9 keys, {ROWS} rows)")
    return fails, n, 3 * len(c["C"])


# ---------------------------------------------------------------- chain cells

def manifest_rows(rr, s, g):
    p = os.path.join(rr, "mix_manifest.tsv")
    if not os.path.isfile(p):
        return []
    rows = [l.split("\t") for l in open(p).read().splitlines() if l.strip()]
    return [r for r in rows if len(r) == 13 and r[1] == str(g) and r[2] == str(s)]


def check_cell(ch, slice_fmt, s, g, keys):
    arm, r, t = ch["arm"], ch["r"], ch["corpus_temperature"]
    rr = os.path.join(ROOT, arm)
    cell = os.path.join(rr, f"seed{s}", f"gen{g}")
    F = []
    ok = lambda cnd, msg: None if cnd else F.append(msg)
    rows = manifest_rows(rr, s, g)
    ok(len(rows) >= 1, f"P1 manifest rows: {len(rows)}")
    if not rows:
        return F
    row = rows[-1]
    ns = math.ceil(r / 100 * N)
    ok(len({x[8] for x in rows}) == 1, "P1 manifest rows disagree on the output md5")
    ok(row[3] == f"{r / 100:.2f}" and int(row[4]) == N and int(row[5]) == ns and int(row[6]) == N - ns,
       f"P1 composition ratio={row[3]} total={row[4]} synth={row[5]} real={row[6]} (expected {r / 100:.2f} {N} {ns} {N - ns})")
    mixed = os.path.join(rr, "mixed", f"seed{s}", f"gen{g}.jsonl")
    ok(row[7] == mixed and os.path.isfile(mixed) and md5(mixed) == row[8] and nlines(mixed) == N, "P2 mixed corpus path / md5 / lines")
    synth = os.path.join(rr, f"seed{s}", f"gen{g - 1}", "synthetic_corpus.jsonl")
    ok(row[11] == synth and os.path.isfile(synth) and md5(synth) == row[12], "P3 lineage: synthetic source is not this chain's previous generation, or changed since")
    if N - ns > 0:
        sl = slice_fmt % g
        ok(row[9] == sl and os.path.isfile(sl) and md5(sl) == row[10], "P3 lineage: real slice path / md5")
    log, txt = newest_log(f"logs/fc75/{arm}-s{s}-g{g}-*.out", [f"=== gen {g} complete"])
    ok(log is not None, "P4 no log")
    if log:
        for want in (f"5.3 mix: ratio={fl(r / 100)} gen={g} seed={s} ", f"prompt source: {mixed} ({N} rows)",
                     "decoding[eval] strategy=top_k temperature=1.0 top_k=50",
                     f"decoding[corpus] strategy=top_k temperature={fl(t if t is not None else 1.0)} top_k=50",
                     f"=== gen {g} complete"):
            ok(want in txt, f"P4 log has no '{want}'")
        ok("Traceback" not in txt, "P4 Traceback in log")
        twin = os.path.join(ROOT, "pilot_gpt2m", f"seed{s}", f"gen{g}")
        cc = subprocess.run(["bash", "scripts/check_cell.sh", "--cell", cell, "--log", log, "--corpus", mixed,
                             "--manifest", os.path.join(rr, "mix_manifest.tsv"), "--expect-steps", str(STEPS),
                             "--expect-keys", str(keys), "--twin", twin], capture_output=True, text=True)
        ok(cc.returncode == 0, f"P5 check_cell.sh rc={cc.returncode}"
           + ("" if cc.returncode == 0 else " :: " + " | ".join(l for l in cc.stdout.splitlines() if l.startswith("FAIL"))))
    nxt = os.path.join(cell, "synthetic_corpus.jsonl")
    ok(os.path.isfile(nxt) and nlines(nxt) == N, f"P6 next corpus missing or not {N} lines")
    if g == 1:
        g0 = os.path.join(rr, f"seed{s}", "gen0")
        want = os.path.join(ROOT, ch["g0_root"], f"seed{s}", "gen0")
        ok(os.path.islink(g0) and os.path.realpath(g0) == os.path.realpath(want), f"P7 gen0 does not link to {want}")
    return F


def main():
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    for f in ("g0", "topkeval", "pilot", "all", "census"):
        m.add_argument("--" + f, action="store_true")
    ap.add_argument("--inverse", action="store_true", help="with --topkeval: canary against greedy completions")
    ap.add_argument("--expect-keys", type=int, default=9, help="9 before the scoring pass, 18 after")
    a = ap.parse_args()
    if not ROOT or not os.path.isdir(ROOT):
        sys.exit(f"STOP: collapse root {ROOT!r} not found")
    c = cond()
    for ch in c["chains"]:
        t = ch["corpus_temperature"]
        ch["g0_root"] = c["gen0_corpus_roots"][f"{t:.2f}"] if t is not None else "pilot_gpt2m"
    cells = [(ch, s, g) for ch in c["chains"] for s in SEEDS for g in GENS]
    done = lambda ch, s, g: os.path.isfile(os.path.join(ROOT, ch["arm"], f"seed{s}", f"gen{g}", "metrics.json"))
    complete = [x for x in cells if done(*x)]
    if a.census:
        print(f"complete cells: {len(complete)}/{len(cells)}")
        for ch in c["chains"]:
            print(f"  {ch['arm']}: " + " ".join(f"s{s}:{sum(done(ch, s, g) for g in GENS)}/7" for s in SEEDS))
        return
    name = "g0" if a.g0 else "topkeval" if a.topkeval else "pilot" if a.pilot else "all"
    if a.g0:
        fails, n, expect = gate_g0(c)
    elif a.topkeval:
        fails, n, expect = gate_topkeval(c, a.inverse)
    else:
        import yaml
        slice_fmt = yaml.safe_load(open("configs/mix10_topk.yaml"))["mix_slice_fmt"]
        todo = [x for x in cells if (x[0]["arm"], x[1], x[2]) == PILOT] if a.pilot else complete
        if a.pilot and not (todo and done(*todo[0])):
            sys.exit("STOP: the pilot cell has no metrics.json yet")
        fails, n = [], 0
        for ch, s, g in todo:
            n += 1
            F = check_cell(ch, slice_fmt, s, g, a.expect_keys)
            if F: fails.append(f"{ch['arm']} s{s} g{g}: " + "; ".join(F))
            elif a.pilot: print(f"PASS  {ch['arm']} s{s} g{g}")
        expect = 1 if a.pilot else len(cells)
    verdict = "PASS" if not fails and n == expect and n > 0 else "FAIL"
    for f in fails:
        print("FAIL ", f)
    print(f"checked {n}, expected {expect}, failed {len(fails)}  ->  GATE {verdict}")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.makedirs("logs/fc75", exist_ok=True)
    out = {"verdict": verdict, "gate": name, "checked": n, "expected": expect, "fails": fails, "head": head,
           "expect_keys": a.expect_keys, "script_md5": md5(os.path.abspath(__file__)), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(out, open(f"logs/fc75/gate_{name}.json", "w"), indent=1)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
