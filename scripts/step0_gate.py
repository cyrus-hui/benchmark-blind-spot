#!/usr/bin/env python
"""Gate for 7.1 step 0 outputs. Reads canary/meta JSON and K3' regen logs only; prints NO statistic of S.
  --pilot : task 1 only.  --all : every manifest task.
PASS requires: json present, no FAIL verdict; npz present for synth/null;
n_scored + n_excluded == candidates; one script_md5 and one git_head across all tasks;
S finite (checked by the scorer); and K3' (decisions.md §45/§52, docs/prereg_2026-09-24.md):
for every seed with a greedy task in scope, a regen log for gpt2m_greedy seed{s}/gen0 showing
all stored rows prompt-prefixed, 512/512 prompts at 32 tokens, and pooled fraction >= 0.99.
Branch 2 (PASS-weak) is not auto-passed: it needs the per-row criterion adjudicated by hand.
The fp32 argmax-match rate is a reported diagnostic only (K3 retired, §44).
"""
import argparse, glob, json, os, re, sys

K3P_MIN = 0.99


def read_regen(path):
    txt = open(path).read()
    m = re.search(r"stored \S*/gpt2m_greedy/seed(\d+)/gen0/synthetic_corpus\.jsonl", txt)
    if not m:
        return None
    seed = int(m.group(1))
    pre = re.search(r"stored rows starting with reconstructed prompt: (\d+)/(\d+)", txt)
    pl = re.search(r"prompts re-encoding to \d+ tokens: (\d+)/(\d+)", txt)
    pooled = re.search(r"PRIMARY positions before first divergence \(pooled\): ([0-9.]+)", txt)
    probs = []
    if not (pre and pl and pooled):
        probs.append("incomplete log")
    else:
        if pre.group(1) != pre.group(2): probs.append(f"prompt prefix {pre.group(1)}/{pre.group(2)}")
        if pl.group(1) != pl.group(2): probs.append(f"prompt length {pl.group(1)}/{pl.group(2)}")
        if float(pooled.group(1)) < K3P_MIN: probs.append(f"pooled {pooled.group(1)} < {K3P_MIN} (branch 2/3: adjudicate)")
    return seed, (float(pooled.group(1)) if pooled else None), probs


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--manifest", default="slurm/step0_manifest.tsv")
    ap.add_argument("--out-dir", default=os.path.expandvars("$SCRATCH/collapse/step0_7p1/scores"))
    ap.add_argument("--k3prime-glob", default="logs/step0/regen-*.out")
    a = ap.parse_args()
    lines = open(a.manifest).read().splitlines()
    cols = lines[0].split("\t")
    tasks = [dict(zip(cols, l.split("\t"))) for l in lines[1:]]
    if a.pilot:
        tasks = tasks[:1]
    bad, shas, heads, excl, rates = [], set(), set(), {}, []
    if any(t.get("enforce_k3") == "1" for t in tasks):
        bad.append("manifest still enforces retired K3")
    greedy_seeds = set()
    for t in tasks:
        if "greedy" in t["task"]:
            m = re.search(r"_s(\d+)_", t["task"])
            if m: greedy_seeds.add(int(m.group(1)))
        js = os.path.join(a.out_dir, t["task"] + ".json")
        if not os.path.exists(js):
            bad.append(f"{t['task']}: no json"); continue
        m = json.load(open(js))
        shas.add(m.get("script_md5")); heads.add(m.get("git_head"))
        if m["verdict"].startswith("FAIL"):
            bad.append(f"{t['task']}: {m['verdict']}"); continue
        if m.get("argmax_match_rate") is not None:
            rates.append((m["argmax_match_rate"], t["task"]))
        if t["kind"] != "canary" and not os.path.exists(os.path.join(a.out_dir, t["task"] + ".npz")):
            bad.append(f"{t['task']}: no npz")
        if m["n_scored"] + m["n_excluded"] != m["n_candidates"]:
            bad.append(f"{t['task']}: count mismatch")
        if t["expect_synth"] != "0" and m.get("n_synth_rows") != int(t["expect_synth"]):
            bad.append(f"{t['task']}: synth rows {m.get('n_synth_rows')} != {t['expect_synth']}")
        excl.setdefault(t["kind"], []).append((m["n_excluded"], m["n_candidates"]))
        if a.pilot:
            print(json.dumps({k: m.get(k) for k in ("task", "verdict", "argmax_match_rate", "n_candidates",
                  "n_scored", "n_excluded", "excluded", "scorer", "scorer_weights_md5", "corpus_md5",
                  "git_head", "script_md5", "seconds", "device")}, indent=1))
    # K3': one passing regen log per greedy seed in scope (a failing log for a seed is never overridden)
    k3p = {}
    for p in sorted(glob.glob(a.k3prime_glob)):
        r = read_regen(p)
        if r is None:
            continue
        seed, pooled, probs = r
        k3p.setdefault(seed, []).append((os.path.basename(p), pooled, probs))
    for s in sorted(greedy_seeds):
        runs = k3p.get(s, [])
        if not runs:
            bad.append(f"K3' seed {s}: no regen log"); continue
        for name, pooled, probs in runs:
            print(f"K3' seed {s}: {name} pooled={pooled} {'PASS' if not probs else 'FAIL ' + '; '.join(probs)}")
            if probs:
                bad.append(f"K3' seed {s}: {name}: {'; '.join(probs)}")
    if len(shas) > 1 or len(heads) > 1:
        bad.append(f"not pinned: {len(shas)} script md5s, {len(heads)} git heads")
    for k, v in sorted(excl.items()):
        ne, nc = sum(x for x, _ in v), sum(y for _, y in v)
        print(f"exclusions {k:6s}: {ne}/{nc} ({100*ne/max(nc,1):.2f}%) over {len(v)} tasks; max per task {max(x for x,_ in v)}")
    if rates:
        r0 = min(rates)
        print(f"diagnostic (not a gate): fp32 argmax_match_rate min {r0[0]:.5f} ({r0[1]}) over {len(rates)} tasks")
    print(f"pinned: script_md5 {shas}  git_head {heads}")
    for b in bad[:50]:
        print("FAIL", b)
    print(f"GATE {'PASS' if not bad else 'FAIL'} ({len(tasks)} tasks, {len(bad)} problems)")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
