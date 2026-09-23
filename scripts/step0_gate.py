#!/usr/bin/env python
"""Gate for 7.1 step 0 outputs. Reads canary/meta JSON only; prints NO statistic of S.
  --pilot : task 1 only.  --all : every manifest task.
PASS requires: json present; verdict PASS on every K3-enforced task (rate >= 0.99);
npz present for synth/null; n_scored + n_excluded == candidates; one script_md5 and
one git_head across all tasks; S finite (checked by the scorer).
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pilot", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--manifest", default="slurm/step0_manifest.tsv")
    ap.add_argument("--out-dir", default=os.path.expandvars("$SCRATCH/collapse/step0_7p1/scores"))
    a = ap.parse_args()
    lines = open(a.manifest).read().splitlines()
    cols = lines[0].split("\t")
    tasks = [dict(zip(cols, l.split("\t"))) for l in lines[1:]]
    if a.pilot:
        tasks = tasks[:1]
    bad, shas, heads, excl, rates = [], set(), set(), {}, []
    for t in tasks:
        js = os.path.join(a.out_dir, t["task"] + ".json")
        if not os.path.exists(js):
            bad.append(f"{t['task']}: no json"); continue
        m = json.load(open(js))
        shas.add(m.get("script_md5")); heads.add(m.get("git_head"))
        if m["verdict"].startswith("FAIL"):
            bad.append(f"{t['task']}: {m['verdict']} rate={m['argmax_match_rate']:.5f}"); continue
        if t["enforce_k3"] == "1":
            rates.append((m["argmax_match_rate"], t["task"]))
            if m["verdict"] != "PASS":
                bad.append(f"{t['task']}: enforced but verdict {m['verdict']}")
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
    if len(shas) > 1 or len(heads) > 1:
        bad.append(f"not pinned: {len(shas)} script md5s, {len(heads)} git heads")
    for k, v in sorted(excl.items()):
        ne, nc = sum(x for x, _ in v), sum(y for _, y in v)
        print(f"exclusions {k:6s}: {ne}/{nc} ({100*ne/max(nc,1):.2f}%) over {len(v)} tasks; max per task {max(x for x,_ in v)}")
    if rates:
        r0 = min(rates)
        print(f"K3 enforced tasks: {len(rates)}; min argmax_match_rate {r0[0]:.5f} ({r0[1]})")
    print(f"pinned: script_md5 {shas}  git_head {heads}")
    for b in bad[:50]:
        print("FAIL", b)
    print(f"GATE {'PASS' if not bad else 'FAIL'} ({len(tasks)} tasks, {len(bad)} problems)")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
