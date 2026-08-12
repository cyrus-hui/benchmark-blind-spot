#!/usr/bin/env python3
"""P0b: does the benchmark rise come from newly-correct hard items (capability)
   or from margin shifts on near-boundary items (sharpening)?
   Usage: python margin_analysis.py {smollm2_1.7b|pilot_gpt2m} <task> [seeds...]"""
import glob, json, os, sys
import numpy as np

def load(run, gen, seed, task):
    pat = (f"{os.environ['SCRATCH']}/collapse/{run}/seed{seed}/gen{gen}"
           f"/benchmarks_logsamples/**/samples_{task}_*.jsonl")
    fs = sorted(glob.glob(pat, recursive=True))
    if not fs: return None
    out = {}
    for line in open(fs[-1]):                      # newest file if re-run
        d = json.loads(line)
        lls = np.array([float(r[0]) for r in d["filtered_resps"]])
        blen = np.array([len(d["arguments"][f"gen_args_{i}"]["arg_1"].encode())
                         for i in range(len(lls))], dtype=float)
        norm = lls / blen                          # lm-eval acc_norm convention
        gold = int(d["target"])
        order = np.argsort(norm)[::-1]
        # margin: gold minus best competitor (positive = correct, by how much)
        best_other = np.max(np.delete(norm, gold))
        out[d["doc_id"]] = dict(
            correct=bool(d["acc_norm"]),
            margin=float(norm[gold] - best_other),
            top_gap=float(norm[order[0]] - norm[order[1]]),  # decision confidence
        )
    return out

def main(run, task, seeds):
    print(f"\n### {run} / {task}   seeds {seeds}")
    agg = {k: [] for k in ("cc", "ci", "ic", "ii")}
    for seed in seeds:
        a, b = load(run, 0, seed, task), load(run, 7, seed, task)
        if a is None or b is None:
            print(f"  seed {seed}: missing gen0 or gen7 samples -- skipped"); continue
        ids = sorted(set(a) & set(b))
        cells = {k: [] for k in agg}
        for i in ids:
            k = ("c" if a[i]["correct"] else "i") + ("c" if b[i]["correct"] else "i")
            cells[k].append(i)
        n = len(ids)
        net = (len(cells["ic"]) - len(cells["ci"])) / n
        print(f"\n  seed {seed}  n={n}  acc {sum(x['correct'] for x in a.values())/n:.4f}"
              f" -> {sum(x['correct'] for x in b.values())/n:.4f}   net {net*100:+.2f}pp")
        print(f"    stable-correct {len(cells['cc']):5d} | "
              f"c->i {len(cells['ci']):4d} | i->c {len(cells['ic']):4d} | "
              f"stable-wrong {len(cells['ii']):5d}")
        # Were the flipped-to-correct items already near the boundary at gen 0?
        for k, lbl in (("ic", "i->c (gained)"), ("ci", "c->i (lost)"),
                       ("ii", "stayed wrong"), ("cc", "stayed right")):
            if not cells[k]: continue
            m0 = np.array([a[i]["margin"] for i in cells[k]])
            print(f"    {lbl:14s} gen0 margin: median {np.median(m0):+.4f}  "
                  f"|m|<0.01: {100*np.mean(np.abs(m0) < 0.01):.1f}%")
            agg[k].append(np.median(m0))
        d0 = np.array([a[i]["top_gap"] for i in ids])
        d7 = np.array([b[i]["top_gap"] for i in ids])
        print(f"    decision gap (top1-top2): {d0.mean():.4f} -> {d7.mean():.4f} "
              f"({100*(d7.mean()-d0.mean())/d0.mean():+.1f}%)")
        q = np.quantile(np.abs([a[i]["margin"] for i in ids]), [0.01, 0.05, 0.25])
        print(f"    |margin| quantiles 1/5/25%: {q[0]:.5f} / {q[1]:.5f} / {q[2]:.5f}"
              f"   (bf16 floor ~0.25/blen ~ 0.006)")

if __name__ == "__main__":
    run = sys.argv[1] if len(sys.argv) > 1 else "smollm2_1.7b"
    task = sys.argv[2] if len(sys.argv) > 2 else "hellaswag"
    seeds = [int(x) for x in sys.argv[3:]] or [43, 44]
    main(run, task, seeds)
