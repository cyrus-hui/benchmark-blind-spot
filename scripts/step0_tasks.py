#!/usr/bin/env python
"""Build slurm/step0_manifest.tsv for 7.1 step 0 (decisions.md §41). No GPU, no scores.

Corpus training gen g (g = 1..4):
  pure trees: <tree>/seed{s}/gen{g-1}/synthetic_corpus.jsonl
  mix trees : <tree>/mixed/seed{s}/gen{g}.jsonl, synthetic rows only
              (real ref = fresh_slices_gpt2m/gen{g}_real.jsonl, §40.4-5)
Scorers: own = pilot_gpt2m/seed{s}/gen0/model; cross = CROSS[s] (fixed below).
Nulls: fresh_slices_gpt2m/gen{g}_real.jsonl under every scorer seed used.
K3 enforced where the scorer IS the generator on a greedy corpus:
  own-scorer tasks at g = 1, plus 'gen' canary tasks at g >= 2 scored by gen{g-1}.
Task 1 is the pilot: gpt2m_greedy_s43_c1_own.
"""
import argparse, hashlib, os, sys

TREES = {"gpt2m_greedy": ("greedy", "pure"), "pilot_gpt2m": ("topk", "pure"),
         "mix50_greedy": ("greedy", "mix"), "mix50_topk": ("topk", "mix")}
PURE_OF = {"greedy": "gpt2m_greedy", "topk": "pilot_gpt2m"}
SEEDS = [43, 44, 45, 46]
CROSS = {43: 44, 44: 45, 45: 46, 46: 43}
GENS = [1, 2, 3, 4]
N_ROWS = 18686
MIX_R = {"mix50_greedy": 0.50, "mix50_topk": 0.50}  # synthetic fraction; 0.5 * 18686 = 9343
COLS = ["task", "kind", "corpus", "scorer", "real_ref", "synth_ref", "expect_synth", "expect_rows", "enforce_k3"]
PILOT = "gpt2m_greedy_s43_c1_own"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--out", default="slurm/step0_manifest.tsv")
    ap.add_argument("--n-rows", type=int, default=N_ROWS, help="self-test only")
    a = ap.parse_args()
    R = a.root
    P = lambda *x: os.path.join(R, *x)
    scorer = lambda s: P("pilot_gpt2m", f"seed{s}", "gen0", "model")
    fresh = lambda g: P("fresh_slices_gpt2m", f"gen{g}_real.jsonl")

    rows, need, used_scorers, inv = [], set(), set(), {}
    for tree, (dec, kind) in TREES.items():
        seeds = [s for s in SEEDS if os.path.isdir(P(tree, f"seed{s}"))]
        inv[tree] = seeds
        for s in seeds:
            for g in GENS:
                if kind == "pure":
                    corpus, rref, sref, es = P(tree, f"seed{s}", f"gen{g-1}", "synthetic_corpus.jsonl"), "-", "-", "0"
                else:
                    corpus, rref, es = P(tree, "mixed", f"seed{s}", f"gen{g}.jsonl"), fresh(g), str(round(MIX_R[tree] * a.n_rows))
                    sref = P(tree, f"seed{s}", f"gen{g-1}", "synthetic_corpus.jsonl")
                    if g == 1 and not os.path.exists(sref):
                        sref = P(PURE_OF[dec], f"seed{s}", "gen0", "synthetic_corpus.jsonl")
                    need.update([rref, sref])
                need.add(corpus)
                for role, sc in (("own", s), ("cross", CROSS[s])):
                    used_scorers.add(sc)
                    k3 = "0"  # K3 retired (decisions.md §44); K3prime checked by step0_gate.py
                    rows.append([f"{tree}_s{s}_c{g}_{role}", "synth", corpus, scorer(sc), rref, sref, es, str(a.n_rows), k3])
                if dec == "greedy" and g >= 2:
                    gm = P(tree, f"seed{s}", f"gen{g-1}", "model")
                    need.add(gm)
                    rows.append([f"{tree}_s{s}_c{g}_gen", "canary", corpus, gm, rref, sref, es, str(a.n_rows), "1"])
    for g in GENS:
        for sc in sorted(used_scorers):
            need.add(fresh(g))
            rows.append([f"null_c{g}_sc{sc}", "null", fresh(g), scorer(sc), "-", "-", "0", str(a.n_rows), "0"])
    need.update(scorer(sc) for sc in used_scorers)

    missing = sorted(p for p in need if not os.path.exists(p))
    print("inventory (seeds present):", {t: v for t, v in inv.items()})
    if missing:
        print(f"REFUSE: {len(missing)} required paths missing:")
        for p in missing[:40]:
            print("  ", p)
        sys.exit(2)
    names = [r[0] for r in rows]
    assert len(set(names)) == len(names), "duplicate task names"
    if PILOT not in names:
        sys.exit(f"REFUSE: pilot {PILOT} not in task list")
    rows.sort(key=lambda r: r[0] != PILOT)  # pilot first, order otherwise preserved
    text = "\t".join(COLS) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    if os.path.exists(a.out) and open(a.out).read() != text:
        sys.exit(f"REFUSE: {a.out} exists with different content; move it aside first")
    with open(a.out + ".tmp", "w") as f:
        f.write(text)
    os.replace(a.out + ".tmp", a.out)
    kinds = {}
    for r in rows:
        kinds[r[1]] = kinds.get(r[1], 0) + 1
    print(f"tasks={len(rows)} {kinds} k3_enforced={sum(r[8] == '1' for r in rows)} "
          f"scorer_seeds={sorted(used_scorers)} cross={CROSS}")
    print(f"task 1 = {rows[0][0]}")
    print(f"manifest {a.out} md5 {hashlib.md5(text.encode()).hexdigest()}  -> sbatch --array=1-{len(rows)}")


if __name__ == "__main__":
    main()
