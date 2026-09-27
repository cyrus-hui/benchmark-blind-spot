#!/usr/bin/env python3
r"""7.1 filter experiment: the registered read. Written blind, before any filtered corpus or chain exists.

REGISTRATION. This file is the registration of the read (§62.9.1): committed and pushed with the filter code before
the pilot, run once unmodified after all 36 chains are complete and gated, its log committed unedited.

  --check     validate every input and the canaries; print counts only, NO value.
  (default)   canaries first (stop on any failure), then 62.6, then 62.7 (a)-(g), in that order.
  --rehearse  mock tree only (refuses the real collapse root; skips the push check).

Refuses unless: this script, src/filter_corpus.py, src/step0_score.py and the registry are committed and unmodified;
HEAD is on origin; the registry md5 is 62d12548...; logs/filt71/gate_all.json is PASS on 252 cells; every cell
carries the scoring-pass keys.

Transcribed verbatim from decisions.md (project notes, not in git): §62, then §64.

### §62 — 7.1 filter experiment: PRE-REGISTRATION (written before any filter code, filtered corpus or filtered chain exists)

**62.1 Question.** When recursive training is at r = 100, does removing the most mode-following fraction of each
generation's synthetic corpus slow collapse, compared with removing the same number of documents at random?

**62.2 Score.** S(d) exactly as §41: over the 96 generated positions, mean of −log q_ref(x_t) − H(q_ref), where q_ref is
the chain's **own-seed gen-0 checkpoint** at every generation. The scorer is the pinned step-0 M2 implementation, not a
reimplementation. Rows that do not re-tokenize to 128 are **unscorable**. They are kept in every arm and never counted in k.

**62.3 Arms.** For generation g's training corpus, the full synthetic corpus written by model g − 1 (18,686 rows), with n
scorable rows:
- **F(q), filter:** drop the k = round(q·n) scorable rows with the lowest S. Ties are broken by row index.
- **R(q), random:** drop k scorable rows chosen by `Random(f"7.1|R|{decoder}|{seed}|{gen}|{q}")`. The same k as F(q) and
  the same unscorable set.
- **U, unfiltered (reference, not a control):** the existing `gpt2m_greedy` / `pilot_gpt2m` chains, reused. It trains on
  18,686 rows, so it is not count-matched and never enters the primary comparison.
- **No label-oracle arm.** At r = 100 every row is synthetic.
- **No upper-tail arm.** The direction is registered and was confirmed in §60.3.

**62.4 Matching.** F(q) and R(q) train on exactly the same number of rows, and every row is 128 tokens (32 prompt + 96
generated), so **token volume is matched exactly**. Bytes are reported, not matched. The step count is 3·ceil(kept/32) in
both arms, checked in every log. **Prompt lineage:** generation always draws its prompts from the **full, unfiltered**
synthetic corpus of the previous generation (the driver's default corpus mode, not `--prompt-corpus`). So every arm and
generation generates for the same 18,686 propagated prompts, and the filter acts on the training set only. This departs
from 5.3's "prompts from the corpus the generation trained on". The reason is that filtering the prompt source would shrink
the prompt set by (1 − q)^g and confound the filter with prompt loss.

**62.5 Cells.** Decoders greedy and top-k; r = 100; q ∈ {0.10, 0.25, 0.50}; seeds 43–45 (n = 3, uniform with the grid);
gens 1–7; arms F and R. That makes **36 chains and 252 training cells**. Gen 1 trains on the existing gen-0 corpora, which
are identical across arms before filtering. Gen-0 models are shared.

**62.6 Primary endpoint. It alone decides the verdict.** Cell: **greedy, q = 0.25.**
- **Unit:** the seed (chain pair). F and R share the seed's gen-0 model and prompts.
- **Statistic:** D_s = Distinct-2_F(7) − Distinct-2_R(7), on the eval completions (the non-Δ instrument; eval prompts and
  decoder are unchanged). The gen-0 value is shared, so this equals the difference of the shifts from gen 0.
- **Sidedness:** one-sided, with positive meaning the filter preserves diversity. Harm is read as its own branch.
- **Threshold:** τ_D, greedy = **0.0137** (registered Sept 21). It is the largest |shift| across real-data control chains,
  which is exactly the chain-to-chain noise scale that separates two different chains at one generation.
- **Verdict:**
  - **WORKS** iff D_s > 0 in 3/3 seeds and mean D_s ≥ 0.0137.
  - **HARMS** iff D_s < 0 in 3/3 and mean D_s ≤ −0.0137.
  - **NO EFFECT** otherwise. That falsifies the filter claim and is reported as such.
- **Power, stated in advance.** Greedy r = 100 Distinct-2 falls about 0.088 from gen 0 to gen 7 (0.1924 × 45.9%). WORKS
  therefore needs F to hold back roughly 16% of the unfiltered fall, net of R's. 3/3 signs is a decision rule, not a
  significance test (one-sided sign p = 0.125).

**62.7 Secondary, reported whatever they show, never overriding 62.6.**
- (a) The same rule in the other five cells: greedy q = 0.10 and 0.50, and top-k q ∈ {0.10, 0.25, 0.50} with τ_D, top-k =
  0.0105. **Forecast on record:** under top-k the whole r = 100 fall is about 0.0164, only 1.6× τ_D, so a top-k WORKS
  would need F to prevent about 64% of the collapse. Top-k is expected to read NO EFFECT on the Distinct-2 rule whatever the
  mechanism.
- (b) KL-unigram mirror: D^K_s = KL_R(7) − KL_F(7) against τ_K (greedy 0.1775, top-k 0.1111), with the same three branches.
- (c) Outcome-label status at gen 7 for every chain, and the first generation at which each chain meets the label.
- (d) Dose-response across q: descriptive, and not a tested outcome.
- (e) OOD perplexity (c4, ccnews, wiki), the driver's benchmark battery, and MAUVE (reported, as at 345M).
- (f) Δ_cal(g) for every chain. **Reported with a circularity flag:** S and Δ_cal share a construction, so Δ_cal is never
  evidence that the filter works.
- (g) R(q) versus U: the effect of removing a fraction q of the corpus at random, which is the cost of any filter at that q.

**62.8 Pipeline canaries (not findings). Each refuses at n = 0.**
- **K-F1:** the filter's S on the gen-0 corpora reproduces the step-0 own-scorer npz values for every scored row
  (max |dS| ≤ 1e-5), which ties the filter to the registered scorer.
- **K-F2:** counts.
  - kept = n_total − k in F and R alike.
  - Identical unscorable sets.
  - Prompt source = 18,686 rows at every generation.
  - Logged step count = 3·ceil(kept/32).
- **K-F3:** selection.
  - Every row F drops has S ≤ every scorable row F keeps.
  - R's dropped-row mean S lies within 4 standard errors of the scorable mean. The inverse, F's dropped rows against the
    same band, must fire.
- **K-F4:** additivity. q = 0 through the driver hook reproduces the unfiltered training corpus byte-identically (md5), and
  the unfiltered config path is unchanged.
- **Pilot:** greedy F(0.25) s43 gen 1 passes K-F1…K-F4 and `check_cell.sh`-equivalent gates before any chain is submitted.

**62.9 Blindness and order.**
1. This section is transcribed verbatim into the docstring of the 7.1 analysis script. That script is committed **and
   pushed** together with the filter code before the pilot runs. This read decides a registered verdict (§61.6).
2. While the chains run, only gate quantities are opened: key census, counts, step counts and canaries. No Distinct-2, KL,
   MAUVE or benchmark value from any F or R chain is opened before the analysis script's pinned read.
3. The read runs once, after all 36 chains are complete and gated. Its log is committed unedited and interpreted in the
   order 62.6 → 62.7.

**62.10 What each outcome licenses.**
- **WORKS:** "at r = 100 under greedy decoding at 345M, removing the quarter of each generation's synthetic corpus with the
  lowest per-document gap preserves eval-completion diversity better than random removal at matched token volume".
  Scoped to the cell. The secondary cells say how far it extends.
- **NO EFFECT:** a fourth registered falsification. Δ_cal remains a validated monitor (Phase 5), not a filter.
- **HARMS:** reported. It would mean the lowest-S documents carry diversity the chain needs.

**62.11 Cost.** A 345M generation cell took about 15 minutes (grid: 126 cells for 31 GPU-h, max 17:05). Filtered cells
train on fewer steps but add an S-scoring pass. Estimate: about 0.25–0.30 GPU-h per cell × 252 ≈ **65–75 GPU-h**, 2–2.5× the
severity grid. The chains are `afterok` with depth 7, like the grid (not parallel). Expect several days of wall time at
fair-share ≈ 0.16.

### §64 — 7.1 build: clarifications registered with the code, before it is committed or run (Sept 26 2026)

Writing the filter against §62 exposed gaps in its text. Each is closed here before any filtered corpus, filtered
chain or statistic exists. The code at the commit that carries this section implements exactly this text, and §62 plus
this section form the docstring of `scripts/filt71_analysis.py`.

**64.1 R(q)'s k after generation 1.** §62.3 says R drops "the same k as F(q)", and §62.4 says both arms train on exactly
the same number of rows. From gen 2 on, the F and R chains hold different corpora, so their scorable counts n differ and
round(q·n) would differ too. **Resolution:** R(q) at (decoder, q, seed, gen) takes k from the paired F(q) manifest row
for the same cell. Then kept = 18,686 − k is identical in both arms at every generation. R(g) therefore runs `afterok`
on F(g) as well as on R(g − 1), and it refuses if F's row is missing. The R rows are still drawn only from R's own
scorable rows.

**64.2 "Identical unscorable sets"** (§62.3, K-F2) can hold only at gen 1, the one generation where F and R filter the
same corpus. It is checked there: R's own k must equal F's, and the md5 of the unscorable index set must match. At gens
2–7 each arm's unscorable count is reported, not matched.

**64.3 Rounding and spellings.**
- k = floor(q·n + 1/2), rounding half up. Python's `round` rounds half to even, so it is not used.
- The R rng string uses Python `str()` of q as YAML parses it (`0.1`, `0.25`, `0.5`) and the decoder names `greedy` /
  `topk`.
- The R draw is `Random(...).sample(sorted scorable row indices, k)`.
- The filtered corpus is the input's lines, byte-for-byte and in their original order, minus the dropped rows.

**64.4 K-F3's standard error.** SE = sd(S over the scorable rows, ddof = 1) / √k × √((n − k)/(n − 1)), the standard
error of a mean of k draws without replacement. Both halves of K-F3 (R inside ±4 SE, F outside) run in both arms. Only
|z| values are printed, never a statistic of S.

**64.5 Where K-F1 and K-F4 run.**
- K-F1 runs at gen 1 only, because only the gen-0 corpora were scored by step 0 with this scorer.
- K-F4's byte identity (a k = 0 rendering through the one writer must reproduce the input md5) is checked in-process at
  every cell.
- K-F4's no-op half is checked by `prepare_7p1.py`: it runs the driver hook's own parser on every existing config and
  requires the no-filter sentinel. It also parses every generated config before writing anything.

**64.6 Hardware.** All 36 chains run on the GPU request used for the pilot (`GRES`, recorded per job in
`logs/filt71/submitted.txt`). If that is a MIG slice, the §62.7(g) R-versus-U comparison crosses hardware, because U ran
on full A100s, and it is reported with that statement. The primary (F vs R) is unaffected.

**64.7 Order is enforced by machine.**
- The driver refuses unless `src/filter_corpus.py` and `src/step0_score.py` are committed and unmodified.
- The submit script refuses unless HEAD is on origin. For the chains it also requires `logs/filt71/gate_pilot.json` to
  read PASS.
- The analysis refuses unless `logs/filt71/gate_all.json` reads PASS on 252 cells, every cell carries the scoring-pass
  keys, the registry md5 is `62d1254829994ab8b2f6dbb6b2edd047`, and the analysis, filter and scorer are committed,
  unmodified and pushed.

**64.8 Walltime** comes from the pilot's observed elapsed time. The submit script has no default outside the pilot.

**64.9 Scoring pass.** After the chains, the 252 lines in `filt71_ood_lines.txt` are appended to
`slurm/ood_manifest.txt` and scored by `score_ood_ppl_array.sbatch`. That supplies `entropy` and `perplexity_ood_*` for
§62.7(e)–(f). Running this pass opens no value.
"""
import argparse, glob, hashlib, json, math, os, subprocess, sys, time

REG_MD5 = "62d1254829994ab8b2f6dbb6b2edd047"
TAU = {("greedy", "D"): 0.0137, ("topk", "D"): 0.0105, ("greedy", "K"): 0.1775, ("topk", "K"): 0.1111}
QS = {10: 0.10, 25: 0.25, 50: 0.50}
DECS = ("greedy", "topk")
U = {"greedy": "gpt2m_greedy", "topk": "pilot_gpt2m"}
SEEDS, GENS = (43, 44, 45), range(1, 8)
PRIMARY = ("greedy", 25)
KEYS = ("distinct2", "kl_unigram", "mauve", "perplexity", "entropy",
        "perplexity_ood_c4", "perplexity_ood_ccnews", "perplexity_ood_wiki")
PINNED = ("scripts/filt71_analysis.py", "src/filter_corpus.py", "src/step0_score.py", "registered_thresholds.json")
FORBID = ("_logsamples", "_temp")


def die(msg):
    print(f"STOP: {msg}"); sys.exit(2)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True)


def pins(rehearse):
    for f in PINNED:
        if git("ls-files", "--error-unmatch", f).returncode or git("diff", "--quiet", "HEAD", "--", f).returncode:
            die(f"{f} is not committed and unmodified")
    if md5("registered_thresholds.json") != REG_MD5:
        die("registry md5 is not the registered 62d12548...")
    if not rehearse and "origin/" not in git("branch", "-r", "--contains", "HEAD").stdout:
        die("HEAD is not on origin: push before the read (§62.9.1)")


def cell(root, arm, qi, dec, s, g):
    return os.path.join(root, f"filt71_{arm}{qi}_{dec}", f"seed{s}", f"gen{g}")


def load(path):
    p = os.path.join(path, "metrics.json")
    if not os.path.isfile(p):
        die(f"missing {p}")
    m = json.load(open(p))
    miss = [k for k in KEYS if k not in m]
    if miss:
        die(f"{p} lacks {miss} (run the §64.9 scoring pass first)")
    return m


def bench(path):
    hits = [h for h in glob.glob(os.path.join(path, "benchmarks", "**", "results*.json"), recursive=True)
            if not any(x in h for x in FORBID)]
    if not hits:
        die(f"no benchmark results under {path}/benchmarks")
    res = json.load(open(sorted(hits)[-1])).get("results", {})
    try:
        return {"hellaswag_acc_norm": res["hellaswag"]["acc_norm,none"],
                "lambada_acc": res["lambada_openai"]["acc,none"],
                "lambada_ppl": res["lambada_openai"]["perplexity,none"]}
    except KeyError as e:
        die(f"benchmark key {e} missing in {sorted(hits)[-1]}")


def verdict(D, tau):
    mu = sum(D) / len(D)
    if all(d > 0 for d in D) and mu >= tau:
        return "WORKS", mu
    if all(d < 0 for d in D) and mu <= -tau:
        return "HARMS", mu
    return "NO EFFECT", mu


def fmt(D):
    return " ".join(f"{d:+.4f}" for d in D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.environ.get("SCRATCH", ""), "collapse"))
    ap.add_argument("--check", action="store_true", help="validate inputs, print counts only")
    ap.add_argument("--rehearse", action="store_true", help="mock tree only: skips the push check")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    real = os.path.realpath(os.path.join(os.environ.get("SCRATCH", "/nonexistent"), "collapse"))
    if a.rehearse:
        if os.path.realpath(a.root) == real:
            die("--rehearse refuses the real collapse root")
        print("=" * 70 + "\nREHEARSAL ON A MOCK TREE — NOT A READ\n" + "=" * 70)
    pins(a.rehearse)
    gate = "logs/filt71/gate_all.json"
    if not os.path.isfile(gate):
        die(f"{gate} missing: run scripts/filt71_gate.py --all")
    gj = json.load(open(gate))
    if gj.get("verdict") != "PASS" or gj.get("complete") != 252:
        die(f"{gate} is {gj.get('verdict')} with {gj.get('complete')}/252 complete")
    reg = json.load(open("registered_thresholds.json"))["outcome"]
    for (d, kind), v in TAU.items():
        if reg[d][f"tau_{kind}"] != v:
            die(f"registry outcome.{d}.tau_{kind} = {reg[d]['tau_' + kind]} != registered {v}")

    # ---- load everything, and run the canaries, before printing any value ----
    M, B = {}, {}
    n = 0
    for dec in DECS:
        for s in SEEDS:
            u0 = os.path.join(a.root, U[dec], f"seed{s}", "gen0")
            M[("U", dec, s, 0)] = load(u0)
            M[("U", dec, s, 7)] = load(os.path.join(a.root, U[dec], f"seed{s}", "gen7"))
            for qi in QS:
                for arm in ("F", "R"):
                    g0 = cell(a.root, arm, qi, dec, s, 0)
                    if os.path.realpath(g0) != os.path.realpath(u0):
                        die(f"C0: {g0} does not resolve to {u0}")
                    for g in GENS:
                        M[(arm, qi, dec, s, g)] = load(cell(a.root, arm, qi, dec, s, g)); n += 1
                    B[(arm, qi, dec, s)] = bench(cell(a.root, arm, qi, dec, s, 7))
                # C2: the paired rows trained on identical row counts at every gen (§64.1)
                for g in GENS:
                    ks = []
                    for arm in ("F", "R"):
                        mf = os.path.join(a.root, f"filt71_{arm}{qi}_{dec}", "filter_manifest.tsv")
                        L = open(mf).read().splitlines(); h = L[0].split("\t")
                        rows = [dict(zip(h, l.split("\t"))) for l in L[1:]]
                        ks.append({r["k"] for r in rows if r["seed"] == str(s) and r["gen"] == str(g)})
                    if len(ks[0]) != 1 or ks[0] != ks[1]:
                        die(f"C2: k differs F {ks[0]} vs R {ks[1]} at q={qi} {dec} s{s} g{g}")
    if n != 252:
        die(f"C1: loaded {n} cells, expected 252")
    print(f"canaries: C0 gen-0 symlinks 36/36, C1 cells {n}/252, C2 matched k 126/126, registry {REG_MD5[:8]}, "
          f"gate {gj['verdict']} {gj['complete']}/252 @ {gj['head']}")
    if a.check:
        print("--check: inputs valid. No value printed.")
        return
    out = {"primary": {}, "secondary": {}}

    def Ds(dec, qi, key):
        if key == "distinct2":   # positive = the filter preserves diversity
            return [M[("F", qi, dec, s, 7)][key] - M[("R", qi, dec, s, 7)][key] for s in SEEDS]
        return [M[("R", qi, dec, s, 7)][key] - M[("F", qi, dec, s, 7)][key] for s in SEEDS]  # KL mirror

    # ---- 62.6 PRIMARY: decides the verdict ----
    dec, qi = PRIMARY
    D = Ds(dec, qi, "distinct2")
    v, mu = verdict(D, TAU[(dec, "D")])
    print("\n" + "=" * 70)
    print(f"62.6 PRIMARY  greedy q=0.25  D_s = Distinct-2_F(7) - Distinct-2_R(7), seeds {SEEDS}")
    print(f"  D_s = {fmt(D)}   mean {mu:+.4f}   tau_D greedy {TAU[(dec, 'D')]}")
    print(f"  VERDICT: {v}")
    print("=" * 70)
    out["primary"] = {"D_s": D, "mean": mu, "verdict": v}

    # ---- 62.7 SECONDARY: reported whatever they show, never overriding 62.6 ----
    print("\n62.7(a) the same rule in the other five cells (Distinct-2)")
    print("   (forecast on record: top-k reads NO EFFECT; its r=100 fall is ~1.6x tau_D)")
    for d in DECS:
        for q in QS:
            if (d, q) == PRIMARY:
                continue
            D = Ds(d, q, "distinct2"); vv, m = verdict(D, TAU[(d, "D")])
            print(f"  {d:6s} q={QS[q]:.2f}  D_s {fmt(D)}  mean {m:+.4f}  tau {TAU[(d, 'D')]}  -> {vv}")
            out["secondary"][f"a_{d}_{q}"] = {"D_s": D, "mean": m, "verdict": vv}
    print("\n62.7(b) KL-unigram mirror  D^K_s = KL_R(7) - KL_F(7)")
    for d in DECS:
        for q in QS:
            D = Ds(d, q, "kl_unigram"); vv, m = verdict(D, TAU[(d, "K")])
            print(f"  {d:6s} q={QS[q]:.2f}  D^K {fmt(D)}  mean {m:+.4f}  tau {TAU[(d, 'K')]}  -> {vv}")
            out["secondary"][f"b_{d}_{q}"] = {"D_s": D, "mean": m, "verdict": vv}

    print("\n62.7(c) outcome label per chain (Distinct-2 fall > tau_D AND KL rise > tau_K vs own gen 0)")
    print("   status at gen 7; first gen meeting the label (8 = never)")
    for d in DECS:
        tD, tK = TAU[(d, "D")], TAU[(d, "K")]
        for q in QS:
            for arm in ("F", "R"):
                row = []
                for s in SEEDS:
                    m0 = M[("U", d, s, 0)]
                    hit = [g for g in GENS if m0["distinct2"] - M[(arm, q, d, s, g)]["distinct2"] > tD
                           and M[(arm, q, d, s, g)]["kl_unigram"] - m0["kl_unigram"] > tK]
                    row.append(f"s{s}:{'C' if 7 in hit else '-'}/{min(hit) if hit else 8}")
                print(f"  {d:6s} q={QS[q]:.2f} {arm}  " + "  ".join(row))

    print("\n62.7(d) dose-response across q (descriptive, not a tested outcome): mean D_s / mean D^K")
    for d in DECS:
        print(f"  {d:6s} " + "   ".join(
            f"q={QS[q]:.2f}: {sum(Ds(d, q, 'distinct2')) / 3:+.4f} / {sum(Ds(d, q, 'kl_unigram')) / 3:+.4f}" for q in QS))

    print("\n62.7(e) gen 7, F - R per seed (mean): OOD perplexity, MAUVE, benchmarks")
    ek = ("perplexity_ood_c4", "perplexity_ood_ccnews", "perplexity_ood_wiki", "mauve")
    bk = ("hellaswag_acc_norm", "lambada_acc", "lambada_ppl")
    for d in DECS:
        for q in QS:
            parts = []
            for k in ek:
                x = [M[("F", q, d, s, 7)][k] - M[("R", q, d, s, 7)][k] for s in SEEDS]
                parts.append(f"{k.replace('perplexity_', '')} {sum(x) / 3:+.3f}")
            for k in bk:
                x = [B[("F", q, d, s)][k] - B[("R", q, d, s)][k] for s in SEEDS]
                parts.append(f"{k} {sum(x) / 3:+.4f}")
            print(f"  {d:6s} q={QS[q]:.2f}  " + "  ".join(parts))

    print("\n62.7(f) Delta_cal(g) = ln PPL - H, mean over seeds, gens 0..7")
    print("   CIRCULARITY FLAG: S and Delta_cal share a construction; this is never evidence that the filter works.")
    dc = lambda m: math.log(m["perplexity"]) - m["entropy"]
    for d in DECS:
        for q in QS:
            for arm in ("F", "R"):
                tr = [sum(dc(M[("U", d, s, 0)]) for s in SEEDS) / 3] + \
                     [sum(dc(M[(arm, q, d, s, g)]) for s in SEEDS) / 3 for g in GENS]
                print(f"  {d:6s} q={QS[q]:.2f} {arm}  " + " ".join(f"{x:+.3f}" for x in tr))

    print("\n62.7(g) R(q) - U at gen 7 (the cost of removing a fraction q at random)")
    print("   (§64.6: if the chains ran on a MIG slice, this comparison crosses hardware)")
    for d in DECS:
        for q in QS:
            x = [M[("R", q, d, s, 7)]["distinct2"] - M[("U", d, s, 7)]["distinct2"] for s in SEEDS]
            y = [M[("R", q, d, s, 7)]["kl_unigram"] - M[("U", d, s, 7)]["kl_unigram"] for s in SEEDS]
            print(f"  {d:6s} q={QS[q]:.2f}  dDistinct-2 {fmt(x)}   dKL {fmt(y)}")

    out["meta"] = {"head": git("rev-parse", "HEAD").stdout.strip(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "script_md5": md5(os.path.abspath(__file__)), "registry_md5": REG_MD5, "rehearse": a.rehearse}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    print(f"\nRESULT OK  head {out['meta']['head'][:7]}  script {out['meta']['script_md5']}")


if __name__ == "__main__":
    main()
