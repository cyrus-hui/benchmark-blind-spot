#!/usr/bin/env python
"""7.1 step 0 analysis. Written blind, before any statistic of S was read (Sept 25 2026).

REGISTRATION. This file is the registration of the step-0 read: committed and pushed before
the read, run once unmodified. Rules (project notes: decisions.md §41 as completed by §56):
 1 K1 cell: corpus training gen 1, r = 100, own-seed gen-0 scorer. Per decoder, synthetic docs
   pooled over seeds, each vs its own scorer's null tau (10th/90th pct of null_c1_sc{s}), strict
   tails. K1 fires for a decoder if both pooled tails' 95% CIs contain 0.10. Both decoders fire ->
   filter branch falsified; one -> 7.1 continues for the other decoder only (disclosed).
 2 A greedy own-scorer pass is expected by construction (self-scored greedy is decoder-truncated).
   Informative: top-k own, and both decoders under the cross-seed scorer (run in full).
 3 K1-material (secondary, never overrides K1): both tails' CIs inside [0.075, 0.125] -> immaterial.
 4 Exclusion bounds: K1 recomputed with every excluded synthetic row in the lower tail, then the
   upper; the verdict is exclusion-robust only if unchanged under both.
 5 K2: per-corpus Spearman(own, cross) on the 7 gen-1 r=100 corpora; fires if median < 0.30.
   Min reported; null-slice Spearman under the same scorer pairs as a human-text reference.
 6 Bootstrap: synth and null resampled jointly within seed, tau re-estimated per replicate,
   counts pooled over seeds; B = 10000; 95% percentile CI; numpy 'linear'; rng from 20260925+cell.
 7 Direction: contradicted for a decoder if pooled upper rate > lower rate (own, K1 cell),
   evaluated only where K1 does not fire; any contradiction -> registered two-arm 7.1.
 8 Canary C1 before any statistic: real rows of the mixed corpora reproduce the null task's S
   for the same text and scorer within 1e-3 nats; else STOP, nothing reported.
 9 M3 (descriptive): K1 cells; drop each tail vs 200 random drops of equal count; retained
   Distinct-2 and unigram KL(null || retained), add-0.5 over the GPT-2 vocab, generated spans.
   Prediction: dropping the lower tail raises D2 and lowers KL more than random.
10 Rank stability (prompt_hash-matched, own scorer, pure chains) and per-task enrichment: descriptive.

  --check : validate every input and print counts only. Prints NO statistic of S.
  (default): canaries first (stop on failure), then the read.
Refuses to run unless this script, the manifest and the scorer are committed and unmodified.
"""
import argparse, hashlib, json, os, re, subprocess, sys, zlib
from collections import defaultdict

import numpy as np

BOOT_SEED = 20260925
B_DEFAULT = 10000
CHUNK = 250
NULL_RATE = 0.10
K2_MIN = 0.30
MAT = (0.075, 0.125)  # K1-material (secondary, §56.3): both tails' CIs inside -> no material subset
C1_TOL = 1e-3          # mixed-corpus real rows must reproduce the null task's S (nats/token)
M3_DRAWS = 200
PURE = {"greedy": "gpt2m_greedy", "topk": "pilot_gpt2m"}
MIX = {"greedy": "mix50_greedy", "topk": "mix50_topk"}
TREE_DEC = {"gpt2m_greedy": "greedy", "pilot_gpt2m": "topk", "mix50_greedy": "greedy", "mix50_topk": "topk"}
CROSS = {43: 44, 44: 45, 45: 46, 46: 43}
TASK_RE = re.compile(r"^(?P<tree>.+)_s(?P<s>\d+)_c(?P<g>\d+)_(?P<role>own|cross|gen)$")
NULL_RE = re.compile(r"^null_c(?P<g>\d+)_sc(?P<sc>\d+)$")


def die(msg, code=2):
    print(f"STOP: {msg}")
    sys.exit(code)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pin_check(paths, skip):
    if skip:
        print("pin check SKIPPED (--no-pin; rehearsal only, never for the read)")
        return "unpinned"
    for p in paths:
        if subprocess.run(["git", "ls-files", "--error-unmatch", p], capture_output=True).returncode:
            die(f"{p} not tracked")
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", p]).returncode:
            die(f"{p} differs from HEAD")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def read_texts(path, key):
    with open(path) as f:
        return [json.loads(l)[key] for l in f if l.strip()]


# ----------------------------------------------------------------------------- loading
class Store:
    def __init__(self, a):
        self.a = a
        lines = open(a.manifest).read().splitlines()
        cols = lines[0].split("\t")
        self.tasks = {}
        for l in lines[1:]:
            t = dict(zip(cols, l.split("\t")))
            self.tasks[t["task"]] = t
        self._npz, self._meta, self._texts, self._labels = {}, {}, {}, {}

    def meta(self, task):
        if task not in self._meta:
            p = os.path.join(self.a.out_dir, task + ".json")
            if not os.path.exists(p):
                die(f"{task}: no json")
            self._meta[task] = json.load(open(p))
        return self._meta[task]

    def npz(self, task):
        if task not in self._npz:
            p = os.path.join(self.a.out_dir, task + ".npz")
            if not os.path.exists(p):
                die(f"{task}: no npz")
            z = np.load(p, allow_pickle=False)
            d = {k: z[k] for k in z.files}
            for k in ("row", "S", "prompt_hash"):
                if k not in d:
                    die(f"{task}: npz lacks '{k}'")
            n = len(d["row"])
            if len(d["S"]) != n or len(d["prompt_hash"]) != n:
                die(f"{task}: npz arrays of unequal length")
            if len(np.unique(d["row"])) != n:
                die(f"{task}: duplicate rows")
            if not np.all(np.isfinite(d["S"])):
                die(f"{task}: non-finite S")
            m = self.meta(task)
            if n != m["n_scored"]:
                die(f"{task}: npz rows {n} != n_scored {m['n_scored']}")
            self._npz[task] = d
        return self._npz[task]

    def texts(self, path):
        if path not in self._texts:
            self._texts[path] = read_texts(path, self.a.text_key)
        return self._texts[path]

    def excluded_rows(self, task):
        return [e["row"] if isinstance(e, dict) else int(e) for e in self.meta(task).get("excluded", [])]

    def synth_mask(self, task):
        """Boolean over the task's scored rows: True = synthetic. Pure trees: all True.
        Mix trees: by membership of the row text in the real reference (§43.2)."""
        if task in self._labels:
            return self._labels[task]
        t, d = self.tasks[task], self.npz(task)
        if t["real_ref"] == "-":
            lab = np.ones(len(d["row"]), bool)
            n_ex_syn = len(self.excluded_rows(task))
        else:
            texts, real = self.texts(t["corpus"]), set(self.texts(t["real_ref"]))
            lab = np.array([texts[r] not in real for r in d["row"]])
            n_ex_syn = sum(texts[r] not in real for r in self.excluded_rows(task))
            es = int(t["expect_synth"])
            if int(lab.sum()) + n_ex_syn != es:
                die(f"{task}: synth scored {int(lab.sum())} + synth excluded {n_ex_syn} != {es}")
        self._labels[task] = lab
        self._labels[task + "#nexsyn"] = n_ex_syn
        return lab

    def synth_S(self, task):
        return self.npz(task)["S"][self.synth_mask(task)]

    def n_ex_synth(self, task):
        self.synth_mask(task)
        return self._labels[task + "#nexsyn"]


def parse(store):
    synth, nulls = [], {}
    for name, t in store.tasks.items():
        if t["kind"] == "null":
            m = NULL_RE.match(name)
            if not m:
                die(f"unparsed null task {name}")
            nulls[(int(m["g"]), int(m["sc"]))] = name
        elif t["kind"] in ("synth", "canary"):
            m = TASK_RE.match(name)
            if not m:
                die(f"unparsed task {name}")
            if t["kind"] == "synth":
                s = int(m["s"])
                sc = s if m["role"] == "own" else CROSS[s]
                synth.append(dict(task=name, tree=m["tree"], s=s, g=int(m["g"]), role=m["role"], sc=sc,
                                  dec=TREE_DEC[m["tree"]]))
        else:
            die(f"unknown kind {t['kind']}")
    return synth, nulls


# ----------------------------------------------------------------------------- statistics
def tails(null_S):
    lo, hi = np.percentile(null_S, [10, 90])
    return lo, hi


def rates(S, lo, hi):
    return float(np.mean(S < lo)), float(np.mean(S > hi))


def rng_for(name):
    return np.random.default_rng([BOOT_SEED, zlib.crc32(name.encode())])


def pooled_boot(pairs, name, B):
    """pairs: list of (synth_S, null_S), one per seed. Resamples synth and null jointly within seed,
    re-estimates tau per replicate, pools counts over seeds. Returns point (lo, hi) and CIs."""
    rng = rng_for(name)
    n_tot = sum(len(s) for s, _ in pairs)
    c_lo = sum(int(np.sum(s < tails(n)[0])) for s, n in pairs)
    c_hi = sum(int(np.sum(s > tails(n)[1])) for s, n in pairs)
    blo, bhi = np.zeros(B), np.zeros(B)
    for b0 in range(0, B, CHUNK):
        b1 = min(B, b0 + CHUNK)
        k = b1 - b0
        for s, n in pairs:
            ni = rng.integers(0, len(n), size=(k, len(n)))
            tl, th = np.percentile(n[ni], [10, 90], axis=1)
            si = rng.integers(0, len(s), size=(k, len(s)))
            ss = s[si]
            blo[b0:b1] += np.sum(ss < tl[:, None], axis=1)
            bhi[b0:b1] += np.sum(ss > th[:, None], axis=1)
    blo /= n_tot
    bhi /= n_tot
    ci = lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
    return c_lo / n_tot, c_hi / n_tot, ci(blo), ci(bhi), n_tot


def spearman(x, y):
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).statistic)
    except ImportError:
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


def aligned(store, ta, tb, synth_only=True):
    """S of two tasks on the same corpus, matched by row."""
    da, db = store.npz(ta), store.npz(tb)
    ma = store.synth_mask(ta) if synth_only else np.ones(len(da["row"]), bool)
    mb = store.synth_mask(tb) if synth_only else np.ones(len(db["row"]), bool)
    ia = {r: i for i, r in enumerate(da["row"]) if ma[i]}
    common = [(ia[r], j) for j, r in enumerate(db["row"]) if mb[j] and r in ia]
    if not common:
        return np.array([]), np.array([])
    i, j = map(np.array, zip(*common))
    return da["S"][i], db["S"][j]


# ----------------------------------------------------------------------------- M3 helpers
class Tok:
    def __init__(self, path):
        if os.environ.get("STEP0_MOCK_TOKENIZER"):
            self.enc = lambda t: [zlib.crc32(w.encode()) % 50257 for w in t.split(" ")]
        else:
            from transformers import AutoTokenizer
            tk = AutoTokenizer.from_pretrained(path)
            self.enc = lambda t: tk(t, add_special_tokens=False)["input_ids"]

    def spans(self, texts, rows):
        """Generated-span token ids (positions 33..128) for the given rows; rows must re-tokenize to 128."""
        out = []
        for r in rows:
            ids = self.enc(texts[r])
            if len(ids) != 128:
                die(f"M3: row {r} re-tokenizes to {len(ids)} though it was scored")
            out.append(np.asarray(ids[32:], dtype=np.int64))
        return out


def m3_metrics(spans, keep, vocab_ref):
    """Distinct-2 (pooled unique within-span bigrams / total bigrams) and unigram
    KL(P_null || P_retained), add-0.5 smoothing over the GPT-2 vocabulary, on retained spans."""
    kept = [spans[i] for i in np.flatnonzero(keep)]
    big = np.concatenate([sp[:-1] * 50257 + sp[1:] for sp in kept])
    d2 = len(np.unique(big)) / len(big)
    cnt = np.bincount(np.concatenate(kept), minlength=50257).astype(float) + 0.5
    q = cnt / cnt.sum()
    kl = float(np.sum(vocab_ref * (np.log(vocab_ref) - np.log(q))))
    return d2, kl


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--manifest", default="slurm/step0_manifest.tsv")
    ap.add_argument("--out-dir", default=os.path.expandvars("$SCRATCH/collapse/step0_7p1/scores"))
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    ap.add_argument("--m3-draws", type=int, default=M3_DRAWS)
    ap.add_argument("--no-pin", action="store_true", help="rehearsal only")
    a = ap.parse_args()

    me = os.path.relpath(os.path.abspath(__file__))
    head = pin_check([me, a.manifest, "src/step0_score.py"], a.no_pin)
    print(f"step0_analysis md5 {md5(__file__)}  manifest md5 {md5(a.manifest)}  head {head}  "
          f"boot {a.boot}  m3_draws {a.m3_draws}  mode {'CHECK' if a.check else 'READ'}")
    if not a.check and (a.boot != B_DEFAULT or a.m3_draws != M3_DRAWS) and not a.no_pin:
        die("the read must use the registered B and M3 draw count")

    st = Store(a)
    synth, nulls = parse(st)
    by = {(x["tree"], x["s"], x["g"], x["role"]): x for x in synth}

    # ---- structural checks (no S statistic)
    for name in list(st.tasks):
        if st.tasks[name]["kind"] == "canary":
            st.meta(name)
            continue
        st.npz(name)
        if st.tasks[name]["kind"] == "synth":
            st.synth_mask(name)
    ex = defaultdict(lambda: [0, 0])
    for x in synth:
        m = st.meta(x["task"])
        k = (x["tree"], x["role"])
        ex[k][0] += st.n_ex_synth(x["task"])
        ex[k][1] += st.n_ex_synth(x["task"]) + int(st.synth_mask(x["task"]).sum())
    print("\nsynthetic-row exclusions by tree and scorer role (re-tokenization != 128):")
    for k in sorted(ex):
        print(f"  {k[0]:13s} {k[1]:5s} {ex[k][0]:6d}/{ex[k][1]:7d} ({100*ex[k][0]/ex[k][1]:.2f}%)")
    for g in range(1, 5):
        for s in CROSS:
            if (g, s) not in nulls:
                die(f"missing null task for gen {g} scorer {s}")
    k1_seeds = {dec: sorted(s for (t, s, g, r) in by if t == PURE[dec] and g == 1 and r == "own") for dec in PURE}
    print(f"\nK1 seeds: {k1_seeds}")
    print(f"tasks: {len(synth)} synth, {len(nulls)} null; all inputs present and consistent")
    # tokenizer used by M3 must load offline and reproduce a scored row's 128 tokens
    for dec, tree in PURE.items():
        x = by[(tree, k1_seeds[dec][0], 1, "own")]
        t = st.tasks[x["task"]]
        Tok(t["scorer"]).spans(st.texts(t["corpus"]), st.npz(x["task"])["row"][:50])
    print("M3 tokenizer: loads, 50/50 scored rows re-tokenize to 128 for each decoder")
    if a.check:
        print("\nCHECK PASS — no statistic of S computed or printed")
        return

    # ---- canary C1: real rows of mixed corpora reproduce the null task's S
    worst, n_cmp = 0.0, 0
    for x in synth:
        if x["tree"] not in MIX.values():
            continue
        t, d = st.tasks[x["task"]], st.npz(x["task"])
        nt = nulls[(x["g"], x["sc"])]
        nd = st.npz(nt)
        if st.tasks[nt]["corpus"] != t["real_ref"]:
            die(f"C1: {x['task']} real_ref is not the corpus of {nt}")
        ntexts, ctexts = st.texts(st.tasks[nt]["corpus"]), st.texts(t["corpus"])
        nS = {ntexts[r]: v for r, v in zip(nd["row"], nd["S"])}
        lab = st.synth_mask(x["task"])
        for r, v, sy in zip(d["row"], d["S"], lab):
            if sy:
                continue
            txt = ctexts[r]
            if txt not in nS:
                die(f"C1: {x['task']} real row {r} not scored in {nt}")
            worst = max(worst, abs(v - nS[txt]))
            n_cmp += 1
    print(f"\nC1 mixed-corpus real rows vs null: {n_cmp} rows compared, max |dS| {worst:.2e} (tol {C1_TOL:g})")
    if not worst <= C1_TOL:
        die("C1 canary fired — nothing below is reported", 3)
    print("CANARIES PASS\n" + "=" * 78)

    # ---- K1
    verdict = {}
    print("\nK1 — corpus training gen 1, r = 100, tail enrichment vs the null 10% (pooled over seeds)")
    for dec, tree in PURE.items():
        for role in ("own", "cross"):
            pairs, pairs_lo, pairs_hi = [], [], []
            for s in k1_seeds[dec]:
                x = by[(tree, s, 1, role)]
                sS, nS = st.synth_S(x["task"]), st.npz(nulls[(1, x["sc"])])["S"]
                e = st.n_ex_synth(x["task"])
                pairs.append((sS, nS))
                pairs_lo.append((np.concatenate([sS, np.full(e, -np.inf)]), nS))
                pairs_hi.append((np.concatenate([sS, np.full(e, np.inf)]), nS))
                lo, hi = tails(nS)
                rl, rh = rates(sS, lo, hi)
                print(f"  {dec:6s} {role:5s} s{s}: n={len(sS)} excl={e} tau=({lo:+.4f}, {hi:+.4f}) "
                      f"lower {rl:.4f} upper {rh:.4f}")
            res = {}
            for vn, pp in (("registered", pairs), ("excl->lower", pairs_lo), ("excl->upper", pairs_hi)):
                pl, ph, cl, ch, n = pooled_boot(pp, f"K1|{dec}|{role}|{vn}", a.boot)
                fires = cl[0] <= NULL_RATE <= cl[1] and ch[0] <= NULL_RATE <= ch[1]
                immaterial = MAT[0] <= cl[0] and cl[1] <= MAT[1] and MAT[0] <= ch[0] and ch[1] <= MAT[1]
                res[vn] = fires
                res[vn + "#mat"] = immaterial
                print(f"  {dec:6s} {role:5s} POOLED [{vn:11s}] n={n} lower {pl:.4f} [{cl[0]:.4f}, {cl[1]:.4f}]  "
                      f"upper {ph:.4f} [{ch[0]:.4f}, {ch[1]:.4f}]  -> {'FIRES' if fires else 'does not fire'}"
                      f"; K1-material: {'IMMATERIAL' if immaterial else 'material'}")
                if vn == "registered":
                    res["point"] = (pl, ph)
            robust = len({res["registered"], res["excl->lower"], res["excl->upper"]}) == 1
            print(f"  {dec:6s} {role:5s} exclusion-robust: {'YES' if robust else 'NO'}")
            verdict[(dec, role)] = res
    own = {dec: verdict[(dec, "own")]["registered"] for dec in PURE}
    print("\nK1 VERDICT (own scorer, registered):")
    for dec in PURE:
        agree = verdict[(dec, "own")]["registered"] == verdict[(dec, "cross")]["registered"]
        note = "  [greedy own-scorer pass is expected by construction, §56.2]" if dec == "greedy" and not own[dec] else ""
        print(f"  {dec:6s}: {'FIRES' if own[dec] else 'does not fire'}; cross-scorer {'agrees' if agree else 'DISAGREES'}{note}")
    mat = {dec: verdict[(dec, "own")]["registered#mat"] for dec in PURE}
    print("K1-material (secondary, §56.3; does not override K1): " +
          "  ".join(f"{d} {'IMMATERIAL' if mat[d] else 'material'}" for d in PURE))
    if all(own.values()):
        print("  => filter branch FALSIFIED (fires in both decoders)")
    elif any(own.values()):
        live = [d for d in PURE if not own[d]]
        print(f"  => filter branch continues for {live} only (§56.1)")
    else:
        print("  => K1 does not fire; filter branch continues")

    # ---- direction check (§56.7)
    print("\nDIRECTION (§41, operationalised §56.7): lower-tail pooled rate vs upper, own scorer, gen 1, r = 100")
    both = False
    for dec in PURE:
        pl, ph = verdict[(dec, "own")]["point"]
        if own[dec]:
            print(f"  {dec:6s}: lower {pl:.4f} upper {ph:.4f} -> not evaluated (K1 fires for this decoder)")
            continue
        c = ph > pl
        both |= c
        print(f"  {dec:6s}: lower {pl:.4f} upper {ph:.4f} -> {'CONTRADICTS derivation' if c else 'consistent'}")
    print(f"  => 7.1 runs {'BOTH directions (registered two-arm)' if both else 'the lower tail (registered direction)'}")

    # ---- K2
    print("\nK2 — Spearman(S own, S cross), gen-1 r = 100 corpora")
    rhos = []
    for dec, tree in PURE.items():
        for s in k1_seeds[dec]:
            xa, xb = aligned(st, by[(tree, s, 1, "own")]["task"], by[(tree, s, 1, "cross")]["task"])
            r = spearman(xa, xb)
            rhos.append(r)
            print(f"  {tree}_s{s}: rho {r:+.4f} (n={len(xa)}, scorers {s} vs {CROSS[s]})")
    med = float(np.median(rhos))
    print(f"  median {med:+.4f}  min {min(rhos):+.4f}  over {len(rhos)} corpora -> "
          f"{'K2 FIRES' if med < K2_MIN else 'K2 does not fire'} (threshold {K2_MIN})")
    print("  reference, null slices (human text) under the same scorer pairs:")
    for g in range(1, 5):
        vals = []
        for s in (43, 44, 45, 46):
            xa, xb = aligned(st, nulls[(g, s)], nulls[(g, CROSS[s])], synth_only=False)
            vals.append(spearman(xa, xb))
        print(f"    gen {g}: " + "  ".join(f"{s}v{CROSS[s]} {v:+.4f}" for s, v in zip((43, 44, 45, 46), vals)))
    print("  descriptive, all other corpora (synthetic rows):")
    for tree in ("gpt2m_greedy", "pilot_gpt2m", "mix50_greedy", "mix50_topk"):
        for g in range(1, 5):
            vals = []
            for s in sorted({k[1] for k in by if k[0] == tree}):
                if (tree, s, g, "own") in by:
                    xa, xb = aligned(st, by[(tree, s, g, "own")]["task"], by[(tree, s, g, "cross")]["task"])
                    vals.append(f"s{s} {spearman(xa, xb):+.3f}")
            print(f"    {tree:13s} c{g}: " + "  ".join(vals))

    # ---- rank stability (pure chains, own scorer, prompt-matched)
    print("\nRANK STABILITY — own scorer, pure chains, documents matched by prompt_hash (unique in both)")
    for dec, tree in PURE.items():
        for s in k1_seeds[dec]:
            out = []
            for g1, g2 in ((1, 2), (2, 3), (3, 4), (1, 4)):
                ka, kb = (tree, s, g1, "own"), (tree, s, g2, "own")
                if ka not in by or kb not in by:
                    continue
                da, db = st.npz(by[ka]["task"]), st.npz(by[kb]["task"])
                ua, ca = np.unique(da["prompt_hash"], return_counts=True)
                ub, cb = np.unique(db["prompt_hash"], return_counts=True)
                keep = np.intersect1d(ua[ca == 1], ub[cb == 1])
                ia = {h: i for i, h in enumerate(da["prompt_hash"])}
                ib = {h: i for i, h in enumerate(db["prompt_hash"])}
                xa = np.array([da["S"][ia[h]] for h in keep])
                xb = np.array([db["S"][ib[h]] for h in keep])
                out.append(f"c{g1}-c{g2} {spearman(xa, xb):+.3f} (n={len(keep)})")
            print(f"  {tree}_s{s}: " + "  ".join(out))

    # ---- descriptive enrichment, every synth task (point estimates)
    print("\nDESCRIPTIVE — enrichment for every synth task vs its scorer's null of the same gen (synthetic rows)")
    for x in sorted(synth, key=lambda z: (z["tree"], z["role"], z["s"], z["g"])):
        nS = st.npz(nulls[(x["g"], x["sc"])])["S"]
        lo, hi = tails(nS)
        rl, rh = rates(st.synth_S(x["task"]), lo, hi)
        line = f"  {x['task']:28s} lower {rl:.4f} upper {rh:.4f}"
        if x["tree"] in MIX.values():
            d = st.npz(x["task"])
            rS = d["S"][~st.synth_mask(x["task"])]
            rrl, rrh = rates(rS, lo, hi)
            line += f"   | real rows lower {rrl:.4f} upper {rrh:.4f}"
        print(line)

    # ---- M3
    print("\nM3 — non-Δ check (§56.9): drop a tail vs random drops of equal count; retained-corpus "
          "Distinct-2 and unigram KL(null || retained), generated spans, own scorer, gen 1, r = 100")
    tok_cache = {}
    for dec, tree in PURE.items():
        for s in k1_seeds[dec]:
            x = by[(tree, s, 1, "own")]
            t, d = st.tasks[x["task"]], st.npz(x["task"])
            scp = t["scorer"]
            if scp not in tok_cache:
                tok_cache[scp] = Tok(scp)
            tk = tok_cache[scp]
            nt = nulls[(1, x["sc"])]
            nd = st.npz(nt)
            nsp = tk.spans(st.texts(st.tasks[nt]["corpus"]), nd["row"])
            ref = np.bincount(np.concatenate(nsp), minlength=50257).astype(float) + 0.5
            ref /= ref.sum()
            spans = tk.spans(st.texts(t["corpus"]), d["row"])
            nbytes = np.array([len(st.texts(t["corpus"])[r].encode()) for r in d["row"]])
            lo, hi = tails(nd["S"])
            full = m3_metrics(spans, np.ones(len(spans), bool), ref)
            rng = rng_for(f"M3|{dec}|{s}")
            for tail, sel in (("lower", d["S"] < lo), ("upper", d["S"] > hi)):
                k = int(sel.sum())
                obs = m3_metrics(spans, ~sel, ref)
                rd2, rkl, rby = [], [], []
                for _ in range(a.m3_draws):
                    drop = np.zeros(len(spans), bool)
                    drop[rng.choice(len(spans), k, replace=False)] = True
                    v = m3_metrics(spans, ~drop, ref)
                    rd2.append(v[0]); rkl.append(v[1]); rby.append(nbytes[drop].sum())
                rd2, rkl = np.array(rd2), np.array(rkl)
                print(f"  {dec:6s} s{s} drop {tail} (k={k}, bytes {nbytes[sel].sum()} vs random {np.mean(rby):.0f}): "
                      f"D2 full {full[0]:.5f} -> {obs[0]:.5f} (random {rd2.mean():.5f}±{rd2.std():.5f}, "
                      f"pct {100*np.mean(rd2 < obs[0]):.1f}) | KL full {full[1]:.5f} -> {obs[1]:.5f} "
                      f"(random {rkl.mean():.5f}±{rkl.std():.5f}, pct {100*np.mean(rkl < obs[1]):.1f})")
    print("\nREAD COMPLETE")


if __name__ == "__main__":
    main()
