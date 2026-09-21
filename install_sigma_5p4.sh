# Install the Sept 17 SIGMA baseline on Narval. Paste the whole block into a shell (any directory).
# Runs in a subshell, so a REFUSE or failed check cannot close your login session.
# Refuses to overwrite existing files; verifies md5 of every file written; runs the self-test.
(
set -euo pipefail
cd ~/modelcollapse
mkdir -p src scripts slurm
mkdir -p logs/sigma
for f in src/sigma.py scripts/sigma_gate.py compute_sigma_thresholds.py slurm/sigma_array.sbatch slurm/sigma_chains_topk.txt slurm/sigma_chains_greedy.txt; do [ -e "$f" ] && { echo "REFUSE: $f exists"; exit 1; }; done
cat > src/sigma.py <<'SIGMA_EOF_0'
#!/usr/bin/env python
"""SIGMA-UB baseline detector (Gu et al. 2026, arXiv:2601.03385 v3, Algorithm 1).

Phase 5.4 baseline (b). Per chain (arm, seed): embed each generation's eval completions
with a frozen sentence encoder, then compute on a FIXED observed index set I_A:

  S_k        = logdet(G_A + delta I)                        G_A = A^T A, A = n_A x m, rows L2-normalised
  Track II   U_cov  = S_k - m log n_A                        (eq. 3.19, "sensitive")
  Track I    G_KF   = logdet(G_A + (beta_k + delta) I) - m log(beta_k + delta),
             beta_k = (n_k - n_A) * rho                      (eqs. 3.16-3.17, "conservative")
  Drift      dU(g) = U_cov(g) - U_cov(0);  dG(g) = G_KF(g) - G_KF(0)       (eq. 3.20)

Collapse = sustained NEGATIVE drift. Detector score used by 5.4 = -dU (higher = more collapsed).
Hyperparameters are the paper's (Table 5) and are FIXED in decisions.md Sept 17 before any real value:
encoder all-MiniLM-L6-v2 (m=384), delta=1e-3, rho=1.0, baseline g=0, fixed I_A per run.
Two sample settings are computed: n_A=500 of n_k=1000 (Algorithm 1 as written; Track I defined)
and n_A=n_k=1000 (full sample; Track I degenerates to Track II + const, so it is not reported there).

BLINDING: this script never prints a metric value. It writes JSON under --out-root and prints counts only.
Usage:
  python src/sigma.py --root $SCRATCH/collapse --arm control_gpt2m --seed 43 --out-root $SCRATCH/collapse/sigma
  python src/sigma.py --self-test
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
import numpy as np

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
M_EXPECTED = 384
DELTA = 1e-3
RHO = 1.0
N_A_SUB = 500
INDEX_SEED = 20260917          # fixed once; identical I_A for every chain, arm and decoder
GENS = range(0, 8)
REFUSE_ARM_SUBSTR = ()          # none: SIGMA is computed on grid chains too; blinding is by not printing


# ----------------------------------------------------------------------------- maths
def logdet_pd(S):
    """log det of a symmetric positive-definite matrix via Cholesky (Appendix B.4)."""
    L = np.linalg.cholesky(S)
    return float(2.0 * np.sum(np.log(np.diag(L))))


def sigma_ub(E_A, n_k, delta=DELTA, rho=RHO):
    """Algorithm 1 for one checkpoint. E_A: (n_A, m) rows = observed embeddings."""
    E_A = np.asarray(E_A, dtype=np.float64)
    n_A, m = E_A.shape
    if n_A <= m:
        raise ValueError(f"n_A={n_A} must exceed m={m} (paper Sec 3.1)")
    if n_A > n_k:
        raise ValueError(f"n_A={n_A} > n_k={n_k}")
    sq = np.einsum("ij,ij->i", E_A, E_A)
    if sq.max() > rho * (1 + 1e-5):
        raise ValueError(f"max ||v||^2 = {sq.max():.6f} exceeds rho={rho}; embeddings must be normalised")
    G = E_A.T @ E_A
    I = np.eye(m)
    beta = (n_k - n_A) * rho
    S = logdet_pd(G + delta * I)
    out = {"n_A": n_A, "n_k": int(n_k), "m": m, "beta": beta, "S": S,
           "U_cov": S - m * np.log(n_A)}
    if n_A < n_k:
        out["G_KF"] = logdet_pd(G + (beta + delta) * I) - m * np.log(beta + delta)
    return out


def fixed_index_set(n_k, n_A, seed=INDEX_SEED):
    idx = np.sort(np.random.default_rng(seed).choice(n_k, size=n_A, replace=False))
    return idx, hashlib.md5(idx.astype(np.int64).tobytes()).hexdigest()


# ----------------------------------------------------------------------------- io
def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_completions(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    prompts = [r["prompt"] for r in rows]
    texts = [r["model"] for r in rows]
    return prompts, texts


def get_encoder(name, device):
    if name == "fake":  # tests only
        def enc(texts):
            rng_rows = []
            for t in texts:
                s = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
                v = np.random.default_rng(s).standard_normal(M_EXPECTED)
                rng_rows.append(v / np.linalg.norm(v))
            return np.asarray(rng_rows, dtype=np.float32)
        return enc
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(name, device=device)

    def enc(texts):
        return model.encode(texts, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    return enc


def run_chain(root, arm, seed, out_root, encoder_name, device, allow_fake=False):
    if encoder_name == "fake" and not allow_fake:
        sys.exit("REFUSE: fake encoder outside --self-test")
    root, out_root = Path(root), Path(out_root)
    ro, oo = root.resolve(), out_root.resolve()
    if (ro in oo.parents or oo == ro) and oo != ro / "sigma":
        sys.exit(f"REFUSE: out-root {oo} is inside the collapse root but is not {ro / 'sigma'}")
    if oo in (ro / arm).resolve().parents or oo == (ro / arm).resolve():
        sys.exit("REFUSE: out-root contains the arm root")
    odir = out_root / arm
    odir.mkdir(parents=True, exist_ok=True)

    enc = None
    rec = {"arm": arm, "seed": seed, "encoder": encoder_name, "delta": DELTA, "rho": RHO,
           "index_seed": INDEX_SEED, "gens": {}}
    prompt_md5 = None
    n_k = None
    embs = {}
    for g in GENS:
        f = root / arm / f"seed{seed}" / f"gen{g}" / "eval_completions.jsonl"
        if not f.exists():
            sys.exit(f"REFUSE: missing {f}")
        prompts, texts = read_completions(f)
        pm = hashlib.md5("\n".join(prompts).encode()).hexdigest()
        if prompt_md5 is None:
            prompt_md5, n_k = pm, len(texts)
        if len(texts) != n_k:
            sys.exit(f"REFUSE: n_k differs at gen {g}: {len(texts)} vs {n_k}")
        if pm != prompt_md5:
            sys.exit(f"REFUSE: prompt set differs at gen {g} (chain must share one frozen prompt bank)")
        fmd5 = md5_file(f)
        cache = odir / f"seed{seed}_gen{g}_emb.npy"
        meta = odir / f"seed{seed}_gen{g}_emb.md5"
        if cache.exists() and meta.exists() and meta.read_text().split()[0] == fmd5:
            E = np.load(cache)
        else:
            if enc is None:
                enc = get_encoder(encoder_name, device)
            E = enc(texts)
            if E.shape != (n_k, M_EXPECTED):
                sys.exit(f"REFUSE: embedding shape {E.shape}, expected ({n_k}, {M_EXPECTED})")
            np.save(cache, E)
            meta.write_text(f"{fmd5}  {f}\n")
        embs[g] = E
        rec["gens"][str(g)] = {"completions_md5": fmd5, "completions_path": str(f),
                                "resolved": str(f.resolve())}

    idx, idx_md5 = fixed_index_set(n_k, N_A_SUB)
    rec.update({"n_k": n_k, "prompt_md5": prompt_md5, "index_md5": idx_md5, "n_A_sub": N_A_SUB})
    for g in GENS:
        sub = sigma_ub(embs[g][idx], n_k)
        full = sigma_ub(embs[g], n_k)
        rec["gens"][str(g)].update({"sub": sub, "full": full})
    base_sub, base_full = rec["gens"]["0"]["sub"], rec["gens"]["0"]["full"]
    for g in GENS:
        d = rec["gens"][str(g)]
        d["dU_sub"] = d["sub"]["U_cov"] - base_sub["U_cov"]
        d["dG_KF_sub"] = d["sub"]["G_KF"] - base_sub["G_KF"]
        d["dU_full"] = d["full"]["U_cov"] - base_full["U_cov"]
    out = odir / f"seed{seed}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    os.replace(tmp, out)
    # counts only -- never values
    print(f"SIGMA wrote {out} | gens={len(GENS)} n_k={n_k} n_A_sub={N_A_SUB} "
          f"index_md5={idx_md5} prompt_md5={prompt_md5}")


# ----------------------------------------------------------------------------- canaries
def self_test():
    rng = np.random.default_rng(0)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
        ok &= bool(cond)

    # C1 Cholesky logdet == eigen logdet
    X = rng.standard_normal((600, 50)); S = X.T @ X + 1e-3 * np.eye(50)
    a, b = logdet_pd(S), float(np.sum(np.log(np.linalg.eigvalsh(S))))
    check("C1 cholesky==eigh logdet", abs(a - b) < 1e-8, f"|diff|={abs(a-b):.2e}")

    # C2 Track I closed form (eq. 3.18): sum log(1 + lambda/(beta+delta))
    E = rng.standard_normal((500, 384)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    r = sigma_ub(E, 1000)
    lam = np.linalg.eigvalsh(E.T @ E)
    cf = float(np.sum(np.log1p(lam / (r["beta"] + DELTA))))
    check("C2 G_KF == eq.3.18 eigen form", abs(r["G_KF"] - cf) < 1e-6, f"|diff|={abs(r['G_KF']-cf):.2e}")
    check("C2b beta == (n_k-n_A)*rho", r["beta"] == 500.0)

    # C3 Track II tracks log det C (LLN, Bridge II) for known diagonal C, m small, n_A large
    m = 16; c = np.linspace(0.05, 1.0, m); n = 40000
    Z = rng.standard_normal((n, m)) * np.sqrt(c) / np.sqrt(c.sum())   # E||v||^2 = 1
    Z /= max(1.0, np.sqrt((Z ** 2).sum(1)).max())                      # enforce ||v||<=1 (rho)
    Cemp = (Z.T @ Z) / n
    r = sigma_ub(Z, n)
    tgt = float(np.sum(np.log(np.linalg.eigvalsh(Cemp))))
    check("C3 U_cov ~= log det C (full sample)", abs(r["U_cov"] - tgt) < 1e-2, f"|diff|={abs(r['U_cov']-tgt):.2e}")

    # C4 size invariance: U_cov at n_A=5000 vs 40000 within LLN tolerance (Thm 3 ~ z*sigma/sqrt(n_A))
    idx, _ = fixed_index_set(n, 5000, seed=1)
    r5 = sigma_ub(Z[idx], n)
    q = np.einsum("ij,jk,ik->i", Z, np.linalg.inv(Cemp), Z)              # Mahalanobis X^T C^-1 X (eq. 3.8)
    bound = 1.96 * q.std() * np.sqrt(1 / 5000 - 1 / n)                   # Thm 3, 95%
    check("C4 U_cov size-invariant within Thm-3 95% bound (5k vs 40k)",
          abs(r5["U_cov"] - r["U_cov"]) < bound,
          f"|diff|={abs(r5['U_cov']-r['U_cov']):.3f} bound={bound:.3f}")

    # C5 positive control: collapse to 30 directions + small noise -> strongly negative drift, both tracks
    def unit(A): return A / np.linalg.norm(A, axis=1, keepdims=True)
    healthy = unit(rng.standard_normal((1000, 384)))
    centers = unit(rng.standard_normal((30, 384)))
    collapsed = unit(centers[rng.integers(0, 30, 1000)] + 0.05 * rng.standard_normal((1000, 384)))
    idx, _ = fixed_index_set(1000, 500)
    h, cl = sigma_ub(healthy[idx], 1000), sigma_ub(collapsed[idx], 1000)
    dU, dG = cl["U_cov"] - h["U_cov"], cl["G_KF"] - h["G_KF"]
    check("C5 collapsed -> dU << 0", dU < -100, f"dU={dU:.1f}")
    check("C5b collapsed -> dG_KF < 0", dG < 0, f"dG={dG:.3f}")

    # C6 inverse canary: fresh draw from the same healthy distribution -> |dU| small vs C5
    healthy2 = unit(rng.standard_normal((1000, 384)))
    h2 = sigma_ub(healthy2[idx], 1000)
    check("C6 same distribution -> |dU| < 5% of C5", abs(h2["U_cov"] - h["U_cov"]) < 0.05 * abs(dU),
          f"|dU|={abs(h2['U_cov']-h['U_cov']):.2f} vs {abs(dU):.1f}")

    # C7 fixed index set is deterministic and recorded
    i1, m1 = fixed_index_set(1000, 500); i2, m2 = fixed_index_set(1000, 500)
    check("C7 I_A deterministic", np.array_equal(i1, i2) and m1 == m2, m1)

    # C8 refusals
    for name, fn in [("C8 unnormalised refused", lambda: sigma_ub(2 * healthy[idx], 1000)),
                     ("C8b n_A<=m refused", lambda: sigma_ub(healthy[:384], 1000)),
                     ("C8c n_A>n_k refused", lambda: sigma_ub(healthy, 999))]:
        try:
            fn(); check(name, False)
        except ValueError:
            check(name, True)

    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root"); ap.add_argument("--arm"); ap.add_argument("--seed", type=int)
    ap.add_argument("--out-root"); ap.add_argument("--encoder", default=ENCODER)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-fake-encoder", action="store_true", help="rehearsal only")
    a = ap.parse_args()
    if a.self_test:
        sys.exit(self_test())
    if not all([a.root, a.arm, a.seed is not None, a.out_root]):
        ap.error("--root --arm --seed --out-root required")
    run_chain(a.root, a.arm, a.seed, a.out_root, a.encoder, a.device, a.allow_fake_encoder)


if __name__ == "__main__":
    main()
SIGMA_EOF_0
cat > scripts/sigma_gate.py <<'SIGMA_EOF_1'
#!/usr/bin/env python
"""Gates for the SIGMA pass. Prints PASS/FAIL and counts; missing evidence is a FAIL.

  --pilot  : control_gpt2m s43 only (control data, values may print)
     K1 known-value canary: metrics.embedding_variance re-run on gen1 texts reproduces the stored emb_variance
        (proves the node loads the recorded encoder; only meaningful if cfg sbert_model == sigma.ENCODER)
     K4 re-encode gen1 fresh; max |diff| vs cached embeddings < 1e-5
  --all LIST : after the array, over every chain in LIST ("ARM SEED" lines). Prints NO metric values.
     K3 chains whose gen-0 completions resolve to the same file have byte-identical gen-0 embeddings
     K5 every chain: n_k=1000, one prompt_md5, one index_md5, 8 gens
     K6 positive-control sign (pre-registered sanity gate): r=100 arms dU_sub(7) < 0, reported as a count
"""
import argparse, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "src"))
import sigma  # noqa: E402

R100 = {"pilot_gpt2m", "gpt2m_greedy"}
fails = 0


def res(name, ok, detail=""):
    global fails
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    fails += (not ok)


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def pilot(root, sroot, device):
    cell = root / "control_gpt2m/seed43/gen1"
    texts = [json.loads(l)["model"] for l in open(cell / "eval_completions.jsonl")]
    import yaml
    cfg = yaml.safe_load(open(HERE / "configs/control_gpt2m.yaml"))
    name = cfg["eval"]["sbert_model"]
    same = name.split("/")[-1] == sigma.ENCODER.split("/")[-1]
    print(f"cfg sbert_model = {name}; sigma encoder = {sigma.ENCODER}; same = {same}")
    from metrics import embedding_variance
    stored = json.load(open(cell / "metrics.json"))["emb_variance"]
    got = embedding_variance(texts, name, device)
    res("K1 emb_variance reproduces stored value", abs(got - stored) <= 1e-4 * max(1, abs(stored)),
        f"stored={stored:.6f} recomputed={got:.6f}")
    cache = sroot / "control_gpt2m/seed43_gen1_emb.npy"
    if not cache.exists():
        res("K4 cached embeddings exist", False, str(cache)); return
    E = sigma.get_encoder(sigma.ENCODER, device)(texts)
    d = float(np.abs(E - np.load(cache)).max())
    res("K4 re-encode matches cache", d < 1e-5, f"max|diff|={d:.2e}")


def all_chains(root, sroot, listfile):
    chains = [l.split() for l in open(listfile) if l.strip()]
    gen0 = defaultdict(set); prompts, idxs = set(), set(); neg, tot = defaultdict(int), defaultdict(int)
    missing = 0
    for arm, seed in chains:
        f = sroot / arm / f"seed{seed}.json"
        if not f.exists():
            missing += 1; print(f"  missing {f}"); continue
        r = json.load(open(f))
        ok = r["n_k"] == 1000 and len(r["gens"]) == 8 and r["arm"] == arm and str(r["seed"]) == seed
        if not ok:
            res(f"K5 record shape {arm} s{seed}", False)
        prompts.add(r["prompt_md5"]); idxs.add(r["index_md5"])
        gen0[r["gens"]["0"]["resolved"]].add(md5(sroot / arm / f"seed{seed}_gen0_emb.npy"))
        if arm in R100:
            tot[arm] += 1; neg[arm] += r["gens"]["7"]["dU_sub"] < 0
    res("K5 all chain records present", missing == 0, f"{len(chains) - missing}/{len(chains)}")
    res("K5 one prompt_md5 across chains", len(prompts) == 1, f"{len(prompts)} distinct")
    res("K5 one index_md5 across chains", len(idxs) == 1, f"{len(idxs)} distinct")
    shared = {k: v for k, v in gen0.items()}
    bad = sum(len(v) != 1 for v in shared.values())
    res("K3 shared gen-0 files -> identical embeddings", bad == 0 and len(shared) > 0,
        f"{len(shared)} distinct gen-0 files, {bad} with >1 embedding md5")
    for arm in sorted(tot):
        print(f"INFO  K6 {arm}: dU_sub(7) < 0 in {neg[arm]}/{tot[arm]} seeds (pre-registered expectation: all)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expandvars("$SCRATCH/collapse"))
    ap.add_argument("--pilot", action="store_true"); ap.add_argument("--all")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    root = Path(a.root); sroot = root / "sigma"
    if a.pilot:
        pilot(root, sroot, a.device)
    if a.all:
        all_chains(root, sroot, a.all)
    if not (a.pilot or a.all):
        ap.error("--pilot and/or --all")
    print("GATE", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
SIGMA_EOF_1
cat > compute_sigma_thresholds.py <<'SIGMA_EOF_2'
#!/usr/bin/env python
"""tau_SIGMA from CONTROL chains only (decisions.md Sept 17, pre-registered).

tau = max over control seeds and gens 1-7 of -dU(g), per decoder and per sample setting (sub / full).
Mirrors compute_label_thresholds.py: refuses any non-control arm, so no grid SIGMA value can enter.
  python compute_sigma_thresholds.py                                   # top-k: control_gpt2m 43-46
  ARM=control_gpt2m_greedyeval SEEDS=43,44,45 python compute_sigma_thresholds.py
"""
import json, os, sys
from pathlib import Path

ALLOWED = {"control_gpt2m", "control_gpt2m_greedyeval"}
ARM = os.environ.get("ARM", "control_gpt2m")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "43,44,45,46").split(",")]
ROOT = Path(os.environ.get("SIGMA_ROOT", os.path.expandvars("$SCRATCH/collapse/sigma")))

if ARM not in ALLOWED:
    sys.exit(f"REFUSE: {ARM} is not a control arm {sorted(ALLOWED)}")
rows = {}
for s in SEEDS:
    f = ROOT / ARM / f"seed{s}.json"
    if not f.exists():
        sys.exit(f"REFUSE: missing {f}")
    r = json.load(open(f))
    if r["arm"] != ARM or r["seed"] != s:
        sys.exit(f"REFUSE: {f} records arm={r['arm']} seed={r['seed']}")
    rows[s] = r
prompts = {r["prompt_md5"] for r in rows.values()}; idx = {r["index_md5"] for r in rows.values()}
if len(prompts) != 1 or len(idx) != 1:
    sys.exit("REFUSE: control chains disagree on prompt_md5 or index_md5")
print(f"arm={ARM} seeds={SEEDS} n_k={rows[SEEDS[0]]['n_k']} index_md5={idx.pop()}")
for key in ("dU_sub", "dU_full", "dG_KF_sub"):
    print(f"\n-{key}(g), control shifts (collapse direction = positive here)")
    print("seed  " + "  ".join(f"g{g:>8}" for g in range(1, 8)))
    best = (float("-inf"), None)
    for s in SEEDS:
        vals = [-rows[s]["gens"][str(g)][key] for g in range(1, 8)]
        print(f"{s:<5} " + "  ".join(f"{v:+9.3f}" for v in vals))
        for g, v in zip(range(1, 8), vals):
            if v > best[0]:
                best = (v, f"s{s} g{g}")
    tag = "reported only" if key == "dG_KF_sub" else "tau"
    print(f"{tag} {key}: {best[0]:+.3f} (set by {best[1]})")
SIGMA_EOF_2
cat > slurm/sigma_array.sbatch <<'SIGMA_EOF_3'
#!/bin/bash
#SBATCH --account=def-enaskt_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=0:40:00
#SBATCH --array=1-16%8
#SBATCH --exclude=ng11002,ng11003
#SBATCH --output=logs/sigma/sigma-%A_%a.out
# SIGMA-UB baseline (5.4 baseline b). CPU-only: embeds eval completions, computes Algorithm 1.
# LIST selects the chain list ("ARM SEED" per line). Writes ONLY under $SCRATCH/collapse/sigma. Prints no values.
set -euo pipefail
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
cd "$HOME/modelcollapse"; source scripts/narval_env.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
LIST=${LIST:-slurm/sigma_chains_topk.txt}
LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
read -r ARM SEED <<<"$LINE"
[ -n "${ARM:-}" ] && [ -n "${SEED:-}" ] || { echo "REFUSE: no line $SLURM_ARRAY_TASK_ID in $LIST"; exit 1; }
echo "[task $SLURM_ARRAY_TASK_ID] list=$LIST arm=$ARM seed=$SEED host=$(hostname) sigma_md5=$(md5sum src/sigma.py | cut -d' ' -f1)"
python src/sigma.py --root "$SCRATCH/collapse" --arm "$ARM" --seed "$SEED" \
  --out-root "$SCRATCH/collapse/sigma" --device cpu
echo "=== done | $(date) ==="
SIGMA_EOF_3
cat > slurm/sigma_chains_topk.txt <<'SIGMA_EOF_4'
control_gpt2m 43
control_gpt2m 44
control_gpt2m 45
control_gpt2m 46
pilot_gpt2m 43
pilot_gpt2m 44
pilot_gpt2m 45
mix25_topk 43
mix25_topk 44
mix25_topk 45
mix50_topk 43
mix50_topk 44
mix50_topk 45
mix75_topk 43
mix75_topk 44
mix75_topk 45
SIGMA_EOF_4
cat > slurm/sigma_chains_greedy.txt <<'SIGMA_EOF_5'
control_gpt2m_greedyeval 43
control_gpt2m_greedyeval 44
control_gpt2m_greedyeval 45
gpt2m_greedy 43
gpt2m_greedy 44
gpt2m_greedy 45
mix25_greedy 43
mix25_greedy 44
mix25_greedy 45
mix50_greedy 43
mix50_greedy 44
mix50_greedy 45
mix75_greedy 43
mix75_greedy 44
mix75_greedy 45
SIGMA_EOF_5
chmod +x src/sigma.py scripts/sigma_gate.py compute_sigma_thresholds.py
md5sum -c <<'MD5_EOF'
f8d4d18c644885545b6afaadc346603c  src/sigma.py
a3cb8d97d179982e533a0cd972c11b64  scripts/sigma_gate.py
2be2281e959e8b1dd05b0737c05acc47  compute_sigma_thresholds.py
6f59df79ec6e7187b3fb4d488ac6bd28  slurm/sigma_array.sbatch
dc88f1b83520afd4db1cfb8425d55b72  slurm/sigma_chains_topk.txt
3cdea3bb084faf5ea2c9b014e090d93b  slurm/sigma_chains_greedy.txt
MD5_EOF
bash -n slurm/sigma_array.sbatch && echo 'sbatch syntax OK'
python src/sigma.py --self-test
)
echo "install exit status: $?"
