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
