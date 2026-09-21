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


def pilot(root, sroot, device, arm="control_gpt2m", seed=43, gen=1, config=None):
    cell = root / f"{arm}/seed{seed}/gen{gen}"
    texts = [json.loads(l)["model"] for l in open(cell / "eval_completions.jsonl")]
    import yaml
    cfg = yaml.safe_load(open(HERE / f"configs/{config or arm}.yaml"))
    name = cfg["eval"]["sbert_model"]
    same = name.split("/")[-1] == sigma.ENCODER.split("/")[-1]
    print(f"cfg sbert_model = {name}; sigma encoder = {sigma.ENCODER}; same = {same}")
    from metrics import embedding_variance
    stored = json.load(open(cell / "metrics.json"))["emb_variance"]
    got = embedding_variance(texts, name, device)
    res("K1 emb_variance reproduces stored value", abs(got - stored) <= 1e-4 * max(1, abs(stored)),
        f"stored={stored:.6f} recomputed={got:.6f}")
    cache = sroot / f"{arm}/seed{seed}_gen{gen}_emb.npy"
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
    ap.add_argument("--arm", default="control_gpt2m"); ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--gen", type=int, default=1); ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    root = Path(a.root); sroot = root / "sigma"
    if a.pilot:
        pilot(root, sroot, a.device, a.arm, a.seed, a.gen, a.config)
    if a.all:
        all_chains(root, sroot, a.all)
    if not (a.pilot or a.all):
        ap.error("--pilot and/or --all")
    print("GATE", "PASS" if fails == 0 else f"FAIL ({fails})")
    sys.exit(1 if fails else 0)
