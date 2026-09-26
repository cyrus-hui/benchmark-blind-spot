#!/usr/bin/env python3
"""M1 (decisions §41, spec §61): eval-side per-document decomposition of Δ_cal on real_test.

For the gen-g model of one (run, seed), score real_test.jsonl with the verbatim
ppl_and_sharpness() loop (same batching, padding, bf16 forward, fp32 logits, mask) and
additionally keep, per document d: n_d (masked tokens), NLL_d and H_d (summed nats),
TOP1_d (summed top-1 prob).  Writes one npz per cell.  Prints NO statistic of Δ_d.

Canary, before anything is written (exit 3, nothing written, on any failure):
  A  n_docs > 0, N > 0, every n_d > 0, sum n_d == N
  B  verbatim totals reproduce metrics.json: |log ppl - log ppl_stored| and
     |H - H_stored| both <= TOL_STORED
  C  bookkeeping: per-document float64 sums reproduce the verbatim totals
     (|Σ n_d·(nll_d - h_d)/N - (log ppl - H)| <= TOL_IDENT)
--check: paths, stored keys, real_test md5 only; no model load (login node).
"""
import argparse, hashlib, json, math, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from util import load_config, read_jsonl  # noqa: E402

REAL_TEST_MD5 = "c5ba60eab60aab3e7d9aa3ed93123a2c"
TOL_STORED = 1e-4   # nats; stored values were produced by the same loop on this cluster
TOL_IDENT = 1e-6    # nats; fp32-vs-fp64 summation order only
OUT_ROOT = Path("/scratch/cyrh/collapse/m1")


def die(msg, code=2):
    print(f"REFUSE: {msg}", flush=True)
    sys.exit(code)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_paths(cfg, seed, g):
    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
    gd = root / f"seed{seed}" / f"gen{g}"          # gen_dir() without its mkdir
    return root, gd


def score(model, tok, texts, block_size, batch_size=32):
    import torch
    nlls, ents, ntok = [], [], 0                    # verbatim aggregates
    n_d, nll_d, h_d, top_d = [], [], [], []         # per-document, float64
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=block_size).to(model.device)
            out = model(**enc, labels=enc["input_ids"])
            logits = out.logits[:, :-1]
            labels = enc["input_ids"][:, 1:]
            mask = enc["attention_mask"][:, 1:].bool()
            flat = logits.reshape(-1, logits.size(-1)).float()
            nll = torch.nn.functional.cross_entropy(
                flat, labels.reshape(-1), reduction="none").reshape(labels.shape)
            nlls.append(nll[mask].sum().item())
            m = mask.reshape(-1)
            ent_pos = torch.zeros(m.numel(), dtype=torch.float32, device=flat.device)
            top_pos = torch.zeros_like(ent_pos)
            for j in range(0, flat.size(0), 2048):
                sl = slice(j, min(j + 2048, flat.size(0)))
                mj = m[sl]
                if not mj.any():
                    continue
                lp = torch.log_softmax(flat[sl][mj], dim=-1)
                e = -(lp.exp() * lp).sum(-1)
                ents.append(e.sum().item())
                idx = torch.arange(sl.start, sl.stop, device=flat.device)[mj]
                ent_pos[idx] = e
                top_pos[idx] = lp.max(-1).values.exp()
            ntok += mask.sum().item()
            md = mask.double()
            n_d += mask.sum(1).tolist()
            nll_d += (nll.double() * md).sum(1).tolist()
            h_d += (ent_pos.reshape(mask.shape).double() * md).sum(1).tolist()
            top_d += (top_pos.reshape(mask.shape).double() * md).sum(1).tolist()
    return (float(np.exp(sum(nlls) / ntok)), float(sum(ents) / ntok), ntok,
            np.array(n_d, np.int64), np.array(nll_d), np.array(h_d), np.array(top_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gens", type=int, nargs="+", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    bs = cfg["data"]["block_size"]
    for g in a.gens:
        root, gd = cell_paths(cfg, a.seed, g)
        rt = root / "real_test.jsonl"
        if not rt.is_file():
            die(f"missing {rt}")
        if md5(rt) != REAL_TEST_MD5:
            die(f"{rt} md5 != {REAL_TEST_MD5}")
        mp = gd / "metrics.json"
        if not mp.is_file():
            die(f"missing {mp}")
        st = json.load(open(mp))
        for k in ("perplexity", "entropy"):
            if k not in st:
                die(f"key {k} absent in {mp}")
        if not (gd / "model").is_dir():
            die(f"missing {gd / 'model'}")
        outp = OUT_ROOT / cfg["run_name"] / f"seed{a.seed}" / f"gen{g}.npz"
        if outp.exists():
            die(f"{outp} exists; refusing to overwrite")
    texts = [r["text"] for r in read_jsonl(rt)]
    print(f"M1 {cfg['run_name']} seed{a.seed} gens {a.gens} block_size {bs} "
          f"real_test {len(texts)} docs md5 ok  script md5 {md5(__file__)}", flush=True)
    if a.check:
        print("CHECK PASS — no model loaded, nothing scored")
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for g in a.gens:
        root, gd = cell_paths(cfg, a.seed, g)
        st = json.load(open(gd / "metrics.json"))
        tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            gd / "model", local_files_only=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        ppl, ent, N, n_d, nll_d, h_d, top_d = score(model, tok, texts, bs)
        del model
        torch.cuda.empty_cache()

        # ---- canary A
        if not (len(n_d) == len(texts) > 0 and N > 0 and (n_d > 0).all() and int(n_d.sum()) == N):
            die(f"gen{g} canary A: docs {len(n_d)}/{len(texts)} N {N} "
                f"min n_d {n_d.min() if len(n_d) else 'NA'} sum {int(n_d.sum())}", 3)
        # ---- canary B
        dlp = abs(math.log(ppl) - math.log(st["perplexity"]))
        dh = abs(ent - st["entropy"])
        print(f"gen{g} canary B: |dlogppl| {dlp:.2e}  |dH| {dh:.2e}  (tol {TOL_STORED:g})", flush=True)
        if not (dlp <= TOL_STORED and dh <= TOL_STORED):
            die(f"gen{g} canary B fired", 3)
        # ---- canary C
        di = abs((nll_d - h_d).sum() / N - (math.log(ppl) - ent))
        print(f"gen{g} canary C: |identity| {di:.2e}  (tol {TOL_IDENT:g})", flush=True)
        if not di <= TOL_IDENT:
            die(f"gen{g} canary C fired", 3)

        outp = OUT_ROOT / cfg["run_name"] / f"seed{a.seed}" / f"gen{g}.npz"
        outp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(outp, n=n_d, nll=nll_d, h=h_d, top1=top_d)
        json.dump({"run": cfg["run_name"], "seed": a.seed, "gen": g, "block_size": bs,
                   "real_test_md5": REAL_TEST_MD5, "script_md5": md5(__file__),
                   "canary_B": [dlp, dh], "canary_C": di,
                   "job": os.environ.get("SLURM_JOB_ID")},
                  open(outp.with_suffix(".json"), "w"), indent=1)
        print(f"gen{g} written {outp.name} ({len(n_d)} docs)", flush=True)
    print("M1 SCORE COMPLETE")


if __name__ == "__main__":
    main()
