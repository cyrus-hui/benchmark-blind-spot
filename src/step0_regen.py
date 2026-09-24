#!/usr/bin/env python
"""K3' regeneration test for 7.1 step 0 (after pilot 3760319 FAIL_K3 at 0.96387).
Re-runs generate.py's own corpus-mode greedy call (imported, not reimplemented) in
bf16 on the first n_batches*batch_size prompts, in corpus order and with the same
batch composition, and compares with the stored synthetic_corpus.jsonl.
Pipeline facts only; computes NO S.
Registered reading (Sept 23, before running): primary = fraction of generated
positions before the first divergence, pooled; >= 0.99 -> stored text is the
generator's output and this becomes K3'. Below 0.99: first divergences at fp32
near-ties -> numerics not reproducible in the current stack; at large gaps -> stop.
"""
import argparse, os, sys
from pathlib import Path
import numpy as np, torch, transformers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import load_config, gen_dir, read_jsonl, device
from generate import build_gen_kwargs, batched_generate

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/gpt2m_greedy.yaml")
ap.add_argument("--seed", type=int, default=43)
ap.add_argument("--generation", type=int, default=0)
ap.add_argument("--n-batches", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=64)
a = ap.parse_args()
print(f"transformers {transformers.__version__} torch {torch.__version__} device {device()}")
cfg = load_config(a.config)
assert cfg["decoding"]["strategy"] == "greedy", "regeneration test is for greedy corpora only"
gd = gen_dir(cfg, a.generation, a.seed)
root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
src_path = root / "real_train.jsonl" if a.generation == 0 else gen_dir(cfg, a.generation - 1, a.seed) / "synthetic_corpus.jsonl"
src, stored = read_jsonl(src_path), read_jsonl(gd / "synthetic_corpus.jsonl")
print(f"prompt source {src_path} ({len(src)} rows); stored {gd / 'synthetic_corpus.jsonl'} ({len(stored)} rows)")
assert len(src) == len(stored), "row count mismatch between prompt source and stored corpus"

tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True, padding_side="left")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
pl = cfg["data"]["prompt_len"]
prompts = [tok.decode(tok(r["text"], return_tensors=None)["input_ids"][:pl]) for r in src]
pref = sum(s["text"].startswith(p) for s, p in zip(stored, prompts))
print(f"stored rows starting with reconstructed prompt: {pref}/{len(stored)}")
n = min(a.n_batches * a.batch_size, len(prompts))
plen = [len(tok(p)["input_ids"]) for p in prompts[:n]]
print(f"sample n={n}; prompts re-encoding to {pl} tokens: {sum(x == pl for x in plen)}/{n} "
      f"(batches with left padding: {sum(len(set(plen[i:i+a.batch_size])) > 1 for i in range(0, n, a.batch_size))})")

model = AutoModelForCausalLM.from_pretrained(gd / "model", local_files_only=True,
                                             torch_dtype=torch.bfloat16).to(device()).eval()
conts = batched_generate(model, tok, prompts[:n], build_gen_kwargs(cfg), a.batch_size)
del model
stored_c = [stored[i]["text"][len(prompts[i]):] if stored[i]["text"].startswith(prompts[i]) else None for i in range(n)]
exact = [s is not None and s == c for s, c in zip(stored_c, conts)]
print(f"\ndocs reproduced exactly (string): {sum(exact)}/{n}")

m32 = AutoModelForCausalLM.from_pretrained(gd / "model", local_files_only=True,
                                           torch_dtype=torch.float32).to(device()).eval()
torch.backends.cuda.matmul.allow_tf32 = False
K, gaps, top2 = [], [], []
for i in range(n):
    if exact[i]:
        K.append(96); continue
    if stored_c[i] is None:
        K.append(0); continue
    s_ids, r_ids = tok(stored_c[i])["input_ids"], tok(conts[i])["input_ids"]
    k = next((j for j in range(min(len(s_ids), len(r_ids))) if s_ids[j] != r_ids[j]), min(len(s_ids), len(r_ids)))
    K.append(min(k, 96))
    if k < min(len(s_ids), len(r_ids)):
        ctx = torch.tensor([tok(prompts[i])["input_ids"] + s_ids[:k]], device=m32.device)
        with torch.inference_mode():
            lg = m32(ctx).logits[0, -1].float()
        v, ix = lg.topk(2)
        gaps.append((v[0] - v[1]).item())
        top2.append({s_ids[k], r_ids[k]} == set(ix.tolist()))
K = np.array(K)
print(f"PRIMARY positions before first divergence (pooled): {K.sum() / (96 * n):.5f}")
print(f"first-divergence position (1..96) among non-exact: "
      + (" ".join(f"{p}%={np.percentile(K[K < 96] + 1, p):.0f}" for p in (10, 50, 90)) if (K < 96).any() else "-"))
if gaps:
    g = np.array(gaps)
    print(f"fp32 top1-top2 gap at first divergence: " + " ".join(f"{p}%={np.percentile(g, p):.4g}" for p in (10, 50, 90, 99)))
    print(f"stored and regenerated tokens are exactly the fp32 top-2 pair: {np.mean(top2):.3f} ({len(top2)} divergences)")
