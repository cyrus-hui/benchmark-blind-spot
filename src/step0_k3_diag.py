#!/usr/bin/env python
"""K3 diagnosis for 7.1 step 0 (pilot 3760319 FAIL_K3, rate 0.96387).
Pipeline facts only: argmax agreement, ranks, top1-top2 logit gaps, positions,
round-trip. Computes NO S and prints nothing derived from S.
Hypotheses: (a) dtype near-tie flips -> isolated mismatches, tiny gaps, rank 2,
fixed under the generation dtype; (b) re-tokenisation drift -> seam-front-loaded
or clustered, round-trip failures; (c) EOS/pad after end-of-text -> clustered
in docs containing token 50256; (d) a logits processor in generation_config.
"""
import argparse, json, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT, BLOCK = 32, 128
ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=0); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--dtypes", default="float32,bfloat16,float16")
a = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(a.model)
texts = [json.loads(l)["text"] for l in open(a.corpus) if l.strip()]
if a.n: texts = texts[:a.n]
ids, rt_ok, n_len = [], 0, 0
for t in texts:
    x = tok(t, add_special_tokens=False)["input_ids"]
    rt_ok += tok.decode(x) == t
    if len(x) == BLOCK: ids.append(x); n_len += 1
X = torch.tensor(ids)
eos = tok.eos_token_id
has_eos = (X == eos).any(1).numpy() if eos is not None else np.zeros(len(X), bool)
print(f"rows {len(texts)}  len128 {n_len}  decode(encode(text))==text {rt_ok}/{len(texts)}")
print(f"docs containing eos({eos}) anywhere {int(has_eos.sum())}; in scored span "
      f"{int((X[:, PROMPT:] == eos).any(1).sum()) if eos is not None else 0}; literal '<|endoftext|>' in text "
      f"{sum('<|endoftext|>' in t for t in texts)}")
print("generation_config:", json.dumps(json.load(open(a.model + '/generation_config.json'))))

for dt in a.dtypes.split(","):
    try:
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, dt)).to(dev).eval()
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, dt)).to(dev).eval()
    M, G, R = [], [], []
    with torch.inference_mode():
        for i in range(0, len(X), a.batch):
            x = X[i:i + a.batch].to(dev)
            lg = m(x).logits[:, PROMPT - 1:BLOCK - 1].float(); tg = x[:, PROMPT:]
            top2 = lg.topk(2, -1).values
            M.append((lg.argmax(-1) == tg).cpu()); G.append((top2[..., 0] - top2[..., 1]).cpu())
            R.append((lg > lg.gather(-1, tg.unsqueeze(-1))).sum(-1).cpu() + 1)
    M, G, R = torch.cat(M).numpy(), torch.cat(G).numpy(), torch.cat(R).numpy()
    mm = ~M; per_doc = mm.sum(1)
    print(f"\n== {dt}: argmax_match_rate {M.mean():.5f}  mismatches {int(mm.sum())}")
    if dt != "float32" and not mm.sum(): continue
    q = lambda v: " ".join(f"{p}%={np.percentile(v, p):.4g}" for p in (10, 50, 90, 99)) if len(v) else "-"
    print(f"  docs by #mismatch: 0:{(per_doc==0).sum()} 1:{(per_doc==1).sum()} 2-5:{((per_doc>=2)&(per_doc<=5)).sum()} "
          f"6-20:{((per_doc>=6)&(per_doc<=20)).sum()} >20:{(per_doc>20).sum()}")
    srt = np.sort(per_doc)[::-1]
    print(f"  share of mismatches in top 1% / 5% docs: {srt[:max(1,len(srt)//100)].sum()/max(1,mm.sum()):.3f} / "
          f"{srt[:max(1,len(srt)//20)].sum()/max(1,mm.sum()):.3f}")
    pos = mm.sum(0)
    print("  mismatch rate by position bin (1-indexed 33..128, bins of 8): " +
          " ".join(f"{pos[b:b+8].sum()/(8*len(M)):.4f}" for b in range(0, 96, 8)))
    print(f"  first scored position (33) mismatch rate {pos[0]/len(M):.4f}")
    print(f"  rank of stored token at mismatches: r2 {(R[mm]==2).mean():.3f}  r3-5 {((R[mm]>=3)&(R[mm]<=5)).mean():.3f}  "
          f">5 {(R[mm]>5).mean():.3f}")
    print(f"  top1-top2 gap at mismatches: {q(G[mm])}")
    print(f"  top1-top2 gap at matches   : {q(G[M])}")
    iso = mm.copy(); iso[:, 1:] &= ~mm[:, :-1]; iso[:, :-1] &= ~mm[:, 1:]
    print(f"  isolated mismatches (neighbours match): {iso.sum()/max(1,mm.sum()):.3f}")
    if has_eos.any():
        print(f"  mismatch rate in docs with eos {mm[has_eos].mean():.4f} vs without {mm[~has_eos].mean():.4f}")
    del m; torch.cuda.empty_cache() if dev == "cuda" else None
