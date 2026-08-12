#!/usr/bin/env python3
"""PPL-only re-score: in-domain + two OOD slices, merged into existing metrics.json.
Loads the saved gen-g model, computes 3 perplexities, leaves all other metrics intact.
Reuses perplexity() from metrics.py so numbers are identical to the main pipeline.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import load_config, set_seed, gen_dir, read_jsonl, device, offline_ok
from metrics import perplexity

PPL_SLICES = (
    ("perplexity",            "real_test.jsonl"),
    ("perplexity_ood_c4",     "ood_c4_test.jsonl"),
    ("perplexity_ood_ccnews", "ood_ccnews_test.jsonl"),
    ("perplexity_ood_wiki",   "ood_wiki_test.jsonl"),
)


@torch.no_grad()
def ppl_and_sharpness(model, tok, texts, block_size, batch_size=32):
    # Verbatim mirror of metrics.perplexity(), plus mean token entropy (nats)
    # and mean top-1 probability from the same logits. Same mask, same positions.
    nlls, ents, tops, ntok = [], [], [], 0
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
        for j in range(0, flat.size(0), 2048):
            sl = slice(j, min(j + 2048, flat.size(0)))
            mj = m[sl]
            if not mj.any():
                continue
            lp = torch.log_softmax(flat[sl][mj], dim=-1)
            ents.append((-(lp.exp() * lp).sum(-1)).sum().item())
            tops.append(lp.max(-1).values.exp().sum().item())
        ntok += mask.sum().item()
    return (float(np.exp(sum(nlls) / ntok)),
            float(sum(ents) / ntok),
            float(sum(tops) / ntok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    offline_ok()

    gd = gen_dir(cfg, args.generation, seed)
    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
    dev = device()
    bs = cfg["data"]["block_size"]

    tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        gd / "model", local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval()

    prevm = gd / "metrics.json"
    prev = json.loads(prevm.read_text()) if prevm.exists() else {}
    add = {}
    for tag, fname in PPL_SLICES:
        texts = [r["text"] for r in read_jsonl(root / fname)]
        print(f"{tag} ({len(texts)} chunks)...")
        ppl, ent, top1 = ppl_and_sharpness(model, tok, texts, bs)
        add[tag] = ppl
        add[tag.replace("perplexity", "entropy")] = ent
        add[tag.replace("perplexity", "top1_prob")] = top1
        if tag in prev:
            print(f"  PPL check: new {ppl:.6f} vs stored {prev[tag]:.6f} "
                  f"(delta {ppl - prev[tag]:+.6f})")
    del model; torch.cuda.empty_cache()

    out = gd / "metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(add)
    out.write_text(json.dumps(existing, indent=2))
    print(json.dumps(add, indent=2))


if __name__ == "__main__":
    main()
