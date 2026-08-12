"""Compute the five non-benchmark metrics for one generation:

  perplexity   — gen-g model on the REAL WikiText-2 test chunks   (Schaeffer defs 1-2)
  distinct2    — Distinct-2 on eval completions                    (def 4, surface)
  emb_variance — trace of SBERT embedding covariance               (def 4, semantic)
  kl_unigram   — KL(human-token-dist || model-token-dist)          (def 7, coverage)
  mauve        — MAUVE(model completions, human continuations)     (def 7, dist. shape)

MMLU/HellaSwag run separately via lm-evaluation-harness (slurm script).
Results append into gen{g}/metrics.json.
"""
import argparse
import json
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import load_config, set_seed, gen_dir, read_jsonl, device, offline_ok
from pathlib import Path


@torch.no_grad()
def perplexity(model, tok, texts, block_size, batch_size=32):
    """Mean PPL over fixed-size chunks (exp of mean token NLL)."""
    nlls, ntok = [], 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=block_size).to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        # recompute summed NLL properly per batch
        logits = out.logits[:, :-1]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            labels.reshape(-1), reduction="none").reshape(labels.shape)
        nlls.append(nll[mask].sum().item())
        ntok += mask.sum().item()
    return float(np.exp(sum(nlls) / ntok))


def distinct_n(texts, n=2):
    total, uniq = 0, set()
    for t in texts:
        toks = t.split()
        grams = list(zip(*[toks[i:] for i in range(n)]))
        total += len(grams)
        uniq.update(grams)
    return len(uniq) / max(total, 1)


def embedding_variance(texts, sbert_name, dev):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(sbert_name, device=dev)
    emb = m.encode(texts, batch_size=128, show_progress_bar=False,
                   normalize_embeddings=True)
    return float(np.trace(np.cov(emb.T)))


def kl_unigram(human_texts, model_texts, tok, eps=1e-9):
    """KL(P_human || Q_model) over the tokenizer vocab, additive smoothing."""
    def dist(texts):
        c = Counter()
        for t in texts:
            c.update(tok(t, return_tensors=None)["input_ids"])
        v = np.full(len(tok), eps)
        for k, n in c.items():
            v[k] += n
        return v / v.sum()
    p, q = dist(human_texts), dist(model_texts)
    return float(np.sum(p * np.log(p / q)))


def mauve_score(model_texts, human_texts, featurizer, dev_id):
    import mauve
    out = mauve.compute_mauve(
        p_text=human_texts, q_text=model_texts,
        featurize_model_name=featurizer, device_id=dev_id,
        max_text_length=256, verbose=False)
    return float(out.mauve)


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

    tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        gd / "model", local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval()

    test_texts = [r["text"] for r in read_jsonl(root / "real_test.jsonl")]
    comp = read_jsonl(gd / "eval_completions.jsonl")
    model_texts = [r["model"] for r in comp]
    human_texts = [r["human"] for r in comp]

    results = {"generation": args.generation, "seed": seed}
    bs = cfg["data"]["block_size"]
    print("perplexity (in-domain WikiText-2)..."); results["perplexity"] = perplexity(
        model, tok, test_texts, bs)

    # Held-out OOD perplexity: breadth-narrowing controls (frozen slices, all gens/seeds)
    for tag, fname in (("perplexity_ood_c4", "ood_c4_test.jsonl"),
                       ("perplexity_ood_ccnews", "ood_ccnews_test.jsonl")):
        ood_texts = [r["text"] for r in read_jsonl(root / fname)]
        print(f"{tag}..."); results[tag] = perplexity(model, tok, ood_texts, bs)

    del model; torch.cuda.empty_cache()

    print("distinct-2..."); results["distinct2"] = distinct_n(model_texts, 2)
    print("embedding variance..."); results["emb_variance"] = embedding_variance(
        model_texts, cfg["eval"]["sbert_model"], dev)
    print("KL unigram..."); results["kl_unigram"] = kl_unigram(
        human_texts, model_texts, tok)
    print("MAUVE..."); results["mauve"] = mauve_score(
        model_texts, human_texts, cfg["eval"]["mauve_featurizer"],
        0 if dev == "cuda" else -1)

    out = gd / "metrics.json"
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing.update(results)
    out.write_text(json.dumps(existing, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
