"""Prepare WikiText-2: chunk train split into block_size-token texts (the gen-0 'real'
training corpus) and build a FIXED evaluation prompt set from the test split.

The eval prompt set is created once and reused identically across all generations and
seeds — this is what makes per-generation metric comparisons valid.

Run ONCE per config (login node is fine; CPU only, dataset must already be cached).
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from util import load_config, set_seed, write_jsonl, gen_dir
from pathlib import Path


def chunk_split(texts, tokenizer, block_size):
    """Concatenate non-empty lines and slice into exact block_size token chunks."""
    joined = "\n".join(t for t in texts if t.strip())
    ids = tokenizer(joined, return_tensors=None)["input_ids"]
    chunks = []
    for i in range(0, len(ids) - block_size + 1, block_size):
        chunks.append(tokenizer.decode(ids[i:i + block_size]))
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_name"])
    ds = load_dataset(cfg["data"]["dataset_name"], cfg["data"]["dataset_config"])
    bs = cfg["data"]["block_size"]

    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
    root.mkdir(parents=True, exist_ok=True)

    # Real training corpus (gen 0 trains on this)
    train_chunks = chunk_split(ds["train"]["text"], tok, bs)
    write_jsonl(root / "real_train.jsonl", [{"text": c} for c in train_chunks])
    print(f"real_train.jsonl: {len(train_chunks)} chunks of {bs} tokens")

    # Real test corpus (perplexity target)
    test_chunks = chunk_split(ds["test"]["text"], tok, bs)
    write_jsonl(root / "real_test.jsonl", [{"text": c} for c in test_chunks])
    print(f"real_test.jsonl: {len(test_chunks)} chunks")

    # Fixed eval prompt set: prompt_len-token prefixes from test chunks + the human
    # continuation (MAUVE / KL reference). Identical across all generations & seeds.
    n = min(cfg["eval"]["n_eval_prompts"], len(test_chunks))
    pl = cfg["data"]["prompt_len"]
    rows = []
    for c in test_chunks[:n]:
        ids = tok(c, return_tensors=None)["input_ids"]
        rows.append({
            "prompt": tok.decode(ids[:pl]),
            "human_continuation": tok.decode(ids[pl:]),
        })
    write_jsonl(root / "eval_prompts.jsonl", rows)
    print(f"eval_prompts.jsonl: {len(rows)} fixed prompts (prefix={pl} tokens)")


if __name__ == "__main__":
    main()
