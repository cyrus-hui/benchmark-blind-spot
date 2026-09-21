#!/usr/bin/env python3
"""Phase 5.2 negative control: seven fresh, article-disjoint REAL training slices.

Each slice is chunked exactly like the in-domain gen-0 corpus (data_prep.chunk_split,
model tokenizer, block_size), holds exactly len(real_train.jsonl) chunks, and comes from
WikiText-103-raw train with (a) every WikiText-2 article removed (all splits), (b) any
article sharing a 13-gram with WikiText-2 train removed, (c) any article sharing a 13-gram
with ood_wiki_test.jsonl removed, so the control never trains on its own held-out wiki
slice. Articles are never split across slices. Run ONCE on the login node:

  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python build_fresh_slices.py \
      --config configs/pilot_gpt2m.yaml --n_slices 7 \
      --eval_wiki $SCRATCH/collapse/pilot_gpt2m/ood_wiki_test.jsonl \
      --out_dir $SCRATCH/collapse/fresh_slices_gpt2m
"""
import argparse, hashlib, json, os, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import chunk_split                      # exact in-domain chunking
from download_ood_wiki import split_articles, norm_title, ngrams, LEAK_N
from util import load_config, read_jsonl

SHUFFLE_SEED = 20260912

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_slices", type=int, default=7)
    ap.add_argument("--eval_wiki", required=True, help="ood_wiki_test.jsonl to screen against")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    cfg = load_config(args.config)
    bs = cfg["data"]["block_size"]
    assert cfg["data"]["dataset_config"] == "wikitext-2-raw-v1", cfg["data"]["dataset_config"]
    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_name"])

    root = Path(os.path.expandvars(cfg["paths"]["workdir"])) / cfg["run_name"]
    n_target = sum(1 for _ in open(root / "real_train.jsonl"))
    print(f"target chunks per slice = {n_target} (from {root/'real_train.jsonl'})")

    # exclusion sets
    wt2_titles, wt2_train_bodies = set(), []
    for split in ("train", "validation", "test"):
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        for t, b in split_articles(r["text"] for r in ds):
            wt2_titles.add(norm_title(t))
            if split == "train":
                wt2_train_bodies.append(b)
    ban = set()
    for b in wt2_train_bodies:
        ban |= ngrams(b.split(), LEAK_N)
    n_wt2 = len(ban)
    for r in read_jsonl(args.eval_wiki):
        ban |= ngrams(r["text"].split(), LEAK_N)
    print(f"exclude titles: {len(wt2_titles)} | banned {LEAK_N}-grams: WT-2 train {n_wt2:,} "
          f"+ eval wiki -> {len(ban):,}")

    # candidate articles (title-disjoint, leak-clean at article level)
    ds103 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    arts, seen, n_all, n_title, n_leak, n_dup = [], set(), 0, 0, 0, 0
    for t, body in split_articles(r["text"] for r in ds103):
        n_all += 1
        nt = norm_title(t)
        if nt in wt2_titles:
            n_title += 1; continue
        if nt in seen:                      # WT-103 repeats some level-1 titles
            n_dup += 1; continue
        if ngrams(body.split(), LEAK_N) & ban:
            n_leak += 1; continue
        seen.add(nt); arts.append((t, body))
    print(f"WT-103 articles {n_all} | title-excluded {n_title} | duplicate titles {n_dup} | "
          f"leak-excluded {n_leak} | kept {len(arts)}")

    random.seed(SHUFFLE_SEED)
    random.shuffle(arts)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": args.config, "block_size": bs, "tokenizer": cfg["model"]["base_name"],
                "chunks_per_slice": n_target, "shuffle_seed": SHUFFLE_SEED,
                "eval_wiki_screened": args.eval_wiki, "slices": []}
    used, ai = set(), 0
    for g in range(1, args.n_slices + 1):
        chunks, titles = [], []
        while len(chunks) < n_target:
            if ai >= len(arts):
                sys.exit(f"ERROR: ran out of articles filling slice {g}")
            t, body = arts[ai]; ai += 1
            assert norm_title(t) not in used
            used.add(norm_title(t)); titles.append(t)
            chunks.extend(chunk_split(body.split("\n"), tok, bs))
        chunks = chunks[:n_target]
        path = out / f"gen{g}_real.jsonl"
        with open(path, "w") as fh:
            for c in chunks:
                fh.write(json.dumps({"text": c}) + "\n")
        # re-tokenization check: report how many chunks round-trip to exactly bs tokens
        exact = sum(1 for c in chunks[:500] if len(tok(c)["input_ids"]) == bs)
        entry = {"gen": g, "path": str(path), "n_chunks": len(chunks), "n_articles": len(titles),
                 "md5": md5(path), "roundtrip_exact_of_500": exact, "titles": titles}
        manifest["slices"].append(entry)
        print(f"slice {g}: {len(chunks)} chunks from {len(titles)} articles, md5 {entry['md5']}, "
              f"round-trip exact {exact}/500")
    assert len(used) == sum(e["n_articles"] for e in manifest["slices"])   # articles disjoint
    with open(out / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"manifest -> {out/'manifest.json'}; {len(arts) - ai} articles unused")

if __name__ == "__main__":
    main()
