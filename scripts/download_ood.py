#!/usr/bin/env python3
"""Download + freeze two OOD eval slices for PPL breadth-narrowing analysis.
RUN ON LOGIN NODE ONLY (needs internet). Writes fixed .jsonl slices to the
run root, matching real_test.jsonl format: one {"text": ...} per line.
"""
import argparse, json, re
from pathlib import Path
from datasets import load_dataset

TARGET_CHUNKS = 2000      # symmetry with MAUVE eval set
MIN_WORDS     = 110       # ~150-170 tokens > block_size=128, truncation matches in-domain
SEED          = 20260621  # fixed shuffle seed; freeze provenance


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def chunkify(stream, text_field, target, min_words):
    out = []
    for ex in stream:
        words = clean(ex[text_field]).split()
        for i in range(0, len(words), min_words):
            w = words[i:i + min_words]
            if len(w) >= min_words:
                out.append(" ".join(w))
            if len(out) >= target:
                return out
    return out


def write(chunks, path):
    with open(path, "w") as f:
        for c in chunks:
            f.write(json.dumps({"text": c}) + "\n")
    print(f"wrote {len(chunks)} chunks -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    print("loading C4 en/validation (streaming)...")
    c4 = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    c4 = c4.shuffle(seed=SEED, buffer_size=10000)
    write(chunkify(c4, "text", TARGET_CHUNKS, MIN_WORDS), root / "ood_c4_test.jsonl")

    print("loading CC-News...")
    try:
        cc = load_dataset("vblagoje/cc_news", split="train", streaming=True)
        field = "text"
    except Exception as e:
        print(f"vblagoje/cc_news failed ({e}); trying cc_news...")
        cc = load_dataset("cc_news", split="train", streaming=True)
        field = "text"
    cc = cc.shuffle(seed=SEED, buffer_size=10000)
    write(chunkify(cc, field, TARGET_CHUNKS, MIN_WORDS), root / "ood_ccnews_test.jsonl")

    print("done. snapshot frozen — do not re-run.")


if __name__ == "__main__":
    main()
