#!/usr/bin/env python3
"""
download_ood_wiki.py  --  REGISTER-MATCHED OOD slice (login-node only, run ONCE).

Builds a held-out, same-register, article-DISJOINT Wikipedia slice to sit alongside
the existing far-register OOD corpora (C4, CC-News). Fills the missing cell in the
2x2: {same register, different register} x {in training distribution, held out}.
This slice is the perplexity analogue of the benchmarks (same register, never trained
on) and is the decisive test for the register-alignment reading of the rising benchmarks.

Source = WikiText-103 minus every article that appears in WikiText-2 (WT-2 is a subset
of WT-103). Same Merity et al. Good/Featured-Wikipedia register as in-domain; disjoint
at the article level; passage-level leakage checked against WT-2 train via 13-grams.

  *** VERIFY BEFORE RUNNING (one line) ***
  The slice MUST match the in-domain corpus's tokenization, or the PPL gap measures
  formatting, not register. Check which form real_train.jsonl is in:

      grep -m1 -c '@-@' $SCRATCH/collapse/pilot_gpt2m/real_train.jsonl

  >0  -> in-domain is the TOKENIZED wikitext (has @-@ / @.@ markers): use --cfg v1
   0  -> in-domain is the RAW wikitext: use --cfg raw
  Default below is v1 (tokenized), matching the @-@/@.@ note in decisions.md. Override
  with --cfg raw if the grep returns 0.

Usage (login node, offline flags overridden for this one command):
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 HF_DATASETS_TRUST_REMOTE_CODE=1 \
      python download_ood_wiki.py --cfg v1 --out pilot_gpt2m/ood_wiki_test.jsonl
  # then symlink into smollm2_1.7b/ exactly like real_test.jsonl / ood_c4_test.jsonl

Self-test (no network; validates the article parser + filter logic):
  python download_ood_wiki.py --selftest
"""
import argparse, json, os, re, random, sys

MIN_WORDS   = 110            # matches in-domain real_test.jsonl chunks (~114 words)
N_CHUNKS    = 2000           # symmetry with C4 / CC-News / MAUVE eval set
SHUFFLE_SEED = 20260621      # SAME seed as download_ood.py -- do not change
LEAK_N      = 13             # n-gram order for passage-level leakage check

# Level-1 article header in HF wikitext: " = Title = " (single =, NOT " = = ... = = ").
H1 = re.compile(r"^ = ([^=].*?) = $")
H_ANY = re.compile(r"^ (=+) .* \1 $")   # any-level header (to skip section heads)

def norm_title(t):
    return re.sub(r"\s+", " ", t).strip().lower()

def split_articles(lines):
    """Yield (title, body_text) for each level-1 article in a wikitext line stream."""
    title, buf = None, []
    for ln in lines:
        m = H1.match(ln)
        if m:                                   # new article boundary
            if title is not None:
                yield title, "\n".join(buf)
            title, buf = m.group(1), []
        else:
            if title is not None:
                buf.append(ln)
    if title is not None:
        yield title, "\n".join(buf)

def chunkify(text, min_words):
    """Whitespace windows of >= min_words words (same scheme as download_ood.py)."""
    words = text.split()
    out, i = [], 0
    while i + min_words <= len(words):
        out.append(" ".join(words[i:i + min_words]))
        i += min_words
    return out

def ngrams(words, n):
    return {tuple(words[i:i+n]) for i in range(len(words) - n + 1)}

def selftest():
    mini = [
        " = Alpha Article = ", "", " = = Section = = ", "the quick brown fox " * 40, "",
        " = Beta Article = ", "lorem ipsum dolor sit amet " * 30, "",
        " = Gamma = ", "held out prose goes here " * 50, "",
    ]
    arts = dict((norm_title(t), b) for t, b in split_articles(mini))
    assert set(arts) == {"alpha article", "beta article", "gamma"}, arts.keys()
    # exclude WT-2-like set {alpha, beta} -> only gamma survives
    wt2 = {"alpha article", "beta article"}
    kept = [t for t in arts if t not in wt2]
    assert kept == ["gamma"], kept
    chunks = chunkify(arts["gamma"], MIN_WORDS)
    assert all(len(c.split()) >= MIN_WORDS for c in chunks)
    # leakage: a 13-gram from a kept chunk must not be in the wt2 13-gram set
    wt2_grams = set()
    for t in wt2:
        wt2_grams |= ngrams(arts[t].split(), LEAK_N)
    leaks = [c for c in chunks if ngrams(c.split(), LEAK_N) & wt2_grams]
    assert leaks == [], "unexpected leak in selftest"
    print("selftest OK: parser segments level-1 articles, title filter + leak check work.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=["v1", "raw"], default="v1",
                    help="v1 = tokenized wikitext (@-@ markers); raw = raw text. MUST match in-domain.")
    ap.add_argument("--out", default="pilot_gpt2m/ood_wiki_test.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from datasets import load_dataset
    suffix = "v1" if args.cfg == "v1" else "raw-v1"

    # 1. Collect ALL WikiText-2 article titles (train+val+test) -> the exclusion set.
    wt2_titles, wt2_train_text = set(), []
    for split in ("train", "validation", "test"):
        ds = load_dataset("Salesforce/wikitext", f"wikitext-2-{suffix}", split=split)
        for t, b in split_articles(r["text"] for r in ds):
            wt2_titles.add(norm_title(t))
            if split == "train":
                wt2_train_text.append(b)
    print(f"WikiText-2 articles to exclude: {len(wt2_titles)}")

    # 2. Build the WT-2-train 13-gram set for passage-level leakage screening.
    wt2_grams = set()
    for b in wt2_train_text:
        wt2_grams |= ngrams(b.split(), LEAK_N)
    print(f"WT-2 train {LEAK_N}-grams for leak screen: {len(wt2_grams):,}")

    # 3. Stream WikiText-103 train, keep only title-disjoint articles.
    ds103 = load_dataset("Salesforce/wikitext", f"wikitext-103-{suffix}", split="train")
    kept_chunks, n_excluded, n_arts = [], 0, 0
    for t, body in split_articles(r["text"] for r in ds103):
        n_arts += 1
        if norm_title(t) in wt2_titles:
            n_excluded += 1
            continue
        kept_chunks.extend(chunkify(body, MIN_WORDS))
    print(f"WT-103 articles: {n_arts} | excluded as WT-2 overlap: {n_excluded} | "
          f"candidate chunks: {len(kept_chunks):,}")

    # 4. Passage-level leakage screen (drop any chunk sharing a 13-gram with WT-2 train).
    clean = [c for c in kept_chunks if not (ngrams(c.split(), LEAK_N) & wt2_grams)]
    dropped = len(kept_chunks) - len(clean)
    print(f"chunks dropped by {LEAK_N}-gram leak screen: {dropped} "
          f"({100*dropped/max(len(kept_chunks),1):.2f}%)")
    if len(clean) < N_CHUNKS:
        sys.exit(f"ERROR: only {len(clean)} clean chunks, need {N_CHUNKS}.")

    # 5. Shuffle with the frozen seed, take N_CHUNKS. SNAPSHOT ONCE.
    random.seed(SHUFFLE_SEED)
    random.shuffle(clean)
    sample = clean[:N_CHUNKS]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        for c in sample:
            fh.write(json.dumps({"text": c}) + "\n")
    print(f"wrote {len(sample)} chunks -> {args.out}")
    print("NEXT: symlink into smollm2_1.7b/ (mirror real_test.jsonl); snapshot is frozen.")

if __name__ == "__main__":
    main()
