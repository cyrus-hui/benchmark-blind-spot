#!/usr/bin/env python3
"""Build a ratio-mixed training corpus for Phase 5.3.

Holds total chunk count constant at every ratio so optimizer steps are
constant across the grid: severity is attributable to composition, not to
training amount. Sampling is line-level and therefore schema-agnostic.

  n_synth = ceil(ratio * N)      from synthetic_corpus.jsonl of generation g-1
  n_real  = N - n_synth          from the frozen fresh real slice g (5.2's)
"""
import argparse, hashlib, json, math, random, sys, time
from pathlib import Path


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln for ln in (l.rstrip("\n") for l in f) if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-slice", required=True, type=Path)
    ap.add_argument("--synth-corpus", required=True, type=Path)
    ap.add_argument("--ratio", required=True, type=float, help="synthetic fraction in [0,1]")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--gen", required=True, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--expect-chunks", required=True, type=int)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if not 0.0 <= a.ratio <= 1.0:
        sys.exit(f"CANARY FAIL: ratio {a.ratio} outside [0,1]")
    if a.out.exists() and not (a.force or a.dry_run):
        sys.exit(f"CANARY FAIL: {a.out} exists; pass --force to overwrite")

    N = a.expect_chunks
    n_synth = math.ceil(a.ratio * N)          # synthetic rounds up
    n_real = N - n_synth                       # real takes the remainder

    real = read_lines(a.real_slice) if n_real else []
    synth = read_lines(a.synth_corpus) if n_synth else []

    if n_real and len(real) < n_real:
        sys.exit(f"CANARY FAIL: real slice has {len(real)} chunks, need {n_real}")
    if n_synth and len(synth) < n_synth:
        sys.exit(f"CANARY FAIL: synth corpus has {len(synth)} chunks, need {n_synth}")

    # Frozen, composition-specific stream: same (gen, seed, ratio) -> same draw.
    rng = random.Random(f"5.3|{a.gen}|{a.seed}|{a.ratio:.4f}")
    mixed = (rng.sample(real, n_real) if n_real else []) + \
            (rng.sample(synth, n_synth) if n_synth else [])
    rng.shuffle(mixed)

    if len(mixed) != N:
        sys.exit(f"CANARY FAIL: assembled {len(mixed)} chunks, expected {N}")
    for i, ln in enumerate(mixed[:200]):
        try:
            json.loads(ln)
        except Exception as e:
            sys.exit(f"CANARY FAIL: line {i} is not valid JSON: {e}")

    print(f"ratio={a.ratio:.2f}  gen={a.gen}  seed={a.seed}  "
          f"total={N}  synth={n_synth}  real={n_real}")
    print(f"  real src  {a.real_slice}  ({len(real)} chunks)")
    print(f"  synth src {a.synth_corpus} ({len(synth)} chunks)")
    if a.dry_run:
        print("DRY RUN — nothing written")
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(a.out.suffix + ".tmp")
    tmp.write_text("\n".join(mixed) + "\n", encoding="utf-8")
    written = sum(1 for _ in open(tmp, encoding="utf-8"))
    if written != N:
        tmp.unlink()
        sys.exit(f"CANARY FAIL: wrote {written} lines, expected {N}")
    tmp.replace(a.out)

    out_md5 = md5(a.out)
    print(f"WROTE {a.out}  lines={written}  md5={out_md5}")

    if a.manifest:
        row = "\t".join(map(str, [
            time.strftime("%Y-%m-%dT%H:%M:%S"), a.gen, a.seed, f"{a.ratio:.2f}",
            N, n_synth, n_real, a.out, out_md5,
            a.real_slice, md5(a.real_slice) if n_real else "-",
            a.synth_corpus, md5(a.synth_corpus) if n_synth else "-",
        ]))
        with open(a.manifest, "a", encoding="utf-8") as f:
            f.write(row + "\n")
        print(f"manifest += {a.manifest}")


if __name__ == "__main__":
    main()
