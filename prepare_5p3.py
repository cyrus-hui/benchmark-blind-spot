#!/usr/bin/env python3
"""Phase 5.3 preparation — run from ~/modelcollapse. Preflight everything, then write.

  SLICE_FMT=/scratch/.../<name>%d.jsonl python prepare_5p3.py --dry-run
  SLICE_FMT=... python prepare_5p3.py

A. Preflight (read-only, all-or-nothing): 7 fresh slices exist, 18,686 lines,
   md5 present in the slice manifest.json; both gen-0 source dirs complete;
   every anchor below matches exactly once; no target file already exists.
B. Patch src/generate.py      : --prompt-corpus flag (default = old behaviour),
                                with an inverse canary proving the edit is additive.
C. Patch slurm/run_ablation.sbatch : mixing block, active only if cfg has mix_ratio.
D. Write 6 configs            : configs/mix{25,50,75}_{topk,greedy}.yaml
E. Run roots                  : shared-file symlinks + seed{43,44,45}/gen0 -> source arm
F. Scoring-manifest lines     : mix53_ood_lines.txt (126 lines, append by hand)
"""
import argparse, hashlib, os, pathlib, shutil, sys

ROOT = pathlib.Path(os.environ.get("SCRATCH", "")) / "collapse"
SEEDS, GENS, RATIOS = (43, 44, 45), range(1, 8), (25, 50, 75)
ARMS = {"topk": ("configs/pilot_gpt2m.yaml", "pilot_gpt2m", "run_name: pilot_gpt2m\n"),
        "greedy": ("configs/gpt2m_greedy.yaml", "gpt2m_greedy", "run_name: gpt2m_greedy\n")}
N = 18686
SHARED = ["eval_prompts.jsonl", "ood_c4_test.jsonl", "ood_ccnews_test.jsonl",
          "ood_wiki_test.jsonl", "real_test.jsonl", "real_train.jsonl"]

GEN_PY = pathlib.Path("src/generate.py")
GEN_EDITS = [
    ('    ap.add_argument("--seed", type=int, default=None)\n',
     '    ap.add_argument("--seed", type=int, default=None)\n'
     '    ap.add_argument("--prompt-corpus", default=None,  # Phase 5.3\n'
     '                    help="jsonl to draw corpus-mode prompts from; default unchanged")\n'),
    ('        if args.generation == 0:\n'
     '            src_rows = read_jsonl(root / "real_train.jsonl")\n'
     '        else:\n'
     '            src_rows = read_jsonl(gen_dir(cfg, args.generation - 1, seed) / "synthetic_corpus.jsonl")\n',
     '        if args.prompt_corpus:  # Phase 5.3: prompts from what this generation actually trained on\n'
     '            src_rows = read_jsonl(Path(args.prompt_corpus))\n'
     '        elif args.generation == 0:\n'
     '            src_rows = read_jsonl(root / "real_train.jsonl")\n'
     '        else:\n'
     '            src_rows = read_jsonl(gen_dir(cfg, args.generation - 1, seed) / "synthetic_corpus.jsonl")\n'
     '        print(f"prompt source: {args.prompt_corpus or \'default\'} ({len(src_rows)} rows)")\n'),
]

SB = pathlib.Path("slurm/run_ablation.sbatch")
MIX_BLOCK = r'''# ---- Phase 5.3 ratio mixing (decisions.md, Sept 15 2026 pre-registration) ----
# No-op unless the config defines mix_ratio: every pre-existing config is unaffected.
PROMPT_ARGS=()
read MIX_RATIO MIX_SLICE_FMT MIX_N < <(python -c "import yaml,sys;c=yaml.safe_load(open(sys.argv[1]));print(c.get('mix_ratio','-'),c.get('mix_slice_fmt','-'),c.get('mix_expect_chunks','-'))" "$CFG")
if [ "$MIX_RATIO" != "-" ] && [ "$GEN" -gt 0 ]; then
    SLICE=$(printf "$MIX_SLICE_FMT" "$GEN")
    MIXED="$RUN_ROOT/mixed/seed$SEED/gen$GEN.jsonl"
    echo "5.3 mix: ratio=$MIX_RATIO gen=$GEN seed=$SEED slice=$SLICE synth=$CORPUS"
    python src/build_mixed_corpus.py --real-slice "$SLICE" --synth-corpus "$CORPUS" \
        --ratio "$MIX_RATIO" --seed "$SEED" --gen "$GEN" --out "$MIXED" \
        --expect-chunks "$MIX_N" --manifest "$RUN_ROOT/mix_manifest.tsv" --force \
        || { echo "5.3 mix: build_mixed_corpus FAILED"; exit 1; }
    CORPUS="$MIXED"
    PROMPT_ARGS=(--prompt-corpus "$MIXED")
fi
'''
SB_FT = 'python src/finetune.py --config $CFG --generation $GEN --corpus "$CORPUS" --seed $SEED\n'
SB_GC = 'python src/generate.py --config $CFG --generation $GEN --mode corpus --seed $SEED\n'
SB_EDITS = [
    (SB_FT, MIX_BLOCK + SB_FT),
    (SB_GC, 'python src/generate.py --config $CFG --generation $GEN --mode corpus --seed $SEED'
            ' ${PROMPT_ARGS[@]+"${PROMPT_ARGS[@]}"}\n'),
]


def die(msg):
    sys.exit(f"PREFLIGHT FAIL: {msg}\nNothing has been written.")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def nlines(p):
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def apply(text, edits, name):
    out = text
    for i, (old, new) in enumerate(edits, 1):
        if out.count(old) != 1:
            die(f"{name} edit {i}: anchor matches {out.count(old)} times, need 1:\n  {old[:90]!r}")
        out = out.replace(old, new)
    back = out
    for old, new in reversed(edits):          # inverse canary: edit must be purely additive
        back = back.replace(new, old)
    if back != text:
        die(f"{name}: inverse canary failed — edit is not reversible")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ---------------- A. preflight ----------------
    if not os.environ.get("SCRATCH"):
        die("$SCRATCH not set")
    fmt = os.environ.get("SLICE_FMT", "")
    if fmt.count("%d") != 1 or not fmt.startswith("/"):
        die("set SLICE_FMT to an absolute path with exactly one %d, e.g. .../fresh_slices_gpt2m/slice_gen%d.jsonl")
    man = pathlib.Path(fmt).parent / "manifest.json"
    if not man.is_file():
        die(f"slice manifest not found: {man}")
    mtext = man.read_text()
    for g in GENS:
        s = pathlib.Path(fmt % g)
        if not s.is_file():
            die(f"slice missing: {s}")
        if nlines(s) != N:
            die(f"{s} has {nlines(s)} lines, expected {N}")
        m = md5(s)
        if m not in mtext:
            die(f"{s} md5 {m} not in {man}")
        print(f"  slice gen{g}: {N} lines, md5 {m} in manifest  OK")

    for arm, (src_cfg, src_run, _) in ARMS.items():
        for sd in SEEDS:
            g0 = ROOT / src_run / f"seed{sd}" / "gen0"
            if not (g0 / "model").is_dir():
                die(f"missing {g0/'model'}")
            if nlines(g0 / "synthetic_corpus.jsonl") != N:
                die(f"{g0/'synthetic_corpus.jsonl'} is not {N} lines")
    for f in SHARED:
        if not (ROOT / "pilot_gpt2m" / f).is_file():
            die(f"shared file missing: {ROOT/'pilot_gpt2m'/f}")
    print("  gen-0 sources (6) and shared files OK")

    gen_new = apply(GEN_PY.read_text(), GEN_EDITS, "generate.py")
    sb_new = apply(SB.read_text(), SB_EDITS, "run_ablation.sbatch")
    print("  generate.py: 2 anchors x1, inverse canary OK")
    print("  run_ablation.sbatch: 2 anchors x1, inverse canary OK")

    configs = {}
    for arm, (src_cfg, src_run, rn_line) in ARMS.items():
        base = pathlib.Path(src_cfg).read_text()
        if base.count(rn_line) != 1:
            die(f"{src_cfg}: '{rn_line.strip()}' matches {base.count(rn_line)} times")
        if "mix_ratio" in base:
            die(f"{src_cfg} already mentions mix_ratio")
        for r in RATIOS:
            name = f"mix{r}_{arm}"
            p = pathlib.Path(f"configs/{name}.yaml")
            if p.exists():
                die(f"{p} already exists")
            if (ROOT / name).exists():
                die(f"run root {ROOT/name} already exists")
            txt = base.replace(rn_line, f"run_name: {name}\n")
            txt = txt.rstrip("\n") + (
                f"\n\n# ---- Phase 5.3 severity grid (decisions.md, Sept 15 2026) ----\n"
                f"# derived from {src_cfg}; only run_name and these keys differ\n"
                f"mix_ratio: {r/100:.2f}\nmix_slice_fmt: {fmt}\nmix_expect_chunks: {N}\n")
            configs[name] = (p, txt, src_run)
    print(f"  {len(configs)} configs to write, no collisions")

    if a.dry_run:
        print("\nDRY RUN — preflight passed, nothing written")
        return

    # ---------------- B-F. write ----------------
    for path, new in ((GEN_PY, gen_new), (SB, sb_new)):
        shutil.copy2(path, str(path) + ".bak_pre_5p3")
        path.write_text(new)
        print(f"WROTE {path} (backup {path}.bak_pre_5p3)")
    lines = []
    for name, (p, txt, src_run) in configs.items():
        p.write_text(txt)
        rr = ROOT / name
        rr.mkdir()
        for f in SHARED:
            (rr / f).symlink_to(ROOT / "pilot_gpt2m" / f)
        for sd in SEEDS:
            (rr / f"seed{sd}").mkdir()
            (rr / f"seed{sd}" / "gen0").symlink_to(ROOT / src_run / f"seed{sd}" / "gen0")
            lines += [f"configs/{name}.yaml {g} {sd}" for g in GENS]
        print(f"WROTE {p} and run root {rr}")
    pathlib.Path("mix53_ood_lines.txt").write_text("\n".join(lines) + "\n")
    print(f"WROTE mix53_ood_lines.txt ({len(lines)} lines) — compare format with the manifest tail before appending")


if __name__ == "__main__":
    main()
