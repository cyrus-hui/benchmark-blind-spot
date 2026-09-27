#!/usr/bin/env python3
"""Phase 7.1 preparation (decisions.md §62, §64). Run from ~/modelcollapse. Preflight everything, then write.

  python prepare_7p1.py --dry-run
  python prepare_7p1.py

A. Preflight (read-only, all-or-nothing):
   - slurm/run_ablation.sbatch and src/step0_score.py carry their expected md5s (committed state).
   - the 6 gen-0 sources (gpt2m_greedy / pilot_gpt2m, seeds 43-45) hold model/ and an 18,686-line corpus.
   - the 6 step-0 own-scorer npz (K-F1 references) exist and are non-empty.
   - the anchor matches once; no config or run root already exists.
B. Patch slurm/run_ablation.sbatch: the filter block, active only if the config has `filter:`.
   Inverse canary: removing the block restores the original byte-for-byte.
   No-op canary (K-F4, second half): the hook's own parser, run on EVERY existing config, prints
   the no-filter sentinel. A config that already has a filter: block stops the script.
C. Write 12 configs configs/filt71_{F,R}{10,25,50}_{greedy,topk}.yaml
D. Run roots: shared-file symlinks + seed{43,44,45}/gen0 -> the unfiltered arm (U).
E. filt71_ood_lines.txt: 252 scoring-manifest lines (append by hand after the chains).
"""
import argparse, glob, hashlib, os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(os.environ.get("SCRATCH", "")) / "collapse"
SEEDS, GENS, QS = (43, 44, 45), range(1, 8), ((10, 0.10), (25, 0.25), (50, 0.50))
U = {"greedy": ("configs/gpt2m_greedy.yaml", "gpt2m_greedy", "run_name: gpt2m_greedy\n"),
     "topk": ("configs/pilot_gpt2m.yaml", "pilot_gpt2m", "run_name: pilot_gpt2m\n")}
N = 18686
SHARED = ["eval_prompts.jsonl", "ood_c4_test.jsonl", "ood_ccnews_test.jsonl",
          "ood_wiki_test.jsonl", "real_test.jsonl", "real_train.jsonl"]
EXPECT_MD5 = {"slurm/run_ablation.sbatch": "44b3d71649a5428a26cc68cab1c32314",   # repo 3c739ee
              "src/step0_score.py": "e6bdbc81f1574cf32b32e0f3b20256c5"}
STEP0 = ROOT / "step0_7p1" / "scores"

SB = pathlib.Path("slurm/run_ablation.sbatch")
PARSE = ("python -c \"import yaml,sys;f=yaml.safe_load(open(sys.argv[1])).get('filter') or {};"
         "print(*(f.get(k,'-') for k in ('arm','q','decoder','pair_root','k1_ref_fmt')))\" \"$CFG\"")
SENTINEL = "- - - - -"
FILTER_BLOCK = r'''# ---- Phase 7.1 corpus filter (decisions.md §62 + §64, registered Sept 26 2026) ----
# No-op unless the config has a filter: block: every pre-existing config is unaffected.
# Prompts are untouched: corpus mode still draws them from gen g-1's FULL synthetic corpus (§62.4).
read FILT_ARM FILT_Q FILT_DEC FILT_PAIR FILT_K1 < <(''' + PARSE + r''')
if [ "$FILT_ARM" != "-" ] && [ "$GEN" -gt 0 ]; then
    [ "$MIX_RATIO" = "-" ] || { echo "7.1 filter: REFUSE config has both mix_ratio and filter"; exit 1; }
    for f in src/filter_corpus.py src/step0_score.py; do
        git ls-files --error-unmatch "$f" >/dev/null 2>&1 && git diff --quiet HEAD -- "$f" \
            || { echo "7.1 filter: REFUSE $f is not committed and unmodified"; exit 1; }
    done
    FILTERED="$RUN_ROOT/filtered/seed$SEED/gen$GEN.jsonl"
    FARGS=(--arm "$FILT_ARM" --q "$FILT_Q" --decoder "$FILT_DEC" --seed "$SEED" --gen "$GEN"
           --corpus "$CORPUS" --scorer "$RUN_ROOT/seed$SEED/gen0/model" --out "$FILTERED"
           --manifest "$RUN_ROOT/filter_manifest.tsv"
           --scores-out "$RUN_ROOT/filter_scores/seed$SEED/gen$GEN.npz")
    [ "$FILT_ARM" = "R" ] && FARGS+=(--pair-manifest "$FILT_PAIR/filter_manifest.tsv")
    [ "$GEN" -eq 1 ] && FARGS+=(--k1-ref "$(printf "$FILT_K1" "$SEED")")
    echo "7.1 filter: arm=$FILT_ARM q=$FILT_Q dec=$FILT_DEC gen=$GEN seed=$SEED head=$(git rev-parse --short HEAD) synth=$CORPUS"
    python src/filter_corpus.py "${FARGS[@]}" || { echo "7.1 filter: filter_corpus FAILED"; exit 1; }
    CORPUS="$FILTERED"
    echo "7.1 filter: training corpus $CORPUS md5 $(md5sum "$CORPUS" | cut -d' ' -f1) lines $(wc -l < "$CORPUS")"
fi
'''
SB_FT = 'python src/finetune.py --config $CFG --generation $GEN --corpus "$CORPUS" --seed $SEED\n'


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


def parse_filter(cfg_path):
    r = subprocess.run(["bash", "-c", PARSE.replace('"$CFG"', '"$1"'), "_", cfg_path],
                       capture_output=True, text=True)
    if r.returncode:
        die(f"hook parser failed on {cfg_path}: {r.stderr.strip()}")
    return r.stdout.strip()


def name(arm, qi, dec):
    return f"filt71_{arm}{qi}_{dec}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ---------------- A. preflight ----------------
    if not os.environ.get("SCRATCH"):
        die("$SCRATCH not set")
    for f, m in EXPECT_MD5.items():
        if md5(f) != m:
            die(f"{f} md5 {md5(f)} != expected {m} (repo 3c739ee); stop and diff before patching")
    for dec, (_, src, _) in U.items():
        for s in SEEDS:
            g0 = ROOT / src / f"seed{s}" / "gen0"
            if not (g0 / "model").is_dir():
                die(f"missing {g0/'model'}")
            if nlines(g0 / "synthetic_corpus.jsonl") != N:
                die(f"{g0/'synthetic_corpus.jsonl'} is not {N} lines")
            ref = STEP0 / f"{src}_s{s}_c1_own.npz"
            if not ref.is_file() or ref.stat().st_size == 0:
                die(f"K-F1 reference missing/empty: {ref}")
    for f in SHARED:
        if not (ROOT / "pilot_gpt2m" / f).is_file():
            die(f"shared file missing: {ROOT/'pilot_gpt2m'/f}")
    print("  gen-0 sources (6), K-F1 references (6), shared files (6) OK")

    old = SB.read_text()
    if old.count(SB_FT) != 1:
        die(f"sbatch anchor matches {old.count(SB_FT)} times, need 1")
    if "Phase 7.1" in old:
        die("sbatch already contains a Phase 7.1 block")
    new = old.replace(SB_FT, FILTER_BLOCK + SB_FT)
    if new.replace(FILTER_BLOCK, "", 1) != old:
        die("inverse canary failed: the edit is not purely additive")
    print("  run_ablation.sbatch: 1 anchor x1, inverse canary OK")

    existing = sorted(glob.glob("configs/*.yaml"))
    if not existing:
        die("no existing configs found (n = 0)")
    for c in existing:
        out = parse_filter(c)
        if out != SENTINEL:
            die(f"no-op canary: {c} parses to '{out}', expected '{SENTINEL}'")
    print(f"  no-op canary: hook parser prints '{SENTINEL}' on all {len(existing)} existing configs")

    configs = {}
    for dec, (src_cfg, src_run, rn_line) in U.items():
        base = pathlib.Path(src_cfg).read_text()
        if base.count(rn_line) != 1 or "filter:" in base or "mix_ratio" in base:
            die(f"{src_cfg}: unexpected content (run_name count / filter / mix_ratio)")
        for arm in ("F", "R"):
            for qi, q in QS:
                nm = name(arm, qi, dec)
                p = pathlib.Path(f"configs/{nm}.yaml")
                if p.exists() or (ROOT / nm).exists():
                    die(f"{p} or run root {ROOT/nm} already exists")
                pair = (f"  pair_root: {ROOT / name('F', qi, dec)}   # the F run root whose manifest supplies k (§64.1)\n"
                        if arm == "R" else "")
                txt = base.replace(rn_line, f"run_name: {nm}\n").rstrip("\n") + (
                    f"\n\n# ---- Phase 7.1 filter experiment (decisions.md §62 + §64, Sept 26 2026) ----\n"
                    f"# derived from {src_cfg}; only run_name and this block differ\n"
                    f"filter:\n"
                    f"  arm: {arm}\n"
                    f"  q: {q}\n"
                    f"  decoder: {dec}\n"
                    + pair +
                    f"  k1_ref_fmt: {STEP0}/{src_run}_s%d_c1_own.npz\n")
                configs[nm] = (p, txt, src_run)
    # every generated config goes through the hook's own parser BEFORE anything is written
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"filt71_preflight_{os.getpid()}.yaml"
    for nm, (p, txt, _) in configs.items():
        tmp.write_text(txt)
        out = parse_filter(str(tmp)).split()
        arm, qi, dec = nm[7], int(nm[8:10]), nm.split("_")[2]
        want = [arm, str(dict(QS)[qi]), dec] + ([str(ROOT / name("F", qi, dec))] if arm == "R" else ["-"])
        if len(out) != 5 or out[:4] != want or "%d" not in out[4]:
            tmp.unlink(); die(f"generated {nm} parses to {out}, expected {want} + k1_ref_fmt")
    tmp.unlink()
    print(f"  {len(configs)} configs to write, no collisions, all parse as intended")

    if a.dry_run:
        print("\nDRY RUN — preflight passed, nothing written")
        return

    # ---------------- B-E. write ----------------
    shutil.copy2(SB, str(SB) + ".bak_pre_7p1")
    SB.write_text(new)
    print(f"WROTE {SB} (backup {SB}.bak_pre_7p1) md5 {md5(SB)}")
    lines = []
    for nm, (p, txt, src_run) in configs.items():
        p.write_text(txt)
        out = parse_filter(str(p))
        if out == SENTINEL or out.split()[0] not in ("F", "R"):
            sys.exit(f"POST-WRITE FAIL: {p} parses to '{out}'")
        rr = ROOT / nm
        rr.mkdir()
        for f in SHARED:
            (rr / f).symlink_to(ROOT / "pilot_gpt2m" / f)
        for s in SEEDS:
            (rr / f"seed{s}").mkdir()
            (rr / f"seed{s}" / "gen0").symlink_to(ROOT / src_run / f"seed{s}" / "gen0")
            lines += [f"configs/{nm}.yaml {g} {s}" for g in GENS]
        print(f"WROTE {p} ({out}) and run root {rr}")
    pathlib.Path("filt71_ood_lines.txt").write_text("\n".join(lines) + "\n")
    print(f"WROTE filt71_ood_lines.txt ({len(lines)} lines) — append to slurm/ood_manifest.txt only after the chains")


if __name__ == "__main__":
    main()
