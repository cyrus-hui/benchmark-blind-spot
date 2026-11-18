#!/usr/bin/env python3
"""Phase 7.5 preparation (decisions.md sec. 73). Run from ~/modelcollapse. Preflight everything, then write.

  python prepare_7p5.py --dry-run
  python prepare_7p5.py

A. Preflight (read-only, all-or-nothing):
   - six pipeline files carry their expected md5s (repo e04d5d7);
   - the 7 fresh real slices, the 3 gen-0 sources and the shared files exist;
   - which greedy-corpus gen-7 checkpoints exist (the C conditions; a condition needs all 3 seeds);
   - every anchor matches once; no target config, run root or task file exists.
B. Patch src/generate.py: decoding.corpus_temperature, read in corpus mode ONLY.
   Inverse canary: reversing the edits restores the file byte-for-byte.
   Kwargs canary: build_gen_kwargs is extracted from the old and the patched text and run on EVERY existing
   config (corpus = eval = old) and on every new config (eval = base; corpus = base with the new temperature).
C. Configs: 7 chains, 3 gen-0 corpus roots, 4 top-k-eval shadows.
D. Run roots. Gen-0 corpus roots hold a real gen0 dir (model symlink, read-only COPY of the pilot's gen-0
   metrics.json). Chain roots symlink gen0 like the 5.3 grid. Shadows hold gen7 dirs with a model symlink.
E. slurm/fc75_topkeval_tasks.txt (canary first), fc75_ood_lines.txt (147 scoring lines; append by hand after
   the chains), fc75_conditions.json (the registered held-out set, pinned before any 7.5 job runs).
"""
import argparse, ast, glob, hashlib, json, math, os, pathlib, shutil, sys

import yaml

ROOT = pathlib.Path(os.environ.get("SCRATCH", "")) / "collapse"
SEEDS, GENS, N = (43, 44, 45), range(1, 8), 18686
SRC_RUN, BASE_CFG, RN_LINE, T_LINE = "pilot_gpt2m", "configs/pilot_gpt2m.yaml", "run_name: pilot_gpt2m\n", "  temperature: 1.0\n"
SHARED = ["eval_prompts.jsonl", "ood_c4_test.jsonl", "ood_ccnews_test.jsonl",
          "ood_wiki_test.jsonl", "real_test.jsonl", "real_train.jsonl"]
EXPECT_MD5 = {  # repo e04d5d7
    "src/generate.py": "af214e5ad57e333a7086ed2de5260f6f",
    "slurm/run_ablation.sbatch": "db71fd461ebb00fa03cfd42f0cd71c92",
    "configs/pilot_gpt2m.yaml": "b744a6fcaf7400e9e0fb24ee760ff310",
    "src/build_mixed_corpus.py": "925e5bb6d8583fc9897cc3cd3243551a",
    "src/detectors.py": "a060b18d3d4035e1711e0cb0647323ca",
    "scripts/check_cell.sh": "22aa3dc80db48eb6fa67e2db2727a9cc",
}
# name, type, r (percent), corpus temperature (None = unchanged, 1.0)
CHAINS = [("mix10_topk", "A", 10, None), ("mix90_topk", "A", 90, None),
          ("fc75_r25_t075", "B", 25, 0.75), ("fc75_r25_t050", "B", 25, 0.50),
          ("fc75_r50_t075", "B", 50, 0.75), ("fc75_r50_t050", "B", 50, 0.50),
          ("fc75_r100_t125", "P", 100, 1.25)]
G0 = {0.75: "fc75_g0_t075", 0.50: "fc75_g0_t050", 1.25: "fc75_g0_t125"}
C_ARMS = [("mix25_greedy", 25), ("mix50_greedy", 50), ("mix75_greedy", 75), ("gpt2m_greedy", 100)]
CANARY = ("gpt2m_greedy", 43, 0)        # top-k eval of a greedy arm's gen-0 weights must reproduce the pilot's gen 0

GEN_PY = pathlib.Path("src/generate.py")
GEN_EDITS = [
    ('def build_gen_kwargs(cfg):\n',
     'def build_gen_kwargs(cfg, mode="eval"):\n'),
    ('    return kw\n',
     '    ct = d.get("corpus_temperature")  # Phase 7.5: training-corpus generation only; absent = unchanged\n'
     '    if mode == "corpus" and ct is not None:\n'
     '        if d["strategy"] == "greedy":\n'
     '            raise ValueError("corpus_temperature is undefined under greedy decoding")\n'
     '        kw["temperature"] = float(ct)\n'
     '    return kw\n'),
    ('    gen_kwargs = build_gen_kwargs(cfg)\n',
     '    gen_kwargs = build_gen_kwargs(cfg, args.mode)\n'
     '    print(f"decoding[{args.mode}] strategy={cfg[\'decoding\'][\'strategy\']} "\n'
     '          f"temperature={gen_kwargs.get(\'temperature\')} top_k={gen_kwargs.get(\'top_k\')}")\n'),
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
    for old, new in reversed(edits):          # inverse canary: the edit must be reversible
        if back.count(new) != 1:
            die(f"{name}: inverse canary: new text matches {back.count(new)} times")
        back = back.replace(new, old)
    if back != text:
        die(f"{name}: inverse canary failed: reversing the edits does not restore the file")
    return out


def kwargs_fn(source):
    """build_gen_kwargs, extracted from source text and compiled alone (no torch import)."""
    tree = ast.parse(source)
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_gen_kwargs"]
    if len(fn) != 1:
        die(f"build_gen_kwargs defined {len(fn)} times")
    ns = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), "<build_gen_kwargs>", "exec"), ns)
    return ns["build_gen_kwargs"]


def chain_cfg(base, name, r, t, slice_fmt):
    txt = base.replace(RN_LINE, f"run_name: {name}\n")
    if t is not None:
        txt = txt.replace(T_LINE, T_LINE + f"  corpus_temperature: {t:.2f}        # Phase 7.5: training corpus only; eval completions stay at temperature\n")
    what = "run_name" + (", the corpus temperature" if t is not None else "") + (" and the mixing keys below" if r is not None else "")
    tail = (f"\n\n# ---- Phase 7.5 early forecasting (decisions.md sec. 73, Nov 14 2026) ----\n"
            f"# derived from {BASE_CFG}; only {what} differ\n")
    if r is not None:
        tail += f"mix_ratio: {r / 100:.2f}\nmix_slice_fmt: {slice_fmt}\nmix_expect_chunks: {N}\n"
    return txt.rstrip("\n") + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ---------------- A. preflight ----------------
    if not os.environ.get("SCRATCH") or not ROOT.is_dir():
        die(f"$SCRATCH not set or {ROOT} missing")
    for f, m in EXPECT_MD5.items():
        if not os.path.isfile(f) or md5(f) != m:
            die(f"{f} md5 {md5(f) if os.path.isfile(f) else 'MISSING'} != expected {m} (repo e04d5d7); diff before patching")
    grid_cfg = yaml.safe_load(open("configs/mix25_topk.yaml"))
    slice_fmt = grid_cfg["mix_slice_fmt"]
    if slice_fmt.count("%d") != 1 or grid_cfg["mix_expect_chunks"] != N:
        die(f"configs/mix25_topk.yaml: unexpected mix_slice_fmt / mix_expect_chunks")
    man = pathlib.Path(slice_fmt).parent / "manifest.json"
    if not man.is_file():
        die(f"slice manifest not found: {man}")
    mtext = man.read_text()
    for g in GENS:
        s = pathlib.Path(slice_fmt % g)
        if not s.is_file() or nlines(s) != N:
            die(f"slice missing or not {N} lines: {s}")
        if md5(s) not in mtext:
            die(f"{s} md5 not in {man}")
    print(f"  7 real slices: {N} lines each, md5 in manifest  OK")
    for sd in SEEDS:
        g0 = ROOT / SRC_RUN / f"seed{sd}" / "gen0"
        if not (g0 / "model" / "config.json").is_file():
            die(f"missing {g0 / 'model' / 'config.json'}")
        if nlines(g0 / "synthetic_corpus.jsonl") != N:
            die(f"{g0 / 'synthetic_corpus.jsonl'} is not {N} lines")
        if not (g0 / "metrics.json").is_file():
            die(f"missing {g0 / 'metrics.json'}")
    for f in SHARED:
        if not (ROOT / SRC_RUN / f).is_file():
            die(f"shared file missing: {ROOT / SRC_RUN / f}")
    print("  gen-0 sources (3) and shared files (6) OK")

    c_ok, c_missing = [], []
    for arm, r in C_ARMS:
        miss = [sd for sd in SEEDS if not (ROOT / arm / f"seed{sd}" / "gen7" / "model" / "config.json").is_file()
                or not (ROOT / arm / f"seed{sd}" / "gen7" / "metrics.json").is_file()]
        (c_missing if miss else c_ok).append((arm, r, miss))
        print(f"  C condition {arm}: " + ("3/3 gen-7 checkpoints present" if not miss else f"MISSING seeds {miss} -> dropped"))
    can_arm, can_seed, can_gen = CANARY
    if c_ok and not (ROOT / can_arm / f"seed{can_seed}" / f"gen{can_gen}" / "model" / "config.json").is_file():
        die(f"canary weights missing: {ROOT / can_arm / f'seed{can_seed}' / f'gen{can_gen}' / 'model'}")
    if not c_ok:
        print("  WARNING: no C condition is available; the T -> 0 row is dropped (recorded in fc75_conditions.json)")

    old = GEN_PY.read_text()
    new = apply(old, GEN_EDITS, "generate.py")
    print(f"  generate.py: {len(GEN_EDITS)} anchors x1, inverse canary OK")

    base = pathlib.Path(BASE_CFG).read_text()
    if base.count(RN_LINE) != 1 or base.count(T_LINE) != 1 or "corpus_temperature" in base or "mix_ratio" in base:
        die(f"{BASE_CFG}: unexpected content (run_name / temperature line counts, or already has 7.5 keys)")
    configs = {}                                   # name -> (text, kind, r, t)
    for name, typ, r, t in CHAINS:
        configs[name] = (chain_cfg(base, name, r, t, slice_fmt), "chain", r, t)
    for t, name in G0.items():
        configs[name] = (chain_cfg(base, name, None, t, slice_fmt), "g0", None, t)
    shadow = {arm: [(sd, 7) for sd in SEEDS] for arm, r, _ in c_ok}     # arm -> cells to evaluate with top-k
    if c_ok:
        shadow.setdefault(can_arm, []).append((can_seed, can_gen))
    for arm in shadow:
        configs[arm + "_topkeval"] = (chain_cfg(base, arm + "_topkeval", None, None, slice_fmt), "shadow", None, None)
    for name in configs:
        if pathlib.Path(f"configs/{name}.yaml").exists() or (ROOT / name).exists():
            die(f"configs/{name}.yaml or run root {ROOT / name} already exists")
    for f in ("slurm/fc75_topkeval_tasks.txt", "fc75_ood_lines.txt", "fc75_conditions.json"):
        if os.path.exists(f):
            die(f"{f} already exists")

    f_old, f_new = kwargs_fn(old), kwargs_fn(new)
    existing = sorted(glob.glob("configs/*.yaml"))
    if not existing:
        die("no existing configs found (n = 0)")
    for c in existing:
        cfg = yaml.safe_load(open(c))
        if "corpus_temperature" in cfg["decoding"]:
            die(f"{c} already has corpus_temperature")
        if not (f_new(cfg) == f_new(cfg, "eval") == f_new(cfg, "corpus") == f_old(cfg)):
            die(f"kwargs canary: {c} changes under the patch")
    print(f"  kwargs canary: unchanged in both modes on all {len(existing)} existing configs")
    k_base = f_old(yaml.safe_load(base))
    n_t = 0
    for name, (txt, kind, r, t) in configs.items():
        cfg = yaml.safe_load(txt)
        if cfg["run_name"] != name or f_new(cfg, "eval") != k_base:
            die(f"generated {name}: run_name or eval kwargs differ from the base")
        want = dict(k_base, temperature=t) if t is not None else k_base
        if f_new(cfg, "corpus") != want or cfg["decoding"]["temperature"] != 1.0:
            die(f"generated {name}: corpus kwargs {f_new(cfg, 'corpus')} != {want}")
        n_t += t is not None
        if kind == "chain" and (abs(cfg["mix_ratio"] - r / 100) > 1e-12 or cfg["mix_slice_fmt"] != slice_fmt
                                or cfg["mix_expect_chunks"] != N):
            die(f"generated {name}: mixing keys wrong")
        if kind != "chain" and "mix_ratio" in cfg:
            die(f"generated {name}: must not mix")
    if n_t != 8:
        die(f"{n_t} configs carry a corpus temperature, expected 8")
    print(f"  {len(configs)} configs to write, no collisions, all parse as intended ({n_t} with a corpus temperature)")

    if a.dry_run:
        print("\nDRY RUN: preflight passed, nothing written")
        return

    # ---------------- B-E. write ----------------
    shutil.copy2(GEN_PY, str(GEN_PY) + ".bak_pre_7p5")
    GEN_PY.write_text(new)
    print(f"WROTE {GEN_PY} (backup {GEN_PY}.bak_pre_7p5) md5 {md5(GEN_PY)}")

    def root_with_shared(name):
        rr = ROOT / name
        rr.mkdir()
        for f in SHARED:
            (rr / f).symlink_to(ROOT / SRC_RUN / f)
        return rr

    for name, (txt, kind, r, t) in configs.items():
        pathlib.Path(f"configs/{name}.yaml").write_text(txt)
    for t, name in G0.items():                      # gen-0 corpus roots first: chains link to them
        rr = root_with_shared(name)
        for sd in SEEDS:
            src = ROOT / SRC_RUN / f"seed{sd}" / "gen0"
            d = rr / f"seed{sd}" / "gen0"
            d.mkdir(parents=True)
            (d / "model").symlink_to((src / "model").resolve())
            shutil.copyfile(src / "metrics.json", d / "metrics.json")
            os.chmod(d / "metrics.json", 0o444)
        print(f"WROTE configs/{name}.yaml and gen-0 corpus root {rr}")
    lines = []
    for name, typ, r, t in CHAINS:
        rr = root_with_shared(name)
        for sd in SEEDS:
            (rr / f"seed{sd}").mkdir()
            g0 = (ROOT / G0[t] if t is not None else ROOT / SRC_RUN) / f"seed{sd}" / "gen0"
            (rr / f"seed{sd}" / "gen0").symlink_to(g0)
            lines += [f"configs/{name}.yaml {g} {sd}" for g in GENS]
        print(f"WROTE configs/{name}.yaml and chain root {rr}")
    tasks = [f"{can_arm}_topkeval {can_seed} {can_gen}"] if c_ok else []      # the canary is task 1
    for arm, cells in shadow.items():
        rr = root_with_shared(arm + "_topkeval")
        for sd, g in cells:
            d = rr / f"seed{sd}" / f"gen{g}"
            d.mkdir(parents=True)
            (d / "model").symlink_to((ROOT / arm / f"seed{sd}" / f"gen{g}" / "model").resolve())
            if (arm, sd, g) != CANARY:
                tasks.append(f"{arm}_topkeval {sd} {g}")
        print(f"WROTE configs/{arm}_topkeval.yaml and shadow root {rr}")
    pathlib.Path("slurm/fc75_topkeval_tasks.txt").write_text("".join(t + "\n" for t in tasks))
    pathlib.Path("fc75_ood_lines.txt").write_text("\n".join(lines) + "\n")
    cond = {"registered": "decisions.md sec. 73 (Nov 14 2026)", "seeds": list(SEEDS), "n_chunks": N,
            "chains": [{"arm": n, "type": ty, "r": r, "corpus_temperature": t} for n, ty, r, t in CHAINS],
            "gen0_corpus_roots": {f"{t:.2f}": n for t, n in G0.items()},
            "C": [{"arm": arm, "r": r, "completions_arm": arm + "_topkeval"} for arm, r, _ in c_ok],
            "C_dropped": [{"arm": arm, "r": r, "missing_seeds": miss} for arm, r, miss in c_missing],
            "canary": {"arm": can_arm + "_topkeval", "seed": can_seed, "gen": can_gen} if c_ok else None}
    pathlib.Path("fc75_conditions.json").write_text(json.dumps(cond, indent=1) + "\n")
    print(f"WROTE slurm/fc75_topkeval_tasks.txt ({len(tasks)} tasks), fc75_ood_lines.txt ({len(lines)} lines), fc75_conditions.json")
    print(f"C conditions registered: {[c['arm'] for c in cond['C']] or 'NONE'}")


if __name__ == "__main__":
    main()
