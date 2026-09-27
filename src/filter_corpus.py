#!/usr/bin/env python
"""7.1 corpus filter — decisions.md §62 (pre-registration) and §64 (clarifications, before code).

One call builds ONE generation's training corpus for ONE chain of arm F(q) or R(q).

  input : the full synthetic corpus written by model g-1 (18,686 rows, 32 prompt + 96 generated)
  score : S(d) from the PINNED step-0 M2 scorer (src/step0_score.py: score_ids), own-seed gen-0
          checkpoint, fp32, TF32 off. Rows that do not re-tokenize to 128 tokens are unscorable:
          kept in every arm, never counted in k (§62.2).
  F(q)  : drop the k = floor(q*n + 1/2) scorable rows with the lowest S; ties by row index (§62.3).
  R(q)  : drop k scorable rows = Random(f"7.1|R|{decoder}|{seed}|{gen}|{q}").sample(scorable, k).
          k is READ from the paired F(q) chain's manifest row (same decoder, q, seed, gen), so
          kept rows are identical in number to F's at every generation (§64.1). R refuses if
          the row is absent.
  output: the kept input lines, byte-for-byte, in their original order.

Canaries (§62.8; every one refuses at n = 0):
  K-F1  gen 1 only: S reproduces the step-0 own-scorer npz on every scored row, max|dS| <= 1e-5.
  K-F2  counts: kept = n_total - k; R's k = F's k; gen 1: R's unscorable set = F's (md5).
        (Prompt source and step count are checked by scripts/filt71_gate.py from the log.)
  K-F3  selection: F's dropped max S <= kept min S. R's dropped mean S lies within 4 SE of the
        scorable mean (SE with finite-population correction, §64.4), and F's dropped rows
        against the same band must fire. Both halves run in BOTH arms.
  K-F4  additivity: the k = 0 rendering through the same writer is byte-identical (md5) to the
        input. (The no-op of every pre-existing config is checked by prepare_7p1.py.)

Prints NO statistic of S. Per-row S goes to an .npz for the registered analysis only.
Exit codes: 0 OK, 2 REFUSE (input/state), 3 canary FAIL.
"""
import argparse, hashlib, math, os, random, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from step0_score import score_ids, md5_file, prompt_hash, PROMPT, BLOCK  # pinned scorer (§62.2)

KF1_TOL = 1e-5
KF3_Z = 4.0
HEADER = ["timestamp", "arm", "q", "decoder", "seed", "gen", "n_total", "k", "out_md5", "kept",
          "n_scorable", "n_unscorable", "unscorable_md5", "dropped_md5", "in_path", "in_md5",
          "out_path", "scorer", "scorer_weights_md5", "k_source", "kf1", "kf3", "kf4",
          "expect_steps", "script_md5", "git_head", "scores_npz"]
# column 9 (1-based) is out_md5, which is what scripts/check_cell.sh --manifest matches


def refuse(msg):
    print(f"7.1 filter REFUSE: {msg}"); sys.exit(2)


def fail(msg):
    print(f"7.1 filter CANARY FAIL: {msg}"); sys.exit(3)


def md5_bytes(b):
    return hashlib.md5(b).hexdigest()


def idx_md5(idx):
    return md5_bytes(",".join(map(str, sorted(idx))).encode())


def render(lines, drop):
    """The only writer. lines: list of raw bytes lines (with their line endings)."""
    return b"".join(l for i, l in enumerate(lines) if i not in drop)


def read_manifest(path):
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path) as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if p and p[0] != "timestamp" and len(p) == len(HEADER):
                rows.append(dict(zip(HEADER, p)))
    return rows


def append_manifest(path, rec):
    new = not os.path.isfile(path)
    with open(path, "a") as f:
        if new:
            f.write("\t".join(HEADER) + "\n")
        f.write("\t".join(str(rec[h]) for h in HEADER) + "\n")


def select_F(S, rows, k):
    order = sorted(range(len(rows)), key=lambda j: (S[j], rows[j]))
    return {rows[j] for j in order[:k]}


def select_R(rows, k, decoder, seed, gen, q):
    rng = random.Random(f"7.1|R|{decoder}|{seed}|{gen}|{q}")
    return set(rng.sample(sorted(rows), k))


def kf3(S, rows, dropF, dropR, k):
    """Returns (ok, detail) without printing any S value."""
    import numpy as np
    n = len(rows)
    pos = {r: j for j, r in enumerate(rows)}
    s = np.asarray(S, dtype="float64")
    keptF = [pos[r] for r in rows if r not in dropF]
    dF = [pos[r] for r in dropF]
    if k == 0:
        return True, "k=0 (nothing dropped; selection checks vacuous by construction)"
    if n < 2 or not keptF:
        return False, f"n={n} kept={len(keptF)}: selection check undefined"
    ok_order = s[dF].max() <= s[keptF].min()
    se = s.std(ddof=1) / math.sqrt(k) * math.sqrt((n - k) / (n - 1))
    mu = s.mean()
    zR = (s[[pos[r] for r in dropR]].mean() - mu) / se if se > 0 else float("inf")
    zF = (s[dF].mean() - mu) / se if se > 0 else float("inf")
    ok = bool(ok_order and abs(zR) <= KF3_Z and abs(zF) > KF3_Z)
    # |z| values are canary quantities about the selection, not statistics of the corpus
    return ok, f"F-order={'ok' if ok_order else 'VIOLATED'} |z_R|={abs(zR):.2f}<=4 |z_F|={abs(zF):.1f}>4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["F", "R"])
    ap.add_argument("--q", required=True, type=float)
    ap.add_argument("--decoder", required=True, choices=["greedy", "topk"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--gen", required=True, type=int)
    ap.add_argument("--corpus", required=True, help="full synthetic corpus written by model gen-1")
    ap.add_argument("--scorer", required=True, help="own-seed gen-0 model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True, help="this run root's filter_manifest.tsv")
    ap.add_argument("--scores-out", required=True, help=".npz of per-row S (never printed)")
    ap.add_argument("--pair-manifest", default="", help="R only: the paired F run root's manifest")
    ap.add_argument("--k1-ref", default="", help="gen 1: step-0 own-scorer npz for this seed")
    ap.add_argument("--expect-rows", type=int, default=18686)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    if a.gen < 1:
        refuse("gen must be >= 1 (gen 0 trains on real data)")
    if not 0.0 < a.q < 1.0:
        refuse(f"q={a.q} outside (0,1)")
    if a.arm == "R" and not a.pair_manifest:
        refuse("arm R needs --pair-manifest (k comes from the paired F chain, §64.1)")
    if a.gen == 1 and not a.k1_ref:
        refuse("gen 1 needs --k1-ref (K-F1)")
    qs = str(a.q)  # the registered rng string uses this spelling: 0.1, 0.25, 0.5 (§64.3)

    with open(a.corpus, "rb") as f:
        lines = f.readlines()
    n_total = len(lines)
    if n_total != a.expect_rows:
        refuse(f"{a.corpus} has {n_total} lines, expected {a.expect_rows}")
    if any(not l.strip() for l in lines):
        refuse("blank line in corpus")
    if not lines[-1].endswith(b"\n"):
        refuse("corpus does not end in a newline")
    raw = b"".join(lines)
    in_md5 = md5_bytes(raw)

    import json
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tok = AutoTokenizer.from_pretrained(a.scorer)
    rows, ids, unscorable = [], [], []
    for i, l in enumerate(lines):
        t = tok(json.loads(l)["text"], add_special_tokens=False)["input_ids"]
        if len(t) == BLOCK:
            rows.append(i); ids.append(t)
        else:
            unscorable.append(i)
    n = len(rows)
    if n == 0:
        fail("n_scorable = 0")

    # ---- k (before scoring, so a missing F row refuses in seconds) ----
    k_own = int(math.floor(a.q * n + 0.5))
    unscorable_md5 = idx_md5(unscorable)
    if a.arm == "F":
        k, k_source = k_own, "own"
    else:
        prs = [r for r in read_manifest(a.pair_manifest)
               if r["arm"] == "F" and r["q"] == qs and r["decoder"] == a.decoder
               and r["seed"] == str(a.seed) and r["gen"] == str(a.gen)]
        if not prs:
            refuse(f"no F row for q={qs} {a.decoder} s{a.seed} g{a.gen} in {a.pair_manifest}")
        ks = {r["k"] for r in prs}
        if len(ks) != 1:
            fail(f"K-F2: paired F rows disagree on k: {sorted(ks)}")
        k, k_source = int(ks.pop()), f"F:{a.pair_manifest}"
        if a.gen == 1:
            if k != k_own:
                fail(f"K-F2: gen 1 shares F's corpus but k_own={k_own} != F's k={k}")
            if {r["unscorable_md5"] for r in prs} != {unscorable_md5}:
                fail("K-F2: gen 1 unscorable set differs from F's")
    if not 0 < k < n:
        fail(f"K-F2: k={k} not in (0, n={n})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.scorer, torch_dtype=torch.float32).to(device).eval()
    wf = [w for w in ("model.safetensors", "pytorch_model.bin") if os.path.exists(os.path.join(a.scorer, w))]
    w_md5 = md5_file(os.path.join(a.scorer, wf[0])) if wf else "-"
    t0 = time.time()
    S, _ = score_ids(model, torch.tensor(ids, dtype=torch.long), device, a.batch)
    if not np.isfinite(S).all():
        fail("non-finite S")

    # ---- K-F1 (gen 1 only: the only generation with a step-0 reference) ----
    kf1 = "n/a(gen>1)"
    if a.gen == 1:
        ref = np.load(a.k1_ref)
        rrow, rS = ref["row"].astype("int64"), ref["S"].astype("float64")
        if len(rrow) == 0:
            fail("K-F1: reference has 0 rows")
        if list(rrow) != rows:
            fail(f"K-F1: scored-row set differs from step-0 ({len(rrow)} vs {n} rows)")
        d = float(np.abs(np.asarray(S, "float64") - rS).max())
        if not d <= KF1_TOL:
            fail(f"K-F1: max|dS| = {d:.2e} > {KF1_TOL:g}")
        kf1 = f"PASS({len(rrow)} rows, max|dS|={d:.1e})"
        print(f"K-F1 PASS: {len(rrow)} rows, max|dS| = {d:.2e} <= {KF1_TOL:g}")

    dropF = select_F(S, rows, k)
    dropR = select_R(rows, k, a.decoder, a.seed, a.gen, qs)
    drop = dropF if a.arm == "F" else dropR
    ok3, det3 = kf3(S, rows, dropF, dropR, k)
    if not ok3:
        fail(f"K-F3: {det3}")
    print(f"K-F3 PASS: {det3}")

    # ---- K-F4: the writer at k = 0 is the identity ----
    if md5_bytes(render(lines, set())) != in_md5:
        fail("K-F4: k=0 rendering is not byte-identical to the input")
    print("K-F4 PASS: k=0 rendering md5 == input md5")

    out = render(lines, drop)
    kept = n_total - k
    n_out = out.count(b"\n")
    if n_out != kept:
        fail(f"K-F2: wrote {n_out} lines, expected kept={kept}")
    out_md5 = md5_bytes(out)
    if os.path.exists(a.out):
        if md5_file(a.out) != out_md5:
            refuse(f"{a.out} exists with different content (never overwritten)")
        print(f"output exists and is identical: {a.out}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out + ".tmp", "wb") as f:
            f.write(out)
        os.replace(a.out + ".tmp", a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.scores_out)), exist_ok=True)
    if not os.path.exists(a.scores_out):
        np.savez(a.scores_out + ".tmp.npz", row=np.array(rows, "int64"), S=np.asarray(S, "float64"),
                 prompt_hash=np.array([prompt_hash(t) for t in ids], "int64"),
                 dropped=np.array(sorted(drop), "int64"))
        os.replace(a.scores_out + ".tmp.npz", a.scores_out)

    try:
        head = subprocess.check_output(["git", "-C", os.path.dirname(HERE), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        head = "-"
    steps = 3 * math.ceil(kept / 32)
    rec = dict(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"), arm=a.arm, q=qs, decoder=a.decoder,
               seed=a.seed, gen=a.gen, n_total=n_total, k=k, out_md5=out_md5, kept=kept,
               n_scorable=n, n_unscorable=len(unscorable), unscorable_md5=unscorable_md5,
               dropped_md5=idx_md5(drop), in_path=os.path.realpath(a.corpus), in_md5=in_md5,
               out_path=os.path.abspath(a.out), scorer=os.path.realpath(a.scorer),
               scorer_weights_md5=w_md5, k_source=k_source, kf1=kf1, kf3="PASS", kf4="PASS",
               expect_steps=steps, script_md5=md5_file(os.path.abspath(__file__)), git_head=head,
               scores_npz=os.path.abspath(a.scores_out))
    append_manifest(a.manifest, rec)
    print(f"7.1 filter OK: arm={a.arm} q={qs} {a.decoder} s{a.seed} g{a.gen} n_total={n_total} "
          f"n_scorable={n} unscorable={len(unscorable)} k={k} ({k_source.split(':')[0]}) kept={kept} "
          f"expect_steps={steps} out_md5={out_md5} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
