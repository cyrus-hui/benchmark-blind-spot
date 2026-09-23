#!/usr/bin/env python
"""7.1 step 0 scorer — decisions.md §40/§41. Scores ONE corpus under ONE scorer.

S(d) = (1/96) sum_{t in generated span} [ -log q_ref(x_t|x_<t) - H(q_ref(.|x_<t)) ]

Surface (§40, do not re-derive): row = 32 prompt tokens + 96 generated tokens
stored as one string; scored targets are token indices 32..127 (1-indexed
positions 33..128), predicted by logits at indices 31..126. Nulls get the same
mask. Any row that does not re-tokenize to exactly 128 tokens is excluded and
counted (§41). For mixed corpora only the synthetic rows are scored (§40.5):
a row is real iff its text is in the real slice; counts are asserted.

Canary (K3): with --enforce-k3, the argmax-match rate on the scored span must
be >= 0.99, else a FAIL json is written, no scores are written, exit 3.

This script prints NO statistic of S. It writes per-row scores to .npz and
canary/meta fields to .json. Reading S is the analysis script's job, gated.
"""
import argparse, hashlib, json, os, subprocess, sys, time

PROMPT, BLOCK = 32, 128
NGEN = BLOCK - PROMPT  # 96
K3_MIN = 0.99


def score_ids(model, ids, device, batch=32):
    """ids: LongTensor [N,128]. Returns (S float64 [N], n_argmax int64 [N])."""
    import torch
    S_all, A_all = [], []
    with torch.inference_mode():
        for i in range(0, ids.shape[0], batch):
            x = ids[i:i + batch].to(device)
            logits = model(x).logits[:, PROMPT - 1:BLOCK - 1, :].float()   # [B,96,V]
            tgt = x[:, PROMPT:BLOCK]                                        # [B,96]
            logp = torch.log_softmax(logits, dim=-1)
            nll = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)           # [B,96]
            H = -(logp.exp() * logp).sum(-1)                                # [B,96]
            g = (nll - H).double()
            S_all.append(g.mean(-1).cpu())
            A_all.append((logits.argmax(-1) == tgt).sum(-1).cpu())
            del logits, logp, nll, H, g
    return torch.cat(S_all).numpy(), torch.cat(A_all).numpy().astype("int64")


def md5_file(p, chunk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_texts(path, key):
    out = []
    with open(path) as f:
        for n, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if key not in row or not isinstance(row[key], str):
                sys.exit(f"REFUSE: row {n} of {path} has no str field '{key}' (keys: {sorted(row)})")
            out.append(row[key])
    return out


def prompt_hash(tok_ids):
    return int.from_bytes(hashlib.md5(",".join(map(str, tok_ids[:PROMPT])).encode()).digest()[:8], "little", signed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task name; output stem")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scorer", required=True, help="HF model dir")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--kind", required=True, choices=["synth", "null", "canary"])
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--real-ref", default="", help="mixed corpora: real slice; rows in it are dropped")
    ap.add_argument("--synth-ref", default="", help="mixed corpora: every kept row must be in this file")
    ap.add_argument("--expect-synth", type=int, default=0, help="mixed corpora: exact synthetic row count")
    ap.add_argument("--expect-rows", type=int, default=18686)
    ap.add_argument("--enforce-k3", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dtype", default="float32", choices=["float32"])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    npz, js = (os.path.join(a.out_dir, a.task + s) for s in (".npz", ".json"))
    if os.path.exists(npz) or os.path.exists(js):
        sys.exit(f"REFUSE: output exists for {a.task}; never overwrite")

    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    texts = load_texts(a.corpus, a.text_key)
    if len(texts) != a.expect_rows:
        sys.exit(f"REFUSE: {a.corpus} has {len(texts)} rows, expected {a.expect_rows}")
    row_idx = list(range(len(texts)))
    meta = {"task": a.task, "kind": a.kind, "corpus": os.path.realpath(a.corpus),
            "corpus_md5": md5_file(a.corpus), "n_rows_file": len(texts)}

    if a.real_ref:
        real = set(load_texts(a.real_ref, a.text_key))
        keep = [i for i in row_idx if texts[i] not in real]
        n_real = len(texts) - len(keep)
        if a.expect_synth and len(keep) != a.expect_synth:
            sys.exit(f"REFUSE: label count synth={len(keep)} real={n_real}, expected synth={a.expect_synth}")
        if a.synth_ref:
            synth = set(load_texts(a.synth_ref, a.text_key))
            stray = sum(texts[i] not in synth for i in keep)
            if stray:
                sys.exit(f"REFUSE: {stray} non-real rows are not in synth ref {a.synth_ref}")
            meta["synth_ref_md5"] = md5_file(a.synth_ref)
        row_idx = keep
        meta.update(real_ref_md5=md5_file(a.real_ref), n_real_dropped=n_real, n_synth_rows=len(keep))

    tok = AutoTokenizer.from_pretrained(a.scorer)
    kept, ids, excl = [], [], []
    for i in row_idx:
        t = tok(texts[i], add_special_tokens=False)["input_ids"]
        if len(t) == BLOCK:
            kept.append(i); ids.append(t)
        else:
            excl.append((i, len(t)))
    meta.update(n_candidates=len(row_idx), n_scored=len(kept), n_excluded=len(excl),
                excluded=[{"row": i, "len": L} for i, L in excl])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.scorer, torch_dtype=torch.float32).to(device).eval()
    wfile = [f for f in ("model.safetensors", "pytorch_model.bin") if os.path.exists(os.path.join(a.scorer, f))]
    meta.update(scorer=os.path.realpath(a.scorer),
                scorer_weights_md5=md5_file(os.path.join(a.scorer, wfile[0])) if wfile else None,
                device=device, dtype=a.dtype)
    t0 = time.time()
    S, A = score_ids(model, torch.tensor(ids, dtype=torch.long), device, a.batch)
    meta["seconds"] = round(time.time() - t0, 1)

    rate = float(A.sum()) / (NGEN * len(kept)) if kept else float("nan")
    meta["argmax_match_rate"] = rate
    meta["k3_enforced"] = bool(a.enforce_k3)
    try:
        meta["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        meta["git_head"] = None
    meta["script_md5"] = md5_file(os.path.abspath(__file__))

    if not np.isfinite(S).all():
        meta["verdict"] = "FAIL_NONFINITE"
    elif a.enforce_k3 and not rate >= K3_MIN:
        meta["verdict"] = "FAIL_K3"
    else:
        meta["verdict"] = "PASS" if a.enforce_k3 else "OK"
    if meta["verdict"].startswith("FAIL"):
        json.dump(meta, open(js + ".tmp", "w"), indent=1); os.replace(js + ".tmp", js)
        print(f"{a.task}: {meta['verdict']} argmax_match_rate={rate:.5f} n_scored={len(kept)}")
        sys.exit(3)

    if a.kind != "canary":
        ph = np.array([prompt_hash(t) for t in ids], dtype="int64")
        np.savez(npz + ".tmp.npz", row=np.array(kept, dtype="int64"), S=S, n_argmax=A, prompt_hash=ph)
        os.replace(npz + ".tmp.npz", npz)
    json.dump(meta, open(js + ".tmp", "w"), indent=1); os.replace(js + ".tmp", js)
    # Deliberately no S statistics printed.
    print(f"{a.task}: {meta['verdict']} kind={a.kind} n_scored={len(kept)} n_excluded={len(excl)} "
          f"argmax_match_rate={rate:.5f} k3_enforced={a.enforce_k3} {meta['seconds']}s")


if __name__ == "__main__":
    main()
