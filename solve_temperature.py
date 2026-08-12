#!/usr/bin/env python3
"""Entropy-matching temperature solver.

Sweeps a T grid over the in-domain eval corpus in ONE forward pass, then
interpolates T_g such that mean predictive entropy under log_softmax(z/T_g)
equals the generation-0 mean for the same seed.

Entropy accumulation mirrors src/score_ood_ppl.ppl_and_sharpness() exactly --
same mask, same positions, same chunking -- so the +-0.000000 PPL-reproduction
warrant carries over. T=1.0 column must reproduce the stored entropy key.

In-domain only: gen-0 baseline entropy is defined there, the calibration gap is
largest there, and a per-register solve would give four T_g with no principled
way to pick one for a register-agnostic benchmark.

Usage: python solve_temperature.py --config configs/pilot_gpt2m.yaml --seed 43
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent / "src"))
from util import load_config, set_seed, gen_dir, read_jsonl, device, offline_ok  # noqa

GRID = np.round(np.arange(1.0, 3.01, 0.05), 2)


@torch.no_grad()
def entropy_grid(model, tok, texts, block_size, grid, batch_size=32):
    ents = {float(T): 0.0 for T in grid}
    ntok = 0
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=block_size).to(model.device)
        out = model(**enc, labels=enc["input_ids"])
        logits = out.logits[:, :-1]
        mask = enc["attention_mask"][:, 1:].bool()
        flat = logits.reshape(-1, logits.size(-1)).float()
        m = mask.reshape(-1)
        for j in range(0, flat.size(0), 2048):
            sl = slice(j, min(j + 2048, flat.size(0)))
            mj = m[sl]
            if not mj.any():
                continue
            z = flat[sl][mj]
            for T in grid:
                lp = torch.log_softmax(z / float(T), dim=-1)
                ents[float(T)] += (-(lp.exp() * lp).sum(-1)).sum().item()
        ntok += mask.sum().item()
    return {T: e / ntok for T, e in ents.items()}, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed); offline_ok()
    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
    gd = gen_dir(cfg, args.generation, args.seed)
    gd0 = gen_dir(cfg, 0, args.seed)

    m0 = json.loads((gd0 / "metrics.json").read_text())
    if "entropy" not in m0:
        sys.exit(f"FATAL: no gen-0 entropy key at {gd0}/metrics.json")
    target = float(m0["entropy"])

    tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        gd / "model", local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device()).eval()

    texts = [r["text"] for r in read_jsonl(root / "real_test.jsonl")]
    curve, ntok = entropy_grid(model, tok, texts, cfg["data"]["block_size"], GRID)

    # gate: T=1.0 must reproduce the stored entropy key for THIS generation
    stored = json.loads((gd / "metrics.json").read_text()).get("entropy")
    got = curve[1.0]
    if stored is not None:
        d = got - float(stored)
        print(f"T=1 gate: {got:.6f} vs stored {float(stored):.6f} (delta {d:+.6f})")
        if abs(d) > 1e-4:
            sys.exit("FATAL: T=1 does not reproduce stored entropy -- loop diverged.")

    Ts = np.array(sorted(curve)); Es = np.array([curve[t] for t in Ts])
    if not (Es[0] <= target <= Es[-1]):
        sys.exit(f"FATAL: target {target:.4f} outside grid range "
                 f"[{Es[0]:.4f}, {Es[-1]:.4f}] -- widen GRID, do NOT clamp.")
    T_star = float(np.interp(target, Es, Ts))   # Es is increasing in T

    print(f"gen {args.generation} seed {args.seed}: target(gen0) {target:.4f} nats | "
          f"gen{args.generation} @T=1 {got:.4f} | T* = {T_star:.4f} | ntok {ntok}")

    out = gd / "temperature.json"
    out.write_text(json.dumps({
        "generation": args.generation, "seed": args.seed,
        "target_entropy_gen0": target, "entropy_at_T1": got,
        "T_star": T_star, "n_tokens": ntok,
        "grid": {str(k): v for k, v in curve.items()},
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
