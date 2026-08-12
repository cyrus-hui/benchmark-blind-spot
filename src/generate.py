"""Generate text from a fine-tuned generation-g model. Two modes:

  --mode corpus : synthetic TRAINING corpus for generation g+1. Prompts are the first
                  prompt_len tokens of each chunk in generation g's own training corpus
                  (so gen-1 prompts come from real data; gen-2 prompts from gen-1's
                  synthetic corpus, etc. — corruption recurses through the prompts too).
                  Stored text = prompt + continuation, same chunk count and block size
                  as the original corpus (Shumailov's 'same size' protocol).

  --mode eval   : completions for the FIXED eval prompt set (identical prompts every
                  generation). All distribution/diversity metrics are computed on these.

Decoding is the locked main condition (top-k 50) unless overridden in the config.
"""
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import load_config, set_seed, gen_dir, read_jsonl, write_jsonl, device, offline_ok


def build_gen_kwargs(cfg):
    d = cfg["decoding"]
    kw = dict(do_sample=True, temperature=d["temperature"],
              max_new_tokens=d["max_new_tokens"])
    if d["strategy"] == "top_k":
        kw["top_k"] = d["top_k"]
    elif d["strategy"] == "nucleus":
        kw["top_p"] = d["top_p"]; kw["top_k"] = 0
    elif d["strategy"] == "temperature":
        kw["top_k"] = 0; kw["top_p"] = 1.0
    elif d["strategy"] == "greedy":
        kw = dict(do_sample=False, max_new_tokens=d["max_new_tokens"])
    else:
        raise ValueError(f"unknown decoding strategy {d['strategy']}")
    return kw


@torch.no_grad()
def batched_generate(model, tok, prompts, gen_kwargs, batch_size=64):
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to(model.device)
        ids = model.generate(**enc, **gen_kwargs, pad_token_id=tok.pad_token_id)
        new = ids[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
        if (i // batch_size) % 10 == 0:
            print(f"  {i + len(batch)}/{len(prompts)}")
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--generation", type=int, required=True, help="model generation g")
    ap.add_argument("--mode", choices=["corpus", "eval"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed + args.generation)  # decouple sampling noise across generations
    offline_ok()

    gd = gen_dir(cfg, args.generation, seed)
    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]
    tok = AutoTokenizer.from_pretrained(gd / "model", local_files_only=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        gd / "model", local_files_only=True, torch_dtype=torch.bfloat16).to(device()).eval()

    gen_kwargs = build_gen_kwargs(cfg)
    pl = cfg["data"]["prompt_len"]

    if args.mode == "corpus":
        # source corpus = what THIS generation was trained on
        if args.generation == 0:
            src_rows = read_jsonl(root / "real_train.jsonl")
        else:
            src_rows = read_jsonl(gen_dir(cfg, args.generation - 1, seed) / "synthetic_corpus.jsonl")
        prompts = []
        for r in src_rows:
            ids = tok(r["text"], return_tensors=None)["input_ids"][:pl]
            prompts.append(tok.decode(ids))
        conts = batched_generate(model, tok, prompts, gen_kwargs, args.batch_size)
        out = gd / "synthetic_corpus.jsonl"
        write_jsonl(out, [{"text": p + c} for p, c in zip(prompts, conts)])
        print(f"synthetic corpus for gen{args.generation + 1}: {len(conts)} chunks -> {out}")

    else:  # eval
        n_eval = cfg["eval"].get("mauve_n_eval_prompts", cfg["eval"]["n_eval_prompts"])
        all_rows = read_jsonl(root / "eval_prompts.jsonl")
        rows = all_rows[:n_eval]
        gk = dict(gen_kwargs); gk["max_new_tokens"] = cfg["eval"]["eval_max_new_tokens"]
        conts = batched_generate(model, tok, [r["prompt"] for r in rows], gk, args.batch_size)
        out = gd / "eval_completions.jsonl"
        write_jsonl(out, [{"prompt": r["prompt"], "model": c,
                           "human": r["human_continuation"]}
                          for r, c in zip(rows, conts)])
        print(f"eval completions: {len(conts)} -> {out}")


if __name__ == "__main__":
    main()
