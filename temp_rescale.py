#!/usr/bin/env python3
"""Temperature-rescaled benchmark scoring.

Subclasses lm-eval 0.4.12's HFLM and divides logits by T before they reach
lm-eval's own loglikelihood / byte-normalisation / acc_norm machinery. Every
downstream step is lm-eval's, so T=1 must reproduce stored acc_norm exactly.

Usage:
  python temp_rescale.py <model_dir> <T> [--tasks arc_easy,hellaswag] [--limit N]
"""
import sys, json, argparse
import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM


class TemperatureHFLM(HFLM):
    def __init__(self, *a, temperature=1.0, **kw):
        super().__init__(*a, **kw)
        self._T = float(temperature)

    def _model_call(self, inps, attn_mask=None, labels=None):
        logits = super()._model_call(inps, attn_mask=attn_mask, labels=labels)
        return logits if self._T == 1.0 else logits / self._T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("temperature", type=float)
    ap.add_argument("--tasks", default="arc_easy,hellaswag")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lm = TemperatureHFLM(
        pretrained=args.model_dir, dtype="bfloat16",
        trust_remote_code=True, batch_size="auto",
        temperature=args.temperature,
    )
    res = simple_evaluate(model=lm, tasks=args.tasks.split(","),
                          num_fewshot=0, limit=args.limit)

    out = {t: {k: v for k, v in m.items() if "acc" in k}
           for t, m in res["results"].items()}
    print(json.dumps({"T": args.temperature, "results": out}, indent=2))
    if args.out:
        import os
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"T": args.temperature, "model_dir": args.model_dir,
                       "results": res["results"]}, f, indent=2)


if __name__ == "__main__":
    main()
