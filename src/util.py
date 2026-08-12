"""Shared utilities for the collapse pipeline."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    workdir = os.path.expandvars(cfg["paths"]["workdir"])
    cfg["paths"]["workdir"] = workdir
    Path(workdir).mkdir(parents=True, exist_ok=True)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gen_dir(cfg, generation, seed=None):
    """Directory for one generation's artifacts: model/, corpus.jsonl, eval/, metrics.json"""
    seed = seed if seed is not None else cfg["seed"]
    d = Path(cfg["paths"]["workdir"]) / cfg["run_name"] / f"seed{seed}" / f"gen{generation}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def offline_ok():
    """Warn loudly if we're on a compute node without offline flags set."""
    flags = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]
    if not any(os.environ.get(f) == "1" for f in flags):
        print("[warn] HF offline flags not set. On Narval compute nodes this WILL hang "
              "on network calls. Run scripts/narval_env.sh first.")
