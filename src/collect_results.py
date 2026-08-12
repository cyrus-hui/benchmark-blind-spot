"""Collect every gen*/metrics.json and lm-eval result into one CSV — fills the
Experimental Log table from the project plan. Run anywhere (login node fine).
"""
import argparse
import csv
import json
from pathlib import Path

from util import load_config

METRIC_COLS = ["perplexity", "mmlu", "hellaswag", "distinct2",
               "emb_variance", "kl_unigram", "mauve"]


def harvest_lm_eval(gen_path):
    """lm-eval writes results under gen{g}/benchmarks/**/results_*.json"""
    out = {}
    bdir = gen_path / "benchmarks"
    if not bdir.exists():
        return out
    for f in sorted(bdir.rglob("results*.json")):
        data = json.loads(f.read_text())
        for task, vals in data.get("results", {}).items():
            for key in ("acc,none", "acc_norm,none", "acc"):
                if key in vals:
                    base = "mmlu" if task.startswith("mmlu") else task
                    out[base] = vals[key]
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["workdir"]) / cfg["run_name"]

    rows = []
    for seed_dir in sorted(root.glob("seed*")):
        for gen_path in sorted(seed_dir.glob("gen*"), key=lambda p: int(p.name[3:])):
            mfile = gen_path / "metrics.json"
            row = {"seed": seed_dir.name, "generation": int(gen_path.name[3:])}
            if mfile.exists():
                row.update(json.loads(mfile.read_text()))
            row.update(harvest_lm_eval(gen_path))
            rows.append(row)

    out = root / "results_table.csv"
    cols = ["seed", "generation"] + METRIC_COLS
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in r.items() if k in cols})


if __name__ == "__main__":
    main()
