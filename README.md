# The Benchmark Blind Spot

Code, figure data and analysis for **"The Benchmark Blind Spot: Lag Profiles of
Distributional Model Collapse and a Low-Cost Calibration-Gap Signal"**
(NeurIPS 2026 workshops; arXiv link to follow).

Recursive fine-tuning on synthetic predecessor output degrades perplexity,
diversity and distributional divergence from generation 1, while
likelihood-scored benchmarks lag, stay silent, or move the wrong way. We
measure the lag across two scales (GPT-2 Medium 345M, SmolLM2-1.7B), two
decoders (top-k, greedy), 8–13 generations and 3–5 seeds, and propose the
calibration gap (log PPL − predictive entropy) as a candidate monitor.

## What is here now

| path | contents |
|---|---|
| `figures/figdata.json` | Canonical per-seed, per-generation data behind every figure and table |
| `figures/tables.txt`, `figures/p0b_items.json` | Rendered tables and the item-level export behind Appendix B |
| `figures/*.pdf` | The nine figures as submitted |
| `make_figures.py`, `render_figures.py`, `p0b_figure.py`, `seed_variance.py` | Harvest → cache → render pipeline; `make_figures.py` carries a 23-anchor canary against the paper's numbers |
| `gather_*.py`, `check_benchmark_sig.py`, `rate_shape.py`, `power_check.py`, `margin_analysis.py` | Analysis and statistics |
| `src/` | Fine-tuning (`finetune.py`, restart-from-base), generation, metrics, OOD/entropy scoring, data preparation |
| `configs/` | One YAML per arm |
| `slurm/`, `scripts/` | SLURM drivers and chain submission scripts (Alliance/DRAC clusters) |
| `logs/` | SLURM logs for every reported job |
| `threshold_transfer.py` | Post-submission: leave-one-scale-out threshold transfer of the calibration gap |

## One-off controls and verification scripts

These are kept deliberately; each supports a reported result.

- `solve_temperature.py`, `temp_rescale.py`, `slurm/temp_gen0_array.sbatch` — entropy-matched temperature rescaling and its generation-0 control (Limitation 1; Appendix Table 4)
- `src/score_ood_ppl.py` + `slurm/ood_manifest.txt` — register-matched and far-OOD perplexity, predictive entropy, calibration gap (§4.3, Appendix E)
- `download_ood_wiki.py`, `scripts/download_ood.py` — the held-out register slices (article-disjoint, 13-gram leakage-screened)
- `slurm/arc_*_array.sbatch`, `slurm/logsamples.sbatch` — ARC re-evaluation and the `--log_samples` tree behind Appendix B
- `slurm/repair_benchmarks.sbatch`, `slurm/rerun_eval_metrics_gen0.sbatch` — benchmark and generation-0 metric re-evaluations

## Coming

Generated corpora and frozen evaluation slices with checksums (Zenodo record;
DOI to be added here). Fine-tuned checkpoints are not released for size.

## Reproducing the numbers

    source scripts/narval_env.sh            # or an equivalent venv from requirements.txt
    python make_figures.py                  # rebuilds figures/ from figdata.json; canary must print 23/23
    python threshold_transfer.py figures/figdata.json

Rerunning the pipeline end to end requires a SLURM cluster; see `scripts/submit_*.sh`.

## License

MIT.
