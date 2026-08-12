# Collapse Pipeline — Phase 2 (Pilot)

Iterative synthetic fine-tuning pipeline measuring **when each metric family first
detects model collapse**. Replace paradigm, restart-from-base each generation
(Shumailov/Drayson regime). Main decoding condition: top-k (k=50) sampling.

Pilot: GPT-2 Medium, 3 generations (0–2), 1 seed, all 7 metrics.

## Narval setup (one-time, login node)

```bash
ssh cyrh@narval.computecanada.ca
git clone <your-repo-url> ~/collapse-pipeline   # or scp this folder up
cd ~/collapse-pipeline

module load StdEnv/2023 python/3.11 scipy-stack/2024a cuda/12.2
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# If numpy/scipy conflicts with scipy-stack module:  pip install --ignore-installed numpy

# Download every model + dataset (compute nodes have NO internet):
bash scripts/download_assets.sh

# Build corpora + the fixed eval prompt set (CPU, ~2 min):
source scripts/narval_env.sh        # turns offline flags ON — cache is populated now
python src/data_prep.py --config configs/pilot_gpt2m.yaml
```

## Run the pilot

```bash
bash scripts/submit_pilot.sh        # gens 0→1→2 as a slurm dependency chain
squeue -u cyrh                      # monitor; email on END/FAIL
```

Or interactively for debugging:
```bash
salloc --account=def-enaskt_gpu --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=3:00:00
cd ~/collapse-pipeline && source scripts/narval_env.sh
bash slurm/run_generation.sbatch 0       # works as a plain script too
```

## Collect results

```bash
python src/collect_results.py --config configs/pilot_gpt2m.yaml
# -> $SCRATCH/collapse/pilot_gpt2m/results_table.csv  (the Experimental Log table)
```

## What one generation step does

```
corpus_g ──finetune (from PRISTINE base)──> model_g
model_g ──generate on FIXED eval prompts──> eval_completions  ──> 5 metrics
model_g ──lm-eval──────────────────────────> HellaSwag, MMLU
model_g ──generate on corpus_g prefixes────> synthetic corpus_{g+1}
```

Artifacts land in `$SCRATCH/collapse/pilot_gpt2m/seed{S}/gen{G}/`:
`model/`, `eval_completions.jsonl`, `metrics.json`, `benchmarks/`, `synthetic_corpus.jsonl`.

## Pilot pass/fail criteria (decision checkpoint)

1. **All 7 metrics compute at gen 0** without NaN/inf.
2. **Benchmark floor gate:** gen-0 HellaSwag must be clearly above 25% chance
   (GPT-2 Medium ≈ 33% acc / ~40% acc_norm — should pass). **MMLU will be at ~25%
   floor for GPT-2 Medium — expected.** If so, that is the empirical justification
   for moving main runs to SmolLM2-1.7B and/or swapping MMLU → ARC-Easy.
3. **Directionality by gen 2:** PPL ↑, MAUVE ↓ (expect the biggest move),
   KL ↑. Distinct-2 may be flat or even rise (Drayson saw this under pure
   sampling — non-monotonicity is a finding, not a bug).
4. Wall-time per generation fits in 6 h (it should, by a wide margin).

## Narval gotchas baked into this repo

- `HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_DATASETS_OFFLINE=1` set by
  `narval_env.sh` — every `from_pretrained` also passes `local_files_only=True`.
- `WANDB_MODE=offline` (sync from a login node later if you want wandb at all;
  the pilot logs JSON + CSV and doesn't need it).
- MAUVE featurizer (`gpt2-large`) and lm-eval datasets are pre-downloaded by
  `download_assets.sh` — forgetting any one of these hangs a compute job.
- Padding side is `left` for generation (decoder-only requirement).

## Design notes to confirm with supervisor (sign-off meeting)

- **Prompt recursion:** generation-g synthetic data is conditioned on prefixes from
  generation g-1's corpus (fully recursive corruption). Alternative: always prompt
  from real prefixes (Drayson-style). Both defensible; ours is the stricter stress test.
- **Full FT vs LoRA:** pilot uses full fine-tuning (cheap at 345M, matches Drayson,
  avoids the expressivity confound — Shumailov's error source #2). Decide for 1.7B.
- **Benchmark floor:** see pass/fail #2 — likely forces SmolLM2-1.7B + ARC-Easy
  for Phase 3 main runs. Bring gen-0 numbers to the meeting.
