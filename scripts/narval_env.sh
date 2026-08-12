#!/bin/bash
# Source this before ANY pipeline command on Narval:  source scripts/narval_env.sh
# Works on both login and compute nodes.

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

# Project venv (separate from pollution_route_pipeline)
VENV="$HOME/modelcollapse/venv"
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
fi

# HF caches on scratch (matches your existing ~/.bash_profile, repeated here defensively)
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"

# Compute nodes have NO outbound internet. Force offline so transformers/datasets/
# lm-eval never attempt a network call (which hangs, then kills the job at timeout).
# Login nodes: temporarily `unset` these when downloading.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# wandb cannot reach the internet from compute nodes — log offline, sync later.
export WANDB_MODE=offline

export TOKENIZERS_PARALLELISM=false
