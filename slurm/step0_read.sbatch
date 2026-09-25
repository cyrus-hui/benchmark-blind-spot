#!/bin/bash
#SBATCH --account=def-enaskt
#SBATCH --job-name=step0_read
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/step0/read-%j.out
# 7.1 step 0 read. CPU only. Submit from ~/modelcollapse, once, after --check passed and the analysis is pushed.
set -eo pipefail
cd "${REPO:-$HOME/modelcollapse}"
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0
source scripts/narval_env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "read job $SLURM_JOB_ID head $(git rev-parse --short HEAD) node $(hostname)"
python -u scripts/step0_analysis.py
