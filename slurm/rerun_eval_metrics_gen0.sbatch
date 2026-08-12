#!/bin/bash
#SBATCH --account=def-enaskt_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=28G
#SBATCH --time=2:00:00
#SBATCH --mail-user=cyrhui@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --job-name=smollm2-reeval-gen0
#SBATCH --output=logs/misc/%x-%j.out

# Re-run generate --mode eval and metrics for gen-0, all 5 seeds,
# now that eval_prompts.jsonl has 2000 rows and generate.py is patched.
# Fine-tune and benchmarks are NOT re-run (models and benchmark results intact).

set -e
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0 faiss/1.12.0

cd "$HOME/modelcollapse"
source scripts/narval_env.sh

CFG=configs/smollm2_1.7b.yaml

for SEED in 43 44 45 46 47; do
    echo "=== re-eval gen 0 seed $SEED | $(date) ==="
    python src/generate.py --config $CFG --generation 0 --mode eval --seed $SEED
    python src/metrics.py  --config $CFG --generation 0 --seed $SEED
done

echo "=== reeval complete | $(date) ==="
