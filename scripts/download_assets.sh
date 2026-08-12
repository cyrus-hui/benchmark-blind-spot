#!/bin/bash
# Run ONCE on a LOGIN node (compute nodes have no internet).
# Downloads every model and dataset the offline pipeline needs into $SCRATCH/hf_cache.
set -e

module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.11 scipy-stack/2024a arrow/17.0.0
source "$HOME/modelcollapse/venv/bin/activate"
export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$SCRATCH/hf_cache/datasets"
# downloads need the network — make sure offline flags are OFF here
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

python - <<'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

print("== base models ==")
for name in ["gpt2-medium", "HuggingFaceTB/SmolLM2-1.7B"]:
    print(name)
    AutoTokenizer.from_pretrained(name)
    AutoModelForCausalLM.from_pretrained(name)

print("== MAUVE featurizer (gpt2-large) ==")
AutoTokenizer.from_pretrained("gpt2-large")
AutoModelForCausalLM.from_pretrained("gpt2-large")

print("== SBERT ==")
SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("== datasets ==")
load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
load_dataset("hellaswag", trust_remote_code=True)          # lm-eval task: hellaswag
load_dataset("cais/mmlu", "all", trust_remote_code=True)   # lm-eval task: mmlu

print("ALL DOWNLOADS COMPLETE")
EOF

# Disk check — gpt2-medium ~1.5G, smollm2-1.7B ~3.5G, gpt2-large ~3G, datasets ~1G
echo "cache size:"
du -sh "$SCRATCH/hf_cache"
