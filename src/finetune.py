"""Fine-tune the BASE model on a given corpus (real for gen 0, synthetic for gen >= 1).

Replace paradigm with restart-from-base every generation — matches Shumailov et al. and
Drayson et al. (SIGMA's 'data-only recursion' regime). The corpus is the only thing that
carries forward between generations; weights never do.
"""
import argparse
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

from util import load_config, set_seed, gen_dir, read_jsonl, offline_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--generation", type=int, required=True)
    ap.add_argument("--corpus", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["seed"]
    set_seed(seed)
    offline_ok()

    out = gen_dir(cfg, args.generation, seed) / "model"

    tok = AutoTokenizer.from_pretrained(cfg["model"]["base_name"], local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ALWAYS load the pristine base — never the previous generation's weights.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_name"], local_files_only=True,
        torch_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float32,
    )

    if cfg["model"]["use_lora"]:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=cfg["model"]["lora_r"], lora_alpha=cfg["model"]["lora_alpha"],
            task_type="CAUSAL_LM"))
        model.print_trainable_parameters()

    rows = read_jsonl(args.corpus)
    bs = cfg["data"]["block_size"]

    def tokenize(batch):
        return tok(batch["text"], truncation=True, max_length=bs, padding="max_length")

    ds = Dataset.from_list(rows).map(tokenize, batched=True, remove_columns=["text"])

    targs = TrainingArguments(
        output_dir=str(out),
        num_train_epochs=cfg["training"]["num_epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        per_device_train_batch_size=cfg["training"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        bf16=cfg["training"]["bf16"],
        logging_steps=50, save_strategy="no", report_to="none", seed=seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=DataCollatorForLanguageModeling(tok, mlm=False))
    trainer.train()

    if cfg["model"]["use_lora"]:
        model = model.merge_and_unload()
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"saved gen{args.generation} model -> {out}")


if __name__ == "__main__":
    main()
