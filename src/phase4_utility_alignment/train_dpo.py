"""Phase 4 / 7B — DPO training (Appendix D.9).

Trains one DPO model per arm from the base backbone (Qwen3-8B), using the
preference pairs built by build_dpo_data.py. GPU-only (TRL DPOTrainer); this
module is written to run on the server, not locally.

Paper hyperparameters: batch size 64, lr 1e-6, <= 20 epochs with early stopping
on a 10% validation split. LoRA is recommended at 8B to fit memory. Seeds are
set and logged.

Heavy deps (torch, transformers, trl, peft, datasets) are imported inside main()
so the module imports cleanly on a machine without them.

Usage (server):
    python -m src.phase4_utility_alignment.train_dpo \
        --pairs outputs/phase4/dpo_thought_guided.jsonl \
        --out outputs/phase4/ckpt_thought_guided
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: str | Path) -> list[dict]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4/7B — DPO training")
    ap.add_argument("--pairs", required=True, help="DPO jsonl from build_dpo_data.py")
    ap.add_argument("--out", required=True, help="output checkpoint dir")
    ap.add_argument("--base", default=None, help="base model (default: config llm.model)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # heavy imports, server-only
    import random

    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from ..config import load_config

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = args.base or load_config()["llm"]["model"]
    print(f"[train_dpo] base={base} pairs={args.pairs} out={args.out} seed={args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_pairs(args.pairs)

    def to_row(p: dict) -> dict:
        # apply the chat template to the conversational prompt; chosen/rejected
        # are the assistant completions DPOTrainer compares.
        prompt = tokenizer.apply_chat_template(
            p["prompt"], tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt, "chosen": p["chosen"], "rejected": p["rejected"]}

    rows = [to_row(p) for p in pairs]
    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    train_ds = Dataset.from_list(rows[n_val:])
    eval_ds = Dataset.from_list(rows[:n_val])
    print(f"[train_dpo] train={len(train_ds)} val={len(eval_ds)}")

    peft_config = None
    if args.lora:
        from peft import LoraConfig

        peft_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=max(1, args.batch_size // 4),  # effective batch 64
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        beta=args.beta,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        bf16=True,
        seed=args.seed,
        report_to=[],
    )

    from transformers import EarlyStoppingCallback

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "train_meta.json").write_text(json.dumps({
        "base": base, "pairs": args.pairs, "n_train": len(train_ds),
        "n_val": len(eval_ds), "batch_size": args.batch_size, "lr": args.lr,
        "epochs": args.epochs, "beta": args.beta, "lora": args.lora, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"[train_dpo] saved -> {args.out}")


if __name__ == "__main__":
    main()
