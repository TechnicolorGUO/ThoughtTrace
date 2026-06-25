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
    ap.add_argument("--batch-size", type=int, default=64, help="effective batch (via grad accum)")
    ap.add_argument("--per-device-batch", type=int, default=1, help="micro-batch on the GPU")
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap optimizer steps; >0 enables SMOKE mode (no eval/early-stop)")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    ap.add_argument("--max-length", type=int, default=1024, help="total seq length cap")
    ap.add_argument("--max-prompt-length", type=int, default=768)
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    # 4-bit QLoRA is the default — an 8B model won't fit on 24GB otherwise.
    ap.add_argument("--4bit", dest="four_bit", action="store_true", default=True)
    ap.add_argument("--bf16-full", dest="four_bit", action="store_false",
                    help="full bf16 weights instead of 4-bit (needs ~40GB+)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    smoke = args.max_steps and args.max_steps > 0

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
    print(f"[train_dpo] base={base} pairs={args.pairs} out={args.out} seed={args.seed} "
          f"4bit={args.four_bit} smoke={smoke}")

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

    # 4-bit QLoRA: load the base quantized so 8B weights drop from ~16GB to ~5.5GB,
    # leaving room for DPO's 4 forward passes (policy+ref x chosen+rejected) on 24GB.
    quant_config = None
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto" if args.four_bit else None,
    )
    if args.four_bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)

    cfg = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=max(1, args.batch_size // args.per_device_batch),
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,                 # -1 = use epochs; >0 = smoke cap
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        gradient_checkpointing=True,              # trade compute for memory on 24GB
        eval_strategy="no" if smoke else "epoch",
        save_strategy="no" if smoke else "epoch",
        load_best_model_at_end=not smoke,
        metric_for_best_model=None if smoke else "eval_loss",
        greater_is_better=False,
        logging_steps=1 if smoke else 10,
        bf16=True,
        seed=args.seed,
        report_to=[],
    )

    callbacks = []
    if not smoke:
        from transformers import EarlyStoppingCallback

        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    trainer = DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=None if smoke else eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    trainer.train()

    if smoke:
        print(f"[train_dpo] SMOKE OK — ran {args.max_steps} step(s), pipeline works. "
              "Not saving a checkpoint. Drop --max-steps for the real run.")
        return

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    Path(args.out, "train_meta.json").write_text(json.dumps({
        "base": base, "pairs": args.pairs, "n_train": len(train_ds),
        "n_val": len(eval_ds), "batch_size": args.batch_size, "lr": args.lr,
        "epochs": args.epochs, "beta": args.beta, "lora": args.lora,
        "four_bit": args.four_bit, "max_length": args.max_length, "seed": args.seed,
    }, indent=2), encoding="utf-8")
    print(f"[train_dpo] saved -> {args.out}")


if __name__ == "__main__":
    main()
