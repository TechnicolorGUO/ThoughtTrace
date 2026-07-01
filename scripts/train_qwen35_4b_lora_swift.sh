#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/autodl-fs/beichen/public_models/Qwen3.5-4B}"
TRAIN_DATA="${TRAIN_DATA:-ThoughtTrace/data/processed_en/user_sim_train.jsonl}"
VAL_DATA="${VAL_DATA:-ThoughtTrace/data/processed_en/user_sim_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35-4b-thoughttrace-user-sim-lora}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$VAL_DATA" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length 4096 \
  --loss_scale last_round \
  --deepspeed zero2 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --warmup_ratio 0.05 \
  --output_dir "$OUTPUT_DIR"
