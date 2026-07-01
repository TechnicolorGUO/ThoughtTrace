#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_PATH="${MODEL_PATH:-${HOME}/autodl-fs/beichen/public_models/Qwen3.5-4B}"
case "$MODEL_PATH" in
  "~")
    MODEL_PATH="$HOME"
    ;;
  "~/"*)
    MODEL_PATH="${HOME}/${MODEL_PATH#\~/}"
    ;;
esac
if [[ "$MODEL_PATH" != /* ]]; then
  echo "MODEL_PATH must be an absolute path, got: $MODEL_PATH" >&2
  exit 1
fi

TRAIN_DATA="${TRAIN_DATA:-data/processed_label_stage/user_sim_train.jsonl}"
VAL_DATA="${VAL_DATA:-data/processed_label_stage/user_sim_val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35-4b-thoughttrace-user-sim-lora-label-stage}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

swift sft \
  --model "$MODEL_PATH" \
  --tuner_type lora \
  --dataset "$TRAIN_DATA" \
  --val_dataset "$VAL_DATA" \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --learning_rate 1e-4 \
  --lora_rank 16 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --max_length "$MAX_LENGTH" \
  --loss_scale last_round \
  --deepspeed zero2 \
  --save_steps 100 \
  --eval_steps 100 \
  --logging_steps 10 \
  --save_total_limit 3 \
  --warmup_ratio 0.05 \
  --output_dir "$OUTPUT_DIR"
