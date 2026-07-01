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

ADAPTER_PATH="${ADAPTER_PATH:-}"
if [[ -z "$ADAPTER_PATH" ]]; then
  echo "Usage: ADAPTER_PATH=/path/to/checkpoint bash scripts/infer_qwen35_lora_swift_label_stage.sh" >&2
  exit 1
fi

VAL_DATA="${VAL_DATA:-data/processed_label_stage/user_sim_val.jsonl}"
EVAL_DATA="${EVAL_DATA:-data/processed_label_stage/user_sim_eval_examples.jsonl}"
ANSWERS_PATH="${ANSWERS_PATH:-data/processed_label_stage/user_sim_eval_answers.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-output/qwen35-4b-thoughttrace-user-sim-lora-label-stage-infer}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0}"

python scripts/make_eval_examples.py \
  --input "$VAL_DATA" \
  --output "$EVAL_DATA" \
  --answers "$ANSWERS_PATH" \
  --num-examples "${NUM_EXAMPLES:-5}"

swift infer \
  --model "$MODEL_PATH" \
  --adapters "$ADAPTER_PATH" \
  --val_dataset "$EVAL_DATA" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMPERATURE" \
  --result_path "$OUTPUT_DIR/results.jsonl"
