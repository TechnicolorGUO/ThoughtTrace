#!/usr/bin/env bash
# Utility 2 (Phase 4) end-to-end runbook: build DPO data -> train -> eval.
# Run from the repo root on a GPU box. Edit the CONFIG block, then run stages
# individually (recommended) or all at once.
#
#   bash scripts/run_util2.sh smoke    # 0. build 8 pairs + run 2 train steps (4090 sanity check)
#   bash scripts/run_util2.sh build    # 1. build DPO pairs (needs an LLM endpoint)
#   bash scripts/run_util2.sh train    # 2. DPO-train the arms (GPU)
#   bash scripts/run_util2.sh eval     # 3. Arena-Hard win rates (GPU + judge)
#
set -euo pipefail

# ============================ CONFIG ========================================
# Rewriter / classifier endpoint used by `build`. Either:
#   (a) a hosted API (cheap, stronger rewrites):
export REWRITER_BASE="https://api.deepseek.com"
export REWRITER_MODEL="deepseek-v4-flash"
export REWRITER_KEY="${DEEPSEEK_KEY:?set DEEPSEEK_KEY in your env, do not hardcode}"
#   (b) or a local vLLM Qwen3-8B: set REWRITER_BASE=http://localhost:8000/v1 etc.

BASE_MODEL="Qwen/Qwen3-8B"          # DPO base / Arena-Hard base row
OUT="outputs/phase4"
ARENA_QUESTIONS="data/arena_hard/question.jsonl"   # from lmarena/arena-hard-auto
N_THOUGHT=1000
N_MESSAGE=450
SEED=0
# ============================================================================

stage="${1:-all}"

# Write a standalone config pointing at the rewriter endpoint, passed to
# build_dpo_data via --config. default.yaml is never touched.
write_build_config() {
  cat > configs/_build.yaml <<YAML
seed: ${SEED}
llm:
  model: ${REWRITER_MODEL}
  base_url: ${REWRITER_BASE}
  api_key: ${REWRITER_KEY}
  temperature: 0.0
  max_tokens: 1024
  enable_thinking: null
  cache_dir: outputs/llm_cache_build
judge:
  model: ${REWRITER_MODEL}
  base_url: ${REWRITER_BASE}
  api_key: ${REWRITER_KEY}
  temperature: 0.0
YAML
}

build() {
  echo "=== [build] DPO data (thought-guided + message-guided/TT) ==="
  write_build_config
  python -m src.phase4_utility_alignment.build_dpo_data --config configs/_build.yaml \
      --arm thought    --n "${N_THOUGHT}" --seed "${SEED}"
  python -m src.phase4_utility_alignment.build_dpo_data --config configs/_build.yaml \
      --arm message_tt --n "${N_MESSAGE}" --seed "${SEED}"
  echo "built: ${OUT}/dpo_thought_guided.jsonl  ${OUT}/dpo_message_guided_tt.jsonl"
  # Optional 4th arm (needs WildChat): wire WildChat convs into
  # collect_message_candidates and run --arm message_wildchat --wildchat PATH.
}

train() {
  echo "=== [train] DPO from ${BASE_MODEL} (one run per arm) ==="
  python -m src.phase4_utility_alignment.train_dpo \
      --pairs "${OUT}/dpo_thought_guided.jsonl" \
      --out   "${OUT}/ckpt_thought_guided" --base "${BASE_MODEL}" --seed "${SEED}"
  python -m src.phase4_utility_alignment.train_dpo \
      --pairs "${OUT}/dpo_message_guided_tt.jsonl" \
      --out   "${OUT}/ckpt_message_tt" --base "${BASE_MODEL}" --seed "${SEED}"
  echo "checkpoints: ${OUT}/ckpt_thought_guided  ${OUT}/ckpt_message_tt"
}

eval_arena() {
  echo "=== [eval] Arena-Hard win rate vs base ==="
  echo "Serve base + each checkpoint as OpenAI-compatible endpoints first, e.g.:"
  echo "  vllm serve ${BASE_MODEL}              --port 8000 &"
  echo "  vllm serve ${OUT}/ckpt_thought_guided --port 8001 --enable-lora &"
  echo "  vllm serve ${OUT}/ckpt_message_tt     --port 8002 --enable-lora &"
  echo "Then (self-contained judge mode):"
  python -m src.phase4_utility_alignment.eval_arenahard \
      --models base=http://localhost:8000/v1 \
               thought=http://localhost:8001/v1 \
               message_tt=http://localhost:8002/v1 \
      --baseline base --self-contained \
      --questions "${ARENA_QUESTIONS}" --seed "${SEED}"
  echo "Expected ranking: thought > message_tt > base"
}

smoke() {
  echo "=== [smoke] tiny build (8 pairs) + 2 train steps — 4090/24GB sanity check ==="
  write_build_config
  python -m src.phase4_utility_alignment.build_dpo_data --config configs/_build.yaml \
      --arm thought --n 8 --seed "${SEED}"
  python -m src.phase4_utility_alignment.train_dpo \
      --pairs "${OUT}/dpo_thought_guided.jsonl" \
      --out   "${OUT}/ckpt_smoke" --base "${BASE_MODEL}" \
      --max-steps 2 --per-device-batch 1 --batch-size 2 --seed "${SEED}"
  echo "smoke done — if it printed 'SMOKE OK', the 4-bit QLoRA DPO pipeline runs on your GPU."
}

case "${stage}" in
  smoke) smoke ;;
  build) build ;;
  train) train ;;
  eval)  eval_arena ;;
  all)   build; train; eval_arena ;;
  *) echo "usage: $0 {smoke|build|train|eval|all}"; exit 1 ;;
esac
