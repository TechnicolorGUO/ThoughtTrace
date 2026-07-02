#!/usr/bin/env bash
# ThoughtTrace on-policy distillation | vLLM rollout | FSDP training | NVIDIA GPUs

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"

# ---- user-adjustable ----
MODEL_PATH="${MODEL_PATH:-${HOME}/autodl-fs/beichen/public_models/Qwen3.5-4B}"
STUDENT_MODEL="${STUDENT_MODEL:-$MODEL_PATH}"
TEACHER_MODEL="${TEACHER_MODEL:-${HOME}/autodl-fs/beichen/public_models/Qwen3.5-9B}"

VERL_ROOT="${VERL_ROOT:-${WORKSPACE_ROOT}/OPD/verl}"

TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/data/processed_en/user_sim_train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_ROOT}/data/processed_en/user_sim_val.parquet}"
CONVERT_DATA_IF_MISSING="${CONVERT_DATA_IF_MISSING:-True}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-3}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-1}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-32}

train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

actor_lr=${ACTOR_LR:-5e-7}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.7}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-20}

project_name=${PROJECT_NAME:-thoughttrace_opd}
student_model_name="${STUDENT_MODEL##*/}"
teacher_model_name="${TEACHER_MODEL##*/}"
timestamp=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
experiment_name=${EXPERIMENT_NAME:-thoughttrace-stu-${student_model_name}-tch-${teacher_model_name}-topk-${distillation_topk}-${timestamp}}
ckpt_dir=${CKPT_DIR:-${PROJECT_ROOT}/output/opd/${experiment_name}}
# ---- end user-adjustable ----

if [[ "$CONVERT_DATA_IF_MISSING" == "True" && ! -f "$TRAIN_DATA" ]]; then
    cd "$PROJECT_ROOT"
    python scripts/convert_swift_sft_to_verl_opd_parquet.py \
        --train-input data/processed_en/user_sim_train.jsonl \
        --val-input data/processed_en/user_sim_val.jsonl \
        --train-output "$TRAIN_DATA" \
        --val-output "$VAL_DATA"
fi

train_files="['$TRAIN_DATA']"
val_files="['$VAL_DATA']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))
########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.prompt_key=prompt
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.default_local_dir=${ckpt_dir}
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### launch ###########################
cd "$VERL_ROOT"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
