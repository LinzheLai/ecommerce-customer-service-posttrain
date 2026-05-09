#!/usr/bin/env bash
set -euo pipefail

# DPO ablation runner.
# Experiments:
# 1. SFT-only vs SFT+DPO on the same task/preference eval set.
# 2. beta ablation: 0.05 / 0.1 / 0.3.
# 3. preference-data quality: noisy chosen/rejected swaps at 10% / 30%.
#
# Usage:
#   bash runs/ablation/run_dpo_ablation.sh
#
# Run selected experiments:
#   EXPERIMENTS="beta005_clean beta01_clean" bash runs/ablation/run_dpo_ablation.sh
#
# Evaluate only:
#   RUN_MODE=eval bash runs/ablation/run_dpo_ablation.sh
#
# Collect only:
#   RUN_MODE=collect bash runs/ablation/run_dpo_ablation.sh

ROOT_DIR="${ROOT_DIR:-/opt/data/llz/ecommerce-customer-service-posttrain}"
cd "$ROOT_DIR"

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_HOME="${HF_HOME:-/opt/data/llz/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

MODEL_NAME="${MODEL_NAME:-/opt/data/llz/hf_models/Qwen3-8B-Base}"
SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT_DIR/outputs/sft/qwen3_8b_rank16_qv_eos_short}"
CLEAN_DPO_TRAIN_PATH="${CLEAN_DPO_TRAIN_PATH:-$ROOT_DIR/data/processed_5000/taobao_dpo_train.json}"
CLEAN_DPO_VAL_PATH="${CLEAN_DPO_VAL_PATH:-$ROOT_DIR/data/processed_5000/taobao_dpo_dev.json}"
TASK_TEST_PATH="${TASK_TEST_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_test.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/configs/deepspeed_zero2.json}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/scripts/dpo/train_dpo_trl.py}"
TASK_EVAL_SCRIPT="${TASK_EVAL_SCRIPT:-$ROOT_DIR/scripts/sft/eval_sft_test.py}"
PREF_EVAL_SCRIPT="${PREF_EVAL_SCRIPT:-$ROOT_DIR/scripts/ablation/eval_dpo_preference.py}"
NOISE_SCRIPT="${NOISE_SCRIPT:-$ROOT_DIR/scripts/ablation/build_noisy_dpo_dataset.py}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-$ROOT_DIR/scripts/ablation/collect_dpo_ablation_results.py}"

ABLATION_ROOT="${ABLATION_ROOT:-$ROOT_DIR/outputs/ablation/dpo}"
REPORT_DIR="${REPORT_DIR:-$ABLATION_ROOT/reports}"
NOISY_DATA_DIR="${NOISY_DATA_DIR:-$ABLATION_ROOT/data}"
mkdir -p "$ABLATION_ROOT" "$REPORT_DIR" "$NOISY_DATA_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29710}"
RUN_MODE="${RUN_MODE:-all}"  # all / train / eval / collect
SELECTED_EXPERIMENTS="${EXPERIMENTS:-}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
LEARNING_RATE="${LEARNING_RATE:-2e-6}"
WARMUP_RATIO="${WARMUP_RATIO:-0.08}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.03}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
MAX_LENGTH="${MAX_LENGTH:-640}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-42}"

TASK_EVAL_MAX_SAMPLES="${TASK_EVAL_MAX_SAMPLES:-500}"
PREF_EVAL_MAX_SAMPLES="${PREF_EVAL_MAX_SAMPLES:-500}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-64}"
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.25}"
GEN_NO_REPEAT_NGRAM_SIZE="${GEN_NO_REPEAT_NGRAM_SIZE:-3}"
GEN_MAX_OUTPUT_CHARS="${GEN_MAX_OUTPUT_CHARS:-0}"
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是电商客服。只回答最后一句用户问题。答案必须简短、直接、保守，最多2句，不主动扩展，不重复寒暄。不确定时说需要帮您核实。}"

# Format:
#   name|group|beta|noise_ratio|train_path|description
EXPERIMENT_MATRIX=(
  "beta005_clean|beta|0.05|0.0|$CLEAN_DPO_TRAIN_PATH|beta ablation: conservative alignment"
  "beta01_clean|beta|0.1|0.0|$CLEAN_DPO_TRAIN_PATH|beta ablation: TRL/common default strength"
  "beta03_clean|beta|0.3|0.0|$CLEAN_DPO_TRAIN_PATH|beta ablation: aggressive alignment"
  "beta005_noise10|noise|0.05|0.1|$NOISY_DATA_DIR/taobao_dpo_train_noise10_seed${SEED}.json|preference data quality: 10% swapped pairs"
  "beta005_noise30|noise|0.05|0.3|$NOISY_DATA_DIR/taobao_dpo_train_noise30_seed${SEED}.json|preference data quality: 30% swapped pairs"
)

test -f "$TRAIN_SCRIPT" || { echo "train script not found: $TRAIN_SCRIPT"; exit 1; }
test -f "$TASK_EVAL_SCRIPT" || { echo "task eval script not found: $TASK_EVAL_SCRIPT"; exit 1; }
test -f "$PREF_EVAL_SCRIPT" || { echo "preference eval script not found: $PREF_EVAL_SCRIPT"; exit 1; }
test -f "$NOISE_SCRIPT" || { echo "noise script not found: $NOISE_SCRIPT"; exit 1; }
test -f "$COLLECT_SCRIPT" || { echo "collect script not found: $COLLECT_SCRIPT"; exit 1; }
test -f "$SFT_ADAPTER_DIR/adapter_config.json" || { echo "SFT adapter not found: $SFT_ADAPTER_DIR"; exit 1; }

should_run_experiment() {
  local name="$1"
  if [[ -z "$SELECTED_EXPERIMENTS" ]]; then
    return 0
  fi
  for selected in $SELECTED_EXPERIMENTS; do
    if [[ "$selected" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

start_gpu_monitor() {
  local output_dir="$1"
  local csv_path="$output_dir/gpu_usage.csv"
  echo "timestamp,index,name,memory_used_mb,memory_total_mb,utilization_gpu_pct,power_draw_w" > "$csv_path"
  (
    while true; do
      nvidia-smi \
        --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw \
        --format=csv,noheader,nounits >> "$csv_path" || true
      sleep "$GPU_MONITOR_INTERVAL"
    done
  ) >/dev/null 2>&1 &
  echo "$!"
}

stop_gpu_monitor() {
  local pid="${1:-}"
  if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
  fi
}

prepare_noisy_data() {
  python "$NOISE_SCRIPT" \
    --input_path "$CLEAN_DPO_TRAIN_PATH" \
    --output_path "$NOISY_DATA_DIR/taobao_dpo_train_noise10_seed${SEED}.json" \
    --noise_ratio 0.1 \
    --seed "$SEED"
  python "$NOISE_SCRIPT" \
    --input_path "$CLEAN_DPO_TRAIN_PATH" \
    --output_path "$NOISY_DATA_DIR/taobao_dpo_train_noise30_seed${SEED}.json" \
    --noise_ratio 0.3 \
    --seed "$SEED"
}

write_meta() {
  local output_dir="$1"
  local name="$2"
  local group="$3"
  local beta="$4"
  local noise_ratio="$5"
  local train_path="$6"
  local description="$7"
  mkdir -p "$output_dir"
  cat > "$output_dir/ablation_meta.json" <<EOF
{
  "name": "$name",
  "group": "$group",
  "description": "$description",
  "beta": $beta,
  "noise_ratio": $noise_ratio,
  "train_path": "$train_path",
  "sft_adapter_path": "$SFT_ADAPTER_DIR",
  "fixed_settings": {
    "learning_rate": "$LEARNING_RATE",
    "num_train_epochs": $NUM_TRAIN_EPOCHS,
    "label_smoothing": $LABEL_SMOOTHING,
    "loss_type": "$LOSS_TYPE",
    "completion_end_token": "eos",
    "max_completion_length": $MAX_COMPLETION_LENGTH
  }
}
EOF
}

train_one() {
  local name="$1"
  local group="$2"
  local beta="$3"
  local noise_ratio="$4"
  local train_path="$5"
  local description="$6"
  local ordinal="$7"
  local output_dir="$ABLATION_ROOT/$name"
  local master_port=$((MASTER_PORT_BASE + ordinal))

  write_meta "$output_dir" "$name" "$group" "$beta" "$noise_ratio" "$train_path" "$description"

  echo "========== Train DPO Ablation: $name =========="
  echo "description: $description"
  echo "beta:        $beta"
  echo "noise_ratio: $noise_ratio"
  echo "train_path:  $train_path"
  echo "output_dir:  $output_dir"
  echo "==============================================="

  if [[ -f "$output_dir/metrics.json" && -f "$output_dir/adapter_config.json" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    echo "[SKIP] Existing DPO adapter found: $output_dir"
    return 0
  fi

  local monitor_pid=""
  monitor_pid="$(start_gpu_monitor "$output_dir")"
  local start_ts
  start_ts="$(date +%s)"

  set +e
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$master_port" \
    "$TRAIN_SCRIPT" \
    --model_name_or_path "$MODEL_NAME" \
    --sft_adapter_path "$SFT_ADAPTER_DIR" \
    --train_path "$train_path" \
    --val_path "$CLEAN_DPO_VAL_PATH" \
    --output_dir "$output_dir" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --train_mode qlora \
    --beta "$beta" \
    --loss_type "$LOSS_TYPE" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --completion_end_token eos \
    --truncation_mode keep_end \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --gradient_checkpointing \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --seed "$SEED" \
    --trust_remote_code \
    --system_prompt "$SYSTEM_PROMPT" \
    --force_replace_system 2>&1 | tee "$output_dir/train.log"
  local train_status="${PIPESTATUS[0]}"
  set -e

  local end_ts
  end_ts="$(date +%s)"
  stop_gpu_monitor "$monitor_pid"

  cat > "$output_dir/time_summary.json" <<EOF
{
  "train_start_unix": $start_ts,
  "train_end_unix": $end_ts,
  "train_wall_time_seconds": $((end_ts - start_ts)),
  "train_exit_status": $train_status
}
EOF

  if [[ "$train_status" != "0" ]]; then
    echo "[ERROR] DPO training failed for $name with status $train_status"
    exit "$train_status"
  fi
}

eval_adapter() {
  local label="$1"
  local adapter_path="$2"
  local output_dir="$3"

  mkdir -p "$output_dir"
  echo "========== Eval Adapter: $label =========="

  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" python "$TASK_EVAL_SCRIPT" \
    --adapter_path "$adapter_path" \
    --eval_path "$TASK_TEST_PATH" \
    --output_dir "$output_dir/eval_task" \
    --max_samples "$TASK_EVAL_MAX_SAMPLES" \
    --system_prompt "$SYSTEM_PROMPT" \
    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
    --repetition_penalty "$GEN_REPETITION_PENALTY" \
    --no_repeat_ngram_size "$GEN_NO_REPEAT_NGRAM_SIZE" \
    --max_output_chars "$GEN_MAX_OUTPUT_CHARS"

  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" python "$PREF_EVAL_SCRIPT" \
    --base_model "$MODEL_NAME" \
    --adapter_path "$adapter_path" \
    --eval_path "$CLEAN_DPO_VAL_PATH" \
    --output_dir "$output_dir/eval_preference" \
    --system_prompt "$SYSTEM_PROMPT" \
    --completion_end_token eos \
    --max_samples "$PREF_EVAL_MAX_SAMPLES" \
    --max_length "$MAX_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --trust_remote_code
}

eval_sft_baseline() {
  eval_adapter "sft_only" "$SFT_ADAPTER_DIR" "$ABLATION_ROOT/sft_only"
}

eval_one() {
  local name="$1"
  local output_dir="$ABLATION_ROOT/$name"
  if [[ ! -f "$output_dir/adapter_config.json" ]]; then
    echo "[WARN] DPO adapter missing, skip eval: $output_dir"
    return 0
  fi
  eval_adapter "$name" "$output_dir" "$output_dir"
}

collect_results() {
  python "$COLLECT_SCRIPT" \
    --ablation_root "$ABLATION_ROOT" \
    --output_dir "$REPORT_DIR"
}

echo "========== DPO Ablation Plan =========="
echo "ROOT_DIR:          $ROOT_DIR"
echo "MODEL_NAME:        $MODEL_NAME"
echo "SFT_ADAPTER_DIR:   $SFT_ADAPTER_DIR"
echo "CLEAN_DPO_TRAIN:   $CLEAN_DPO_TRAIN_PATH"
echo "CLEAN_DPO_VAL:     $CLEAN_DPO_VAL_PATH"
echo "TASK_TEST_PATH:    $TASK_TEST_PATH"
echo "ABLATION_ROOT:     $ABLATION_ROOT"
echo "RUN_MODE:          $RUN_MODE"
echo "EXPERIMENTS:       ${SELECTED_EXPERIMENTS:-<all>}"
echo "======================================="

if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "train" ]]; then
  prepare_noisy_data
fi

if [[ "$RUN_MODE" == "all" || "$RUN_MODE" == "eval" ]]; then
  eval_sft_baseline
fi

ordinal=0
for row in "${EXPERIMENT_MATRIX[@]}"; do
  IFS='|' read -r name group beta noise_ratio train_path description <<< "$row"
  ordinal=$((ordinal + 1))
  if ! should_run_experiment "$name"; then
    continue
  fi

  case "$RUN_MODE" in
    all)
      train_one "$name" "$group" "$beta" "$noise_ratio" "$train_path" "$description" "$ordinal"
      eval_one "$name"
      ;;
    train)
      train_one "$name" "$group" "$beta" "$noise_ratio" "$train_path" "$description" "$ordinal"
      ;;
    eval)
      eval_one "$name"
      ;;
    collect)
      ;;
    *)
      echo "Unknown RUN_MODE: $RUN_MODE. Use all / train / eval / collect."
      exit 1
      ;;
  esac
done

collect_results

echo "========== DPO Ablation Finished =========="
echo "Report CSV: $REPORT_DIR/dpo_ablation_summary.csv"
echo "Report MD:  $REPORT_DIR/dpo_ablation_summary.md"
