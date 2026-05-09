#!/usr/bin/env bash
set -euo pipefail

# LoRA ablation runner for ecommerce customer-service SFT.
# Goals:
# 1. rank ablation: r=4 / 16 / 64
# 2. target_modules ablation: q_proj+v_proj vs all linear layers
# 3. QLoRA vs LoRA: 4bit NF4 vs FP16/BF16 full-weight loading
#
# Suggested usage on the training machine:
#   bash runs/ablation/run_lora_ablation_sft.sh
#
# Run only selected experiments:
#   EXPERIMENTS="rank4_qv_qlora rank16_qv_qlora" bash runs/ablation/run_lora_ablation_sft.sh
#
# Only evaluate trained adapters and collect reports:
#   RUN_MODE=eval bash runs/ablation/run_lora_ablation_sft.sh
#
# Only collect existing outputs:
#   RUN_MODE=collect bash runs/ablation/run_lora_ablation_sft.sh

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
TRAIN_PATH="${TRAIN_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_train.json}"
VAL_PATH="${VAL_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_dev.json}"
TEST_PATH="${TEST_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_test.json}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$ROOT_DIR/configs/deepspeed_zero2.json}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT_DIR/scripts/sft/train_sft_trl.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT_DIR/scripts/sft/eval_sft_test.py}"
COLLECT_SCRIPT="${COLLECT_SCRIPT:-$ROOT_DIR/scripts/ablation/collect_lora_ablation_results.py}"

ABLATION_ROOT="${ABLATION_ROOT:-$ROOT_DIR/outputs/ablation/lora_sft}"
REPORT_DIR="${REPORT_DIR:-$ABLATION_ROOT/reports}"
mkdir -p "$ABLATION_ROOT" "$REPORT_DIR"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29610}"
RUN_MODE="${RUN_MODE:-all}"  # all / train / eval / collect

# Keep these fixed so each ablation changes only the intended variable.
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.08}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-768}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-8}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-150}"
EVAL_STEPS="${EVAL_STEPS:-150}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SEED="${SEED:-42}"

# Evaluation generation settings. max_output_chars=0 means no display truncation.
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-64}"
GEN_REPETITION_PENALTY="${GEN_REPETITION_PENALTY:-1.25}"
GEN_NO_REPEAT_NGRAM_SIZE="${GEN_NO_REPEAT_NGRAM_SIZE:-3}"
GEN_MAX_OUTPUT_CHARS="${GEN_MAX_OUTPUT_CHARS:-0}"

# GPU monitor interval in seconds.
GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-5}"

SYSTEM_PROMPT="${SYSTEM_PROMPT:-你是电商客服。只回答最后一句用户问题。答案必须简短、直接、保守，最多2句，不主动扩展，不重复寒暄。不确定时说需要帮您核实。}"

# Format:
#   name|train_mode|rank|alpha|target_modules|description
DEFAULT_EXPERIMENT_MATRIX=(
  "rank4_qv_qlora|qlora|4|8|q_proj,v_proj|rank ablation: rank=4, low capacity"
  "rank16_qv_qlora|qlora|16|32|q_proj,v_proj|rank ablation baseline: rank=16, short SFT setting"
  "rank64_qv_qlora|qlora|64|128|q_proj,v_proj|rank ablation: rank=64, high capacity"
  "rank16_all_qlora|qlora|16|32|q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj|target_modules ablation: all common linear layers"
  "rank16_qv_lora|lora|16|32|q_proj,v_proj|QLoRA vs LoRA: FP16/BF16 base loading"
)

selected_experiments="${EXPERIMENTS:-}"

write_experiment_meta() {
  local output_dir="$1"
  local name="$2"
  local train_mode="$3"
  local rank="$4"
  local alpha="$5"
  local target_modules="$6"
  local description="$7"

  mkdir -p "$output_dir"
  cat > "$output_dir/ablation_meta.json" <<EOF
{
  "name": "$name",
  "description": "$description",
  "variable_group": "$(case "$name" in rank*) echo rank ;; *all*) echo target_modules ;; *lora) echo train_mode ;; *) echo mixed ;; esac)",
  "train_mode": "$train_mode",
  "lora_r": $rank,
  "lora_alpha": $alpha,
  "target_modules": "$target_modules",
  "fixed_settings": {
    "num_train_epochs": $NUM_TRAIN_EPOCHS,
    "learning_rate": "$LEARNING_RATE",
    "warmup_ratio": $WARMUP_RATIO,
    "max_length": $MAX_LENGTH,
    "completion_end_token": "eos",
    "system_prompt": "$SYSTEM_PROMPT"
  }
}
EOF
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

train_one() {
  local name="$1"
  local train_mode="$2"
  local rank="$3"
  local alpha="$4"
  local target_modules="$5"
  local description="$6"
  local ordinal="$7"

  local output_dir="$ABLATION_ROOT/$name"
  local master_port=$((MASTER_PORT_BASE + ordinal))

  write_experiment_meta "$output_dir" "$name" "$train_mode" "$rank" "$alpha" "$target_modules" "$description"

  echo "========== Train Ablation: $name =========="
  echo "description:    $description"
  echo "train_mode:     $train_mode"
  echo "lora_r:         $rank"
  echo "lora_alpha:     $alpha"
  echo "target_modules: $target_modules"
  echo "output_dir:     $output_dir"
  echo "master_port:    $master_port"
  echo "==========================================="

  if [[ -f "$output_dir/metrics.json" && -f "$output_dir/adapter_config.json" && "${FORCE_RETRAIN:-0}" != "1" ]]; then
    echo "[SKIP] Existing trained adapter found: $output_dir"
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
    --train_path "$TRAIN_PATH" \
    --val_path "$VAL_PATH" \
    --output_dir "$output_dir" \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --train_mode "$train_mode" \
    --lora_r "$rank" \
    --lora_alpha "$alpha" \
    --lora_dropout "$LORA_DROPOUT" \
    --target_modules "$target_modules" \
    --completion_end_token eos \
    --gradient_checkpointing \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --max_length "$MAX_LENGTH" \
    --eval_max_length "$EVAL_MAX_LENGTH" \
    --no-packing \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --seed "$SEED" \
    --trust_remote_code \
    --system_prompt "$SYSTEM_PROMPT" \
    --force_replace_system \
    --check_chatml_boundary 2>&1 | tee "$output_dir/train.log"
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
    echo "[ERROR] Training failed for $name with status $train_status"
    exit "$train_status"
  fi
}

eval_one() {
  local name="$1"
  local output_dir="$ABLATION_ROOT/$name"
  local eval_dir="$output_dir/eval_test"

  echo "========== Eval Ablation: $name =========="
  if [[ ! -f "$output_dir/adapter_config.json" ]]; then
    echo "[WARN] Adapter missing, skip eval: $output_dir"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}" python "$EVAL_SCRIPT" \
    --adapter_path "$output_dir" \
    --eval_path "$TEST_PATH" \
    --output_dir "$eval_dir" \
    --max_samples "$EVAL_MAX_SAMPLES" \
    --system_prompt "$SYSTEM_PROMPT" \
    --max_new_tokens "$GEN_MAX_NEW_TOKENS" \
    --repetition_penalty "$GEN_REPETITION_PENALTY" \
    --no_repeat_ngram_size "$GEN_NO_REPEAT_NGRAM_SIZE" \
    --max_output_chars "$GEN_MAX_OUTPUT_CHARS"
}

collect_results() {
  echo "========== Collect Ablation Results =========="
  python "$COLLECT_SCRIPT" \
    --ablation_root "$ABLATION_ROOT" \
    --output_dir "$REPORT_DIR"
}

should_run_experiment() {
  local name="$1"
  if [[ -z "$selected_experiments" ]]; then
    return 0
  fi
  for selected in $selected_experiments; do
    if [[ "$selected" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

test -f "$TRAIN_SCRIPT" || { echo "train script not found: $TRAIN_SCRIPT"; exit 1; }
test -f "$EVAL_SCRIPT" || { echo "eval script not found: $EVAL_SCRIPT"; exit 1; }
test -f "$COLLECT_SCRIPT" || { echo "collect script not found: $COLLECT_SCRIPT"; exit 1; }

echo "========== LoRA Ablation Plan =========="
echo "ROOT_DIR:       $ROOT_DIR"
echo "MODEL_NAME:     $MODEL_NAME"
echo "TRAIN_PATH:     $TRAIN_PATH"
echo "VAL_PATH:       $VAL_PATH"
echo "TEST_PATH:      $TEST_PATH"
echo "ABLATION_ROOT:  $ABLATION_ROOT"
echo "RUN_MODE:       $RUN_MODE"
echo "EXPERIMENTS:    ${selected_experiments:-<all>}"
echo "========================================"

ordinal=0
for row in "${DEFAULT_EXPERIMENT_MATRIX[@]}"; do
  IFS='|' read -r name train_mode rank alpha target_modules description <<< "$row"
  ordinal=$((ordinal + 1))

  if ! should_run_experiment "$name"; then
    continue
  fi

  case "$RUN_MODE" in
    all)
      train_one "$name" "$train_mode" "$rank" "$alpha" "$target_modules" "$description" "$ordinal"
      eval_one "$name"
      ;;
    train)
      train_one "$name" "$train_mode" "$rank" "$alpha" "$target_modules" "$description" "$ordinal"
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

echo "========== LoRA Ablation Finished =========="
echo "Report CSV: $REPORT_DIR/lora_ablation_summary.csv"
echo "Report MD:  $REPORT_DIR/lora_ablation_summary.md"
