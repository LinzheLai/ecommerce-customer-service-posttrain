#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/data/llz/ecommerce-customer-service-posttrain}"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-/opt/data/llz/hf_models/Qwen3-8B-Base}"
TEST_PATH="${TEST_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_test.json}"
INFER_SCRIPT="${INFER_SCRIPT:-$ROOT_DIR/scripts/sft/infer_sft.py}"

SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT_DIR/outputs/sft/qwen3_8b_rank16_qv_eos_short}"
DPO_ADAPTER_DIR="${DPO_ADAPTER_DIR:-$ROOT_DIR/outputs/dpo/qwen3_8b_rank16_qv_eos_short_dpo_beta005}"

NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
MAX_OUTPUT_CHARS="${MAX_OUTPUT_CHARS:-56}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.12}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"

SAMPLE_INDICES_PATH="${SAMPLE_INDICES_PATH:-$ROOT_DIR/outputs/compare/sample_indices/taobao_messages_test_random${NUM_SAMPLES}_seed${SEED}.json}"
INDEX_SOURCE_PATH="${INDEX_SOURCE_PATH:-}"
BASE_OUTPUT="${BASE_OUTPUT:-$ROOT_DIR/outputs/base/batch_infer/taobao_messages_test_random${NUM_SAMPLES}_base.json}"
SFT_OUTPUT="${SFT_OUTPUT:-$ROOT_DIR/outputs/sft/batch_infer/taobao_messages_test_random${NUM_SAMPLES}_sft.json}"
DPO_OUTPUT="${DPO_OUTPUT:-$ROOT_DIR/outputs/dpo/batch_infer/taobao_messages_test_random${NUM_SAMPLES}_dpo.json}"

mkdir -p \
  "$(dirname "$SAMPLE_INDICES_PATH")" \
  "$(dirname "$BASE_OUTPUT")" \
  "$(dirname "$SFT_OUTPUT")" \
  "$(dirname "$DPO_OUTPUT")"

test -f "$INFER_SCRIPT" || { echo "infer script not found: $INFER_SCRIPT"; exit 1; }

has_adapter() {
  local adapter_dir="$1"
  [[ -f "$adapter_dir/adapter_config.json" ]]
}

run_infer() {
  local model_label="$1"
  local adapter_path="$2"
  local output_path="$3"
  local index_arg_mode="$4"

  local index_args=()
  if [[ -n "$INDEX_SOURCE_PATH" ]]; then
    index_args=(--sample_indices_path "$INDEX_SOURCE_PATH")
  elif [[ "$index_arg_mode" == "save" ]]; then
    index_args=(--save_sample_indices_path "$SAMPLE_INDICES_PATH")
  else
    index_args=(--sample_indices_path "$SAMPLE_INDICES_PATH")
  fi

  python "$INFER_SCRIPT" \
    --base_model "$MODEL_NAME" \
    --adapter_path "$adapter_path" \
    --test_path "$TEST_PATH" \
    --output_path "$output_path" \
    --num_samples "$NUM_SAMPLES" \
    --seed "$SEED" \
    --model_label "$model_label" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
    --max_output_chars "$MAX_OUTPUT_CHARS" \
    "${index_args[@]}"
}

echo "========== Compare Batch Infer =========="
echo "ROOT_DIR:            $ROOT_DIR"
echo "MODEL_NAME:          $MODEL_NAME"
echo "TEST_PATH:           $TEST_PATH"
echo "SAMPLE_INDICES_PATH: $SAMPLE_INDICES_PATH"
echo "INDEX_SOURCE_PATH:   ${INDEX_SOURCE_PATH:-<none>}"
echo "BASE_OUTPUT:         $BASE_OUTPUT"
echo "SFT_OUTPUT:          $SFT_OUTPUT"
echo "DPO_OUTPUT:          $DPO_OUTPUT"
echo "========================================="

if [[ -f "$SAMPLE_INDICES_PATH" ]]; then
  run_infer base none "$BASE_OUTPUT" reuse
else
  run_infer base none "$BASE_OUTPUT" save
fi

if has_adapter "$SFT_ADAPTER_DIR"; then
  run_infer sft "$SFT_ADAPTER_DIR" "$SFT_OUTPUT" reuse
else
  echo "[ERROR] SFT adapter not found: $SFT_ADAPTER_DIR"
  exit 1
fi

if has_adapter "$DPO_ADAPTER_DIR"; then
  run_infer dpo "$DPO_ADAPTER_DIR" "$DPO_OUTPUT" reuse
else
  echo "[WARN] DPO adapter not found, skipped: $DPO_ADAPTER_DIR"
fi
