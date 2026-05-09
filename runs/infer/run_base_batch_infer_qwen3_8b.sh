#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/opt/data/llz/ecommerce-customer-service-posttrain}"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-/opt/data/llz/hf_models/Qwen3-8B-Base}"
TEST_PATH="${TEST_PATH:-$ROOT_DIR/data/processed_5000/taobao_messages_test.json}"
INFER_SCRIPT="${INFER_SCRIPT:-$ROOT_DIR/scripts/sft/infer_sft.py}"

NUM_SAMPLES="${NUM_SAMPLES:-20}"
SEED="${SEED:-42}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
MAX_OUTPUT_CHARS="${MAX_OUTPUT_CHARS:-56}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.12}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-4}"

OUTPUT_PATH="${OUTPUT_PATH:-$ROOT_DIR/outputs/base/batch_infer/taobao_messages_test_random${NUM_SAMPLES}_base.json}"
SAMPLE_INDICES_PATH="${SAMPLE_INDICES_PATH:-$ROOT_DIR/outputs/compare/sample_indices/taobao_messages_test_random${NUM_SAMPLES}_seed${SEED}.json}"
INDEX_SOURCE_PATH="${INDEX_SOURCE_PATH:-}"

mkdir -p "$(dirname "$OUTPUT_PATH")" "$(dirname "$SAMPLE_INDICES_PATH")"

test -f "$INFER_SCRIPT" || { echo "infer script not found: $INFER_SCRIPT"; exit 1; }

echo "========== Base Batch Infer =========="
echo "ROOT_DIR:            $ROOT_DIR"
echo "MODEL_NAME:          $MODEL_NAME"
echo "TEST_PATH:           $TEST_PATH"
echo "OUTPUT_PATH:         $OUTPUT_PATH"
echo "SAMPLE_INDICES_PATH: $SAMPLE_INDICES_PATH"
echo "INDEX_SOURCE_PATH:   ${INDEX_SOURCE_PATH:-<none>}"
echo "NUM_SAMPLES:         $NUM_SAMPLES"
echo "SEED:                $SEED"
echo "======================================"

index_args=()
if [[ -n "$INDEX_SOURCE_PATH" ]]; then
  index_args=(--sample_indices_path "$INDEX_SOURCE_PATH")
elif [[ -f "$SAMPLE_INDICES_PATH" ]]; then
  index_args=(--sample_indices_path "$SAMPLE_INDICES_PATH")
else
  index_args=(--save_sample_indices_path "$SAMPLE_INDICES_PATH")
fi

python "$INFER_SCRIPT" \
  --base_model "$MODEL_NAME" \
  --adapter_path none \
  --test_path "$TEST_PATH" \
  --output_path "$OUTPUT_PATH" \
  --num_samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --model_label base \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --repetition_penalty "$REPETITION_PENALTY" \
  --no_repeat_ngram_size "$NO_REPEAT_NGRAM_SIZE" \
  --max_output_chars "$MAX_OUTPUT_CHARS" \
  "${index_args[@]}"
