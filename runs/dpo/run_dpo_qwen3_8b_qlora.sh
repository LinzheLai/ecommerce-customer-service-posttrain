#!/usr/bin/env bash
set -euo pipefail

# =========================
# Paths and env
# =========================
ROOT_DIR="/opt/data/llz/ecommerce-customer-service-posttrain"
cd "$ROOT_DIR"

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HF_HOME="/opt/data/llz/.cache/huggingface"
export HF_DATASETS_CACHE="/opt/data/llz/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/opt/data/llz/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

MODEL_NAME="/opt/data/llz/hf_models/Qwen3-8B-Base"
SFT_ADAPTER_DIR="$ROOT_DIR/outputs/sft/qwen3_8b_rank16_qv_eos_short"
OUTPUT_DIR="$ROOT_DIR/outputs/dpo/qwen3_8b_rank16_qv_eos_short_dpo_beta005"
DEEPSPEED_CONFIG="$ROOT_DIR/configs/deepspeed_zero2.json"
TRAIN_SCRIPT="$ROOT_DIR/scripts/dpo/train_dpo_trl.py"
INFER_SCRIPT="$ROOT_DIR/scripts/sft/infer_sft.py"

SYSTEM_PROMPT="你是电商客服。只回答最后一句用户问题。答案必须简短、直接、保守，最多2句，不主动扩展，不重复寒暄。不确定时说需要帮您核实。"

MASTER_PORT="${MASTER_PORT:-29511}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-0,1}"
MAX_ALLOWED_GPU_USED_MB="${MAX_ALLOWED_GPU_USED_MB:-2048}"
export CUDA_VISIBLE_DEVICES_VALUE
export MAX_ALLOWED_GPU_USED_MB

mkdir -p "$OUTPUT_DIR"

echo "========== DPO Formal Train (2x4090 + DeepSpeed ZeRO-2) =========="
echo "ROOT_DIR:        $ROOT_DIR"
echo "MODEL:           $MODEL_NAME"
echo "SFT_ADAPTER_DIR: $SFT_ADAPTER_DIR"
echo "TRAIN_PATH:      $ROOT_DIR/data/processed_5000/taobao_dpo_train.json"
echo "VAL_PATH:        $ROOT_DIR/data/processed_5000/taobao_dpo_dev.json"
echo "OUTPUT_DIR:      $OUTPUT_DIR"
echo "DS_CONFIG:       $DEEPSPEED_CONFIG"
echo "TRAIN_SCRIPT:    $TRAIN_SCRIPT"
echo "INFER_SCRIPT:    $INFER_SCRIPT"
echo "HF_HOME:         $HF_HOME"
echo "MASTER_PORT:     $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES_VALUE"
echo "=================================================================="

test -f "$TRAIN_SCRIPT" || { echo "训练脚本不存在: $TRAIN_SCRIPT"; exit 1; }
test -f "$INFER_SCRIPT" || { echo "推理脚本不存在: $INFER_SCRIPT"; exit 1; }
test -f "$SFT_ADAPTER_DIR/adapter_model.safetensors" || { echo "SFT adapter 不存在: $SFT_ADAPTER_DIR"; exit 1; }
test -f "$ROOT_DIR/data/processed_5000/taobao_dpo_train.json" || { echo "DPO train json 不存在"; exit 1; }
test -f "$ROOT_DIR/data/processed_5000/taobao_dpo_dev.json" || { echo "DPO val json 不存在"; exit 1; }

echo "========== Preflight Check =========="
if find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
  echo "[提醒] 当前 OUTPUT_DIR 中已经存在旧 checkpoint:"
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '  %f\n' | sort -V
  echo "[提醒] 如果这次是重新完整训练而不是断点续训，建议先手动备份或清理旧产物。"
else
  echo "[检查] 当前 OUTPUT_DIR 下没有旧 checkpoint。"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[检查] 当前 GPU 显存占用:"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
fi
echo "===================================="

# =========================
# Training notes
# =========================
# =========================
# DPO 参数说明：短答 SFT 后的业务纠偏版
# =========================
# 目标问题：
#   当前 SFT 已经明显变短，但存在“短而错”的问题：
#   - 可以发邮政 -> 回成不支持
#   - 不能提前优惠 -> 回成可以
#   - 今天不能发货 -> 回成顺丰包邮/可发货
#   - 赠品/夹子/改地址等规则类问题不稳
#
# 本版策略：
#   1) 从新的 EOS 短答 SFT adapter 继续训练；
#   2) DPO chosen/rejected 也使用 <|endoftext|> 结尾，保持短答自停能力；
#   3) 用更低 learning_rate 和更少 epoch，避免把 SFT 的短答能力训坏；
#   4) max_completion_length 仍限制为 64，让偏好优化聚焦短答案质量；
#   5) label_smoothing=0.03，降低偏好数据少量噪声的影响。

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" torchrun \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  --model_name_or_path "$MODEL_NAME" \
  --sft_adapter_path "$SFT_ADAPTER_DIR" \
  --train_path "$ROOT_DIR/data/processed_5000/taobao_dpo_train.json" \
  --val_path "$ROOT_DIR/data/processed_5000/taobao_dpo_dev.json" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --train_mode qlora \
  --beta 0.05 \
  --loss_type sigmoid \
  --label_smoothing 0.03 \
  --completion_end_token eos \
  --truncation_mode keep_end \
  --max_length 640 \
  --max_prompt_length 512 \
  --max_completion_length 64 \
  --gradient_checkpointing \
  --num_train_epochs 1.0 \
  --learning_rate 2e-6 \
  --warmup_ratio 0.08 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --logging_steps 10 \
  --save_steps 100 \
  --eval_steps 100 \
  --save_total_limit 3 \
  --trust_remote_code \
  --system_prompt "$SYSTEM_PROMPT" \
  --force_replace_system

echo "========== Train Finished =========="
echo "Adapter saved to: $OUTPUT_DIR"
echo "===================================="

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE%%,*}" python "$INFER_SCRIPT" \
  --base_model "$MODEL_NAME" \
  --adapter_path "$OUTPUT_DIR" \
  --prompt "商品已经拆封了还能退吗" \
  --max_new_tokens 64 \
  --repetition_penalty 1.25 \
  --no_repeat_ngram_size 3 \
  --max_output_chars 0
