#!/usr/bin/env bash
set -euo pipefail

# =========================
# 基础路径与环境变量
# =========================

# 项目根目录
ROOT_DIR="/opt/data/llz/ecommerce-customer-service-posttrain"
cd "$ROOT_DIR"

# NCCL 通信限制：有些机器上禁用这些能减少多卡通信问题
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# 缓解显存碎片问题
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hugging Face 缓存目录：放到 /opt/data/llz 下，避免系统重建后缓存丢失
export HF_HOME="/opt/data/llz/.cache/huggingface"
export HF_DATASETS_CACHE="/opt/data/llz/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/opt/data/llz/.cache/huggingface/transformers"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# 底座模型路径
MODEL_NAME="/opt/data/llz/hf_models/Qwen3-8B-Base"

# 多卡训练通信端口，避免冲突
MASTER_PORT="${MASTER_PORT:-29501}"

# 输出目录：保存 LoRA adapter、tokenizer、metrics 等
# 本版是 short-answer SFT，不覆盖旧 qwen3_8b_rank16_zero2，避免新旧 checkpoint 混在一起。
OUTPUT_DIR="$ROOT_DIR/outputs/sft/qwen3_8b_rank16_qv_eos_short"

# DeepSpeed ZeRO-2 配置文件
DEEPSPEED_CONFIG="$ROOT_DIR/configs/deepspeed_zero2.json"

# 训练脚本路径
TRAIN_SCRIPT="$ROOT_DIR/scripts/sft/train_sft_trl.py"

# 推理脚本路径
INFER_SCRIPT="$ROOT_DIR/scripts/sft/infer_sft.py"

# 本版 system prompt 目标：
# 1) 解决回复严重过长、废话多、寒暄重复的问题；
# 2) 要求只回答最后一句用户问题；
# 3) 不确定时保守核实，不主动扩展售后/退款/发货等无关流程。
SYSTEM_PROMPT="你是电商客服。只回答最后一句用户问题。答案必须简短、直接、保守，最多2句，不主动扩展，不重复寒暄。不确定时说需要帮您核实。"

mkdir -p "$OUTPUT_DIR"

echo "========== SFT Formal Train (2x4090 + DeepSpeed ZeRO-2) =========="
echo "ROOT_DIR:     $ROOT_DIR"
echo "MODEL:        $MODEL_NAME"
echo "TRAIN_PATH:   $ROOT_DIR/data/processed_5000/taobao_messages_train.json"
echo "VAL_PATH:     $ROOT_DIR/data/processed_5000/taobao_messages_dev.json"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "DS_CONFIG:    $DEEPSPEED_CONFIG"
echo "TRAIN_SCRIPT: $TRAIN_SCRIPT"
echo "INFER_SCRIPT: $INFER_SCRIPT"
echo "HF_HOME:      $HF_HOME"
echo "MASTER_PORT:  $MASTER_PORT"
echo "=================================================================="

# 先检查脚本文件是否存在
test -f "$TRAIN_SCRIPT" || { echo "训练脚本不存在: $TRAIN_SCRIPT"; exit 1; }
test -f "$INFER_SCRIPT" || { echo "推理脚本不存在: $INFER_SCRIPT"; exit 1; }

# 训练前核对输出目录，避免旧 checkpoint 和本轮结果混在一起时难以分辨
echo "========== Preflight Check =========="
if find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q .; then
  echo "[提醒] 当前 OUTPUT_DIR 中已存在旧 checkpoint："
  find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' -printf '  %f\n' | sort -V
  echo "[提醒] 如果这次是重新完整训练而不是断点续训，建议先手动备份或清理旧产物，再开始。"
else
  echo "[检查] 当前 OUTPUT_DIR 下没有旧 checkpoint。"
fi
echo "===================================="

# =========================
# 训练参数说明
# =========================
# --nproc_per_node=2              单机 2 卡训练
# --master_port                   多卡通信端口
# --model_name_or_path            底座模型路径
# --train_path                    训练集路径
# --val_path                      验证集路径
# --output_dir                    输出目录
# --deepspeed                     DeepSpeed 配置
# --train_mode qlora              使用 QLoRA，更省显存
# --lora_r 16                     LoRA rank
# --lora_alpha 32                 LoRA 缩放系数
# --lora_dropout 0.05             LoRA dropout
# --target_modules q_proj,v_proj  本版只训 q/v，降低小数据上学偏、复读和无关扩展的风险
# --completion_end_token eos      本版关键改动：用 <|endoftext|> 监督答案结束，让模型更容易在短答后停止
# --gradient_checkpointing        开启，节省显存
# --num_train_epochs 2            数据已清洗为短答，训练 2 epoch 强化短答风格
# --learning_rate 1e-5            原来 3e-5；本版降低学习率，减少模板化复读和灾难性漂移
# --warmup_ratio 0.08             原来 0.05；本版更平滑
# --max_length 512                原来 640；本版缩短上下文，减少长历史串场和过度展开
# --eval_max_length 768           原来 1024；和短答训练目标对齐
# --no-packing                    原来就是关闭 packing；继续关闭，避免多轮边界被拼接干扰
# --per_device_train_batch_size 1 单卡训练 batch size
# --per_device_eval_batch_size 1  单卡验证 batch size
# --gradient_accumulation_steps 8 梯度累积，等效增大 batch
# --dataset_num_proc 8            datasets 预处理并行进程数
# --dataloader_num_workers 4      dataloader worker 数
# --logging_steps 10              原来 20；现在更快观察 loss 走势
# --save_steps 150                原来 200；现在小数据集保留更细颗粒度 checkpoint
# --eval_steps 150                原来 200；当前虽默认不用 built-in eval，但若开启时也同步更细
# --save_total_limit 3            原来 2；现在多保留一个中间点，便于回看
# --trust_remote_code             允许加载模型自定义代码

# =========================
# 训练阶段
# =========================
# 本轮关键参数调整汇总：
# 1) 目标问题：解决 SFT 输出过长、反复“亲亲”、编造无关流程、回答停不下来。
# 2) 数据：旧 data/processed_5000 文件已直接清洗，删除超长/硬噪声/无效样本，答案 max 64 字。
# 3) completion_end_token：im_end -> eos，让模型学习在短答后输出 EOS 停止。
# 4) target_modules：q,k,v,o,gate,up,down -> q_proj,v_proj，降低 LoRA 侵入性。
# 5) learning_rate：3e-5 -> 1e-5，降低模板化和复读。
# 6) num_train_epochs：1.5 -> 2，短答数据上多训半轮。
# 7) warmup_ratio：0.05 -> 0.08，更平滑。
# 8) max_length：640 -> 512，减少长历史串场。
# 9) eval_max_length：1024 -> 768，和短答训练目标对齐。
# 10) output_dir：改为 qwen3_8b_rank16_qv_eos_short，避免覆盖旧模型。
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --nproc_per_node=2 \
  --master_port="$MASTER_PORT" \
  "$TRAIN_SCRIPT" \
  --model_name_or_path "$MODEL_NAME" \
  --train_path "$ROOT_DIR/data/processed_5000/taobao_messages_train.json" \
  --val_path "$ROOT_DIR/data/processed_5000/taobao_messages_dev.json" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --train_mode qlora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --target_modules q_proj,v_proj \
  --completion_end_token eos \
  --gradient_checkpointing \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.08 \
  --max_length 512 \
  --eval_max_length 768 \
  --no-packing \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --logging_steps 10 \
  --save_steps 150 \
  --eval_steps 150 \
  --save_total_limit 3 \
  --trust_remote_code \
  --system_prompt "$SYSTEM_PROMPT" \
  --force_replace_system \
  --check_chatml_boundary
  
echo "========== Train Finished =========="
echo "Adapter saved to: $OUTPUT_DIR"
echo "===================================="

# 训练后自动核对，避免 run_summary 只停留在训练开始前，或根目录 adapter 和最新 checkpoint 对不上
echo "========== Post-Train Verify =========="
OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
summary_path = output_dir / "run_summary.json"
metrics_path = output_dir / "metrics.json"

if not summary_path.exists():
    raise SystemExit(f"缺少 run_summary.json: {summary_path}")
if not metrics_path.exists():
    raise SystemExit(f"缺少 metrics.json: {metrics_path}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

report = {
    "summary_stage": summary.get("summary_stage"),
    "summary_written_at": summary.get("summary_written_at"),
    "num_train_epochs": summary.get("num_train_epochs"),
    "learning_rate": summary.get("learning_rate"),
    "warmup_ratio": summary.get("warmup_ratio"),
    "max_length": summary.get("max_length"),
    "eval_max_length": summary.get("eval_max_length"),
    "save_steps": summary.get("save_steps"),
    "checkpoint_dirs": summary.get("checkpoint_dirs"),
    "latest_checkpoint_dir": summary.get("latest_checkpoint_dir"),
    "root_adapter_matches_latest_checkpoint": summary.get("root_adapter_matches_latest_checkpoint"),
    "final_epoch": metrics.get("epoch"),
    "train_loss": metrics.get("train_loss"),
    "eval_loss": metrics.get("eval_loss"),
}
print(json.dumps(report, ensure_ascii=False, indent=2))

if summary.get("summary_stage") != "post_train":
    raise SystemExit("run_summary.json 还停留在 pre_train，说明训练结束后的最终摘要没有回写成功。")

match_flag = summary.get("root_adapter_matches_latest_checkpoint")
if match_flag is False:
    raise SystemExit("根目录 adapter 与最新 checkpoint 不一致，请先排查保存流程。")
PY
echo "======================================"

# =========================
# 单条推理测试参数说明
# =========================
# --base_model                    推理时先加载底座模型
# --adapter_path                  再加载训练好的 LoRA adapter
# --prompt                        单条测试问题
# --do_sample                     开启采样，输出更灵活但不完全稳定
# --max_new_tokens 64             短答模型的生成安全上限；正常应靠 EOS 自行停止
# --temperature 0.6               采样温度，偏保守
# --top_p 0.85                    nucleus sampling
# --repetition_penalty 1.25       本版提高重复惩罚，压制“亲亲/流程话术”复读
# --no_repeat_ngram_size 3        禁止重复 3-gram
# --max_output_chars 0            不做字符截断，观察模型是否能依靠 EOS 自停

# =========================
# 单条推理测试
# =========================
CUDA_VISIBLE_DEVICES=0 python "$INFER_SCRIPT" \
  --base_model "$MODEL_NAME" \
  --adapter_path "$OUTPUT_DIR" \
  --prompt "商品已经拆封了还能退吗" \
  --max_new_tokens 64 \
  --repetition_penalty 1.25 \
  --no_repeat_ngram_size 3 \
  --max_output_chars 0
