import argparse
import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def require_package(name: str, install_hint: str = ""):
    try:
        return __import__(name)
    except ImportError as e:
        hint = f"\n请先安装依赖: {install_hint}" if install_hint else ""
        raise ImportError(f"缺少依赖包: {name}.{hint}") from e


datasets = require_package("datasets", "pip install datasets")
transformers = require_package("transformers", "pip install transformers")
trl = require_package("trl", "pip install trl")
peft = require_package("peft", "pip install peft")
accelerate = require_package("accelerate", "pip install accelerate")

from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


SIMPLE_CHATML_TEMPLATE = r"""
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
{{- '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'user' %}
{{- '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}
    {%- elif message['role'] == 'assistant' %}
{{- '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}
""".strip()

DEFAULT_SYSTEM_PROMPT = (
    "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。"
    "如果历史对话里没有明确信息，只能给出简短、保守回复。"
)


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    if is_dist_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(
        description="电商客服 SFT 训练脚本（字符串版 prompt/completion + 稳定手动验证）"
    )

    # 数据
    parser.add_argument("--train_path", type=str, required=True, help="训练集路径，支持 json/jsonl")
    parser.add_argument("--val_path", type=str, default=None, help="验证集路径")
    parser.add_argument("--validation_split_ratio", type=float, default=0.0, help="若未提供 val_path，则从 train 内部分割验证集")
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--group_by_length", action="store_true")
    parser.add_argument("--drop_empty_samples", action="store_true")
    parser.add_argument(
        "--train_on_last_turn_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="仅监督最后一轮 assistant 回复；建议保持开启",
    )

    # 模型
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/opt/data/llz/hf_models/Qwen3-8B-Base",
        help="本地模型路径或 HuggingFace 模型名",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    parser.add_argument("--train_mode", type=str, default="qlora", choices=["qlora", "lora"])
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed 配置文件路径")
    parser.add_argument(
        "--use_simple_chatml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="覆盖默认 chat template，使用简单 ChatML 模板（主要给推理/评估对齐）",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="统一覆盖训练时使用的 system prompt，确保与推理一致",
    )
    parser.add_argument(
        "--force_replace_system",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否强制覆盖数据中的 system prompt",
    )
    parser.add_argument(
        "--check_chatml_boundary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练前检查 ChatML special token 与边界是否正常",
    )
    parser.add_argument(
        "--completion_end_token",
        type=str,
        default="im_end",
        choices=["im_end", "eos", "both"],
        help=(
            "completion 结尾监督符。im_end 使用 <|im_end|>；"
            "eos 使用 <|endoftext|>；both 同时使用两者。"
            "短答客服模型建议 eos，让 generate 更容易正常停止。"
        ),
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help='逗号分隔，或直接写 "all-linear"',
    )

    # 训练
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=3000)
    parser.add_argument("--eval_steps", type=int, default=3000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none", help="none / tensorboard / wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)

    # 验证相关
    parser.add_argument(
        "--use_builtin_eval",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否启用 Trainer 内置 eval。默认关闭，改用更稳的手动验证。",
    )
    parser.add_argument(
        "--eval_max_length",
        type=int,
        default=768,
        help="手动验证时使用的最大长度；可设得比训练 max_length 更大。",
    )
    parser.add_argument(
        "--manual_eval_on_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="训练结束后是否执行手动验证并计算稳定的 eval_loss。",
    )

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_get_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        return torch.float16, False, True
    return torch.float32, False, False


def parse_target_modules(raw: str):
    raw = raw.strip()
    if raw == "all-linear":
        return "all-linear"
    return [x.strip() for x in raw.split(",") if x.strip()]


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0000", "")
    return text.strip()


def normalize_messages(messages: List[Dict]) -> List[Dict]:
    new_messages = []
    for m in messages:
        role = str(m.get("role", "")).strip().lower()
        content = normalize_text(m.get("content", ""))
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        new_messages.append({"role": role, "content": content})
    return new_messages


def apply_consistent_system_prompt(
    messages: List[Dict],
    system_prompt: str,
    force_replace_system: bool = True,
) -> List[Dict]:
    """
    统一 system prompt，保证训练与推理一致。
    - 如果第一条是 system：
        - force_replace_system=True -> 直接替换
        - False -> 保留原始 system
    - 如果没有 system，则在最前面补一个
    """
    messages = normalize_messages(messages)
    if not messages:
        return []

    system_prompt = normalize_text(system_prompt)
    if not system_prompt:
        return messages

    if messages[0]["role"] == "system":
        if force_replace_system:
            messages[0] = {"role": "system", "content": system_prompt}
        return messages

    return [{"role": "system", "content": system_prompt}] + messages


def render_chatml(messages: List[Dict], add_generation_prompt: bool = False) -> str:
    parts = []
    for m in messages:
        role = str(m["role"]).strip().lower()
        content = normalize_text(m["content"])
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")

    return "".join(parts)


def ensure_completion_suffix(text: str, completion_end_token: str = "im_end") -> str:
    text = normalize_text(text)
    if not text:
        return ""

    if completion_end_token == "eos":
        suffix = "<|endoftext|>"
    elif completion_end_token == "both":
        suffix = "<|im_end|>\n<|endoftext|>"
    else:
        suffix = "<|im_end|>"

    stripped = text.rstrip()
    for known_suffix in ("<|im_end|>", "<|endoftext|>"):
        if stripped.endswith(known_suffix):
            return stripped + "\n"
    return stripped + suffix + "\n"


def messages_to_prompt_completion(
    messages: List[Dict],
    system_prompt: str,
    force_replace_system: bool = True,
    completion_end_token: str = "im_end",
):
    messages = normalize_messages(messages)
    if not messages:
        return None

    if messages[-1]["role"] != "assistant":
        return None

    messages = apply_consistent_system_prompt(
        messages,
        system_prompt=system_prompt,
        force_replace_system=force_replace_system,
    )

    prompt_messages = messages[:-1]
    assistant_text = normalize_text(messages[-1]["content"])

    if not prompt_messages or not assistant_text:
        return None

    prompt_text = render_chatml(prompt_messages, add_generation_prompt=True)
    completion_text = ensure_completion_suffix(assistant_text, completion_end_token)

    return {
        "prompt": prompt_text,
        "completion": completion_text,
    }


def convert_row(
    row: Dict,
    system_prompt: str,
    force_replace_system: bool,
    completion_end_token: str = "im_end",
):
    # 1) messages -> 字符串版 prompt/completion
    if "messages" in row:
        return messages_to_prompt_completion(
            row["messages"],
            system_prompt=system_prompt,
            force_replace_system=force_replace_system,
            completion_end_token=completion_end_token,
        )

    # 2) 已有 prompt/completion
    if "prompt" in row and "completion" in row:
        # 已经是字符串版，直接用
        if isinstance(row["prompt"], str) and isinstance(row["completion"], str):
            prompt_text = normalize_text(row["prompt"])
            completion_text = ensure_completion_suffix(row["completion"], completion_end_token)
            if not prompt_text or not completion_text:
                return None
            return {
                "prompt": prompt_text,
                "completion": completion_text,
            }

        # 还是消息列表版，就转成字符串版
        prompt_msgs = apply_consistent_system_prompt(
            row["prompt"],
            system_prompt=system_prompt,
            force_replace_system=force_replace_system,
        )
        completion_msgs = normalize_messages(row["completion"])

        if not prompt_msgs or not completion_msgs:
            return None
        if completion_msgs[-1]["role"] != "assistant":
            return None

        prompt_text = render_chatml(prompt_msgs, add_generation_prompt=True)
        completion_text = ensure_completion_suffix(
            completion_msgs[-1]["content"],
            completion_end_token,
        )

        return {
            "prompt": prompt_text,
            "completion": completion_text,
        }

    # 3) instruction/input/output -> 字符串版 prompt/completion
    user_input = normalize_text(row.get("input", ""))
    output = normalize_text(row.get("output", ""))

    if not user_input or not output:
        return None

    prompt_messages = [
        {"role": "system", "content": normalize_text(system_prompt)},
        {"role": "user", "content": user_input},
    ]

    prompt_text = render_chatml(prompt_messages, add_generation_prompt=True)
    completion_text = ensure_completion_suffix(output, completion_end_token)

    return {
        "prompt": prompt_text,
        "completion": completion_text,
    }


def load_and_prepare_dataset(
    path: str,
    dataset_num_proc: int,
    drop_empty_samples: bool,
    system_prompt: str,
    force_replace_system: bool,
    completion_end_token: str = "im_end",
):
    ds = load_dataset("json", data_files=path)["train"]
    original_columns = ds.column_names

    def _map_fn(row):
        item = convert_row(
            row,
            system_prompt=system_prompt,
            force_replace_system=force_replace_system,
            completion_end_token=completion_end_token,
        )
        if item is None:
            return {"__keep__": False}
        item["__keep__"] = True
        return item

    ds = ds.map(_map_fn, remove_columns=original_columns, num_proc=dataset_num_proc)
    before = len(ds)
    ds = ds.filter(lambda x: bool(x.get("__keep__", False)), num_proc=dataset_num_proc)
    after = len(ds)

    if drop_empty_samples:
        if "prompt" in ds.column_names and "completion" in ds.column_names:
            ds = ds.filter(
                lambda x: bool(str(x["prompt"]).strip()) and bool(str(x["completion"]).strip()),
                num_proc=dataset_num_proc,
            )

    if "__keep__" in ds.column_names:
        ds = ds.remove_columns(["__keep__"])

    main_print(f"[数据] {path} 原始样本数: {before}，保留样本数: {after}，删除样本数: {before - after}")
    main_print(f"[数据] 字段: {ds.column_names}")

    if after > 0 and is_main_process():
        sample = ds[0]
        preview = {
            "prompt_preview": sample["prompt"][:500],
            "completion_preview": sample["completion"][:200],
        }
        main_print("[数据预览]")
        main_print(json.dumps(preview, ensure_ascii=False, indent=2))

    return ds


def check_special_tokens_and_chatml_boundary(
    tokenizer,
    system_prompt: str,
    completion_end_token: str = "im_end",
):
    main_print("\n[检查] 开始检查 ChatML special token / 边界一致性...")

    test_messages = [
        {"role": "system", "content": normalize_text(system_prompt)},
        {"role": "user", "content": "商品已经拆封了还能退吗"},
    ]

    prompt_text = render_chatml(test_messages, add_generation_prompt=True)
    completion_text = ensure_completion_suffix("需要看是否影响二次销售哦", completion_end_token)

    im_start_ids = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

    main_print("[检查] <|im_start|> token ids:", im_start_ids)
    main_print("[检查] <|im_end|> token ids:", im_end_ids)

    if len(im_start_ids) == 0:
        raise ValueError("tokenizer 无法编码 <|im_start|>")
    if len(im_end_ids) == 0:
        raise ValueError("tokenizer 无法编码 <|im_end|>")

    main_print("[检查] prompt_text 预览:")
    main_print(prompt_text[:400])

    main_print("[检查] completion_text 预览:")
    main_print(completion_text)

    main_print("[检查] prompt 最后20个 token ids:")
    main_print(prompt_ids[-20:])

    main_print("[检查] completion 最后20个 token ids:")
    main_print(completion_ids[-20:])

    decoded_completion_tail = tokenizer.decode(completion_ids[-20:], skip_special_tokens=False)
    main_print("[检查] completion 尾部 decode:")
    main_print(decoded_completion_tail)

    if completion_end_token in {"im_end", "both"} and "<|im_end|>" not in completion_text:
        raise ValueError("completion_text 没有正确包含 <|im_end|>")
    if completion_end_token in {"eos", "both"} and "<|endoftext|>" not in completion_text:
        raise ValueError("completion_text 没有正确包含 <|endoftext|>")

    main_print("[检查] ChatML 边界检查通过。\n")


def build_model_and_tokenizer(args):
    dtype, use_bf16, use_fp16 = maybe_get_dtype()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_simple_chatml:
        tokenizer.chat_template = SIMPLE_CHATML_TEMPLATE
        main_print("[模板] 已覆盖为简单 ChatML 模板。")

    if "Qwen3" in args.model_name_or_path and "Base" in args.model_name_or_path:
        main_print("[提示] 当前使用 Qwen3-8B-Base。若后续格式稳定后仍想提升对话跟随性，再考虑换更偏指令的底座。")

    if args.check_chatml_boundary and is_main_process():
        check_special_tokens_and_chatml_boundary(
            tokenizer=tokenizer,
            system_prompt=args.system_prompt,
            completion_end_token=args.completion_end_token,
        )

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }

    if args.train_mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError("QLoRA 模式需要 bitsandbytes 与 peft 的量化支持，请安装 bitsandbytes。") from e

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["quantization_config"] = quantization_config

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": local_rank}

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs,
        )
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=parse_target_modules(args.target_modules),
    )

    return model, tokenizer, peft_config, use_bf16, use_fp16


def build_sft_config(args, use_bf16: bool, use_fp16: bool, has_eval: bool):
    sig = inspect.signature(SFTConfig.__init__)
    params = sig.parameters

    group_by_length = args.group_by_length
    if args.packing and group_by_length:
        main_print("[提示] packing=True 时，group_by_length 收益很小，这里自动关闭 group_by_length。")
        group_by_length = False

    cfg = dict(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to if args.report_to != "none" else [],
        seed=args.seed,
        max_length=args.max_length,
        dataset_num_proc=args.dataset_num_proc,
        completion_only_loss=True,
        bf16=use_bf16,
        fp16=use_fp16,
        weight_decay=args.weight_decay,
        packing=args.packing,
    )

    if "deepspeed" in params and args.deepspeed:
        cfg["deepspeed"] = args.deepspeed
    if "dataloader_num_workers" in params:
        cfg["dataloader_num_workers"] = args.dataloader_num_workers
    if "group_by_length" in params:
        cfg["group_by_length"] = group_by_length
    if "ddp_find_unused_parameters" in params:
        cfg["ddp_find_unused_parameters"] = False
    if "gradient_checkpointing_kwargs" in params and args.gradient_checkpointing:
        cfg["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "logging_first_step" in params:
        cfg["logging_first_step"] = True

    if "load_best_model_at_end" in params:
        cfg["load_best_model_at_end"] = has_eval
    if "metric_for_best_model" in params and has_eval:
        cfg["metric_for_best_model"] = "eval_loss"
    if "greater_is_better" in params and has_eval:
        cfg["greater_is_better"] = False

    if has_eval:
        if "eval_steps" in params:
            cfg["eval_steps"] = args.eval_steps
        if "eval_strategy" in params:
            cfg["eval_strategy"] = "steps"
        elif "evaluation_strategy" in params:
            cfg["evaluation_strategy"] = "steps"
    else:
        if "eval_strategy" in params:
            cfg["eval_strategy"] = "no"
        elif "evaluation_strategy" in params:
            cfg["evaluation_strategy"] = "no"

    if "save_strategy" in params:
        cfg["save_strategy"] = "steps"

    return SFTConfig(**cfg)


def build_trainer(model, tokenizer, peft_config, train_dataset, eval_dataset, sft_config):
    sig = inspect.signature(SFTTrainer.__init__)
    params = sig.parameters

    kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    return SFTTrainer(**kwargs)


def tokenize_prompt_completion_for_eval(
    tokenizer,
    prompt_text: str,
    completion_text: str,
    max_length: int,
    completion_end_token: str = "im_end",
):
    prompt_text = normalize_text(prompt_text)
    completion_text = ensure_completion_suffix(completion_text, completion_end_token)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]

    if len(completion_ids) == 0:
        return None

    # 优先保留完整 completion；如果总长超限，则从左侧裁掉旧 prompt
    if len(completion_ids) >= max_length:
        prompt_ids = []
        completion_ids = completion_ids[:max_length]
    else:
        keep_prompt_len = max_length - len(completion_ids)
        if len(prompt_ids) > keep_prompt_len:
            prompt_ids = prompt_ids[-keep_prompt_len:]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    attention_mask = [1] * len(input_ids)

    valid_label_count = sum(1 for x in labels[1:] if x != -100)
    if valid_label_count <= 0:
        return None

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def collate_eval_batch(features: List[Dict], pad_token_id: int):
    max_len = max(len(x["input_ids"]) for x in features)

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for x in features:
        seq_len = len(x["input_ids"])
        pad_len = max_len - seq_len

        batch_input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        batch_attention_mask.append(x["attention_mask"] + [0] * pad_len)
        batch_labels.append(x["labels"] + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
        "labels": torch.tensor(batch_labels, dtype=torch.long),
    }


def manual_evaluate_loss(
    model,
    tokenizer,
    eval_dataset,
    eval_max_length: int,
    batch_size: int,
    completion_end_token: str = "im_end",
):
    start_time = time.time()

    prepared = []
    skipped = 0

    for sample in eval_dataset:
        item = tokenize_prompt_completion_for_eval(
            tokenizer=tokenizer,
            prompt_text=sample["prompt"],
            completion_text=sample["completion"],
            max_length=eval_max_length,
            completion_end_token=completion_end_token,
        )
        if item is None:
            skipped += 1
            continue
        prepared.append(item)

    if len(prepared) == 0:
        return {
            "eval_loss": float("nan"),
            "eval_runtime": round(time.time() - start_time, 4),
            "eval_samples_per_second": 0.0,
            "eval_steps_per_second": 0.0,
            "eval_mean_token_accuracy": 0.0,
            "eval_num_samples": 0,
            "eval_num_skipped_samples": skipped,
            "eval_max_length_used": eval_max_length,
        }

    dataloader = DataLoader(
        prepared,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_eval_batch(x, tokenizer.pad_token_id),
    )

    device = torch.device(
        f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"
        if torch.cuda.is_available()
        else "cpu"
    )

    model.eval()

    total_loss_sum = 0.0
    total_valid_tokens = 0
    total_correct = 0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs.logits[:, :-1, :]
            shift_labels = batch["labels"][:, 1:]
            valid_mask = shift_labels != -100

            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue

            preds = logits.argmax(dim=-1)
            correct = ((preds == shift_labels) & valid_mask).sum().item()

            total_loss_sum += float(outputs.loss.item()) * valid_count
            total_valid_tokens += valid_count
            total_correct += correct
            total_steps += 1

    runtime = time.time() - start_time

    if total_valid_tokens == 0:
        eval_loss = float("nan")
        eval_acc = 0.0
    else:
        eval_loss = total_loss_sum / total_valid_tokens
        eval_acc = total_correct / total_valid_tokens

    return {
        "eval_loss": round(eval_loss, 6) if eval_loss == eval_loss else float("nan"),
        "eval_runtime": round(runtime, 4),
        "eval_samples_per_second": round(len(prepared) / runtime, 4) if runtime > 0 else 0.0,
        "eval_steps_per_second": round(total_steps / runtime, 4) if runtime > 0 else 0.0,
        "eval_mean_token_accuracy": round(eval_acc, 6),
        "eval_num_samples": len(prepared),
        "eval_num_skipped_samples": skipped,
        "eval_max_length_used": eval_max_length,
    }


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_checkpoint_dirs(output_dir: str) -> List[Path]:
    base = Path(output_dir)
    if not base.exists():
        return []

    def checkpoint_step(p: Path) -> int:
        try:
            return int(p.name.split("-", 1)[1])
        except Exception:
            return -1

    return sorted(
        [p for p in base.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=checkpoint_step,
    )


def collect_artifact_summary(output_dir: str) -> Dict:
    base = Path(output_dir)
    checkpoints = list_checkpoint_dirs(output_dir)
    latest_checkpoint = checkpoints[-1] if checkpoints else None

    summary = {
        "checkpoint_dirs": [p.name for p in checkpoints],
        "latest_checkpoint_dir": latest_checkpoint.name if latest_checkpoint is not None else None,
        "root_adapter_exists": (base / "adapter_model.safetensors").exists(),
        "root_metrics_exists": (base / "metrics.json").exists(),
        "root_run_summary_exists": (base / "run_summary.json").exists(),
    }

    if latest_checkpoint is None:
        return summary

    trainer_state_path = latest_checkpoint / "trainer_state.json"
    if trainer_state_path.exists():
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
        summary["latest_checkpoint_state"] = {
            "global_step": trainer_state.get("global_step"),
            "max_steps": trainer_state.get("max_steps"),
            "epoch": trainer_state.get("epoch"),
            "num_train_epochs": trainer_state.get("num_train_epochs"),
        }

    root_adapter = base / "adapter_model.safetensors"
    latest_adapter = latest_checkpoint / "adapter_model.safetensors"
    if root_adapter.exists() and latest_adapter.exists():
        root_sha = sha256_file(root_adapter)
        latest_sha = sha256_file(latest_adapter)
        summary["root_adapter_matches_latest_checkpoint"] = root_sha == latest_sha
        summary["root_adapter_sha256"] = root_sha
        summary["latest_checkpoint_adapter_sha256"] = latest_sha

    return summary


def save_run_summary(args, train_dataset, eval_dataset, stage: str = "pre_train", metrics: Optional[Dict] = None):
    if not is_main_process():
        return

    summary = {
        "summary_stage": stage,
        "summary_written_at": now_str(),
        "train_path": args.train_path,
        "val_path": args.val_path,
        "validation_split_ratio": args.validation_split_ratio,
        "dataset_num_proc": args.dataset_num_proc,
        "dataloader_num_workers": args.dataloader_num_workers,
        "group_by_length": args.group_by_length,
        "drop_empty_samples": args.drop_empty_samples,
        "model_name_or_path": args.model_name_or_path,
        "trust_remote_code": args.trust_remote_code,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "train_mode": args.train_mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": parse_target_modules(args.target_modules),
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
        "output_dir": args.output_dir,
        "deepspeed": args.deepspeed,
        "train_on_last_turn_only": args.train_on_last_turn_only,
        "use_simple_chatml": args.use_simple_chatml,
        "packing": args.packing,
        "use_builtin_eval": args.use_builtin_eval,
        "eval_max_length": args.eval_max_length,
        "manual_eval_on_end": args.manual_eval_on_end,
        "system_prompt": args.system_prompt,
        "force_replace_system": args.force_replace_system,
        "check_chatml_boundary": args.check_chatml_boundary,
        "completion_end_token": args.completion_end_token,
    }

    if metrics is not None:
        summary["final_metrics"] = metrics
        summary.update(collect_artifact_summary(args.output_dir))

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    main_print(json.dumps(summary, ensure_ascii=False, indent=2))


def save_dataset_preview(output_dir: str, train_dataset, eval_dataset):
    if not is_main_process():
        return

    preview = {
        "train_preview": train_dataset[:2],
        "eval_preview": eval_dataset[:2] if eval_dataset is not None else None,
    }
    with open(Path(output_dir) / "dataset_preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    if not args.train_on_last_turn_only:
        raise ValueError("这版脚本固定为只训练最后一轮 assistant 回复，请保持 --train_on_last_turn_only 开启。")

    train_dataset = load_and_prepare_dataset(
        args.train_path,
        args.dataset_num_proc,
        drop_empty_samples=args.drop_empty_samples,
        system_prompt=args.system_prompt,
        force_replace_system=args.force_replace_system,
        completion_end_token=args.completion_end_token,
    )

    loaded_eval_dataset = None
    if args.val_path:
        loaded_eval_dataset = load_and_prepare_dataset(
            args.val_path,
            args.dataset_num_proc,
            drop_empty_samples=args.drop_empty_samples,
            system_prompt=args.system_prompt,
            force_replace_system=args.force_replace_system,
            completion_end_token=args.completion_end_token,
        )
    elif args.validation_split_ratio and args.validation_split_ratio > 0:
        split = train_dataset.train_test_split(test_size=args.validation_split_ratio, seed=args.seed)
        train_dataset = split["train"]
        loaded_eval_dataset = split["test"]

    # 是否将验证集交给 Trainer 内置 eval
    trainer_eval_dataset = loaded_eval_dataset if args.use_builtin_eval else None

    model, tokenizer, peft_config, use_bf16, use_fp16 = build_model_and_tokenizer(args)
    sft_config = build_sft_config(args, use_bf16, use_fp16, trainer_eval_dataset is not None)

    save_run_summary(args, train_dataset, loaded_eval_dataset, stage="pre_train")
    save_dataset_preview(args.output_dir, train_dataset, loaded_eval_dataset)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=train_dataset,
        eval_dataset=trainer_eval_dataset,
        sft_config=sft_config,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)

    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)

    metrics = dict(train_result.metrics)

    # 可选：保留 built-in eval
    if trainer_eval_dataset is not None and args.use_builtin_eval:
        builtin_eval_metrics = trainer.evaluate()
        metrics.update({f"builtin_{k}": v for k, v in builtin_eval_metrics.items()})

    # 更稳的手动验证：默认开启
    if loaded_eval_dataset is not None and args.manual_eval_on_end and is_main_process():
        main_print("[手动验证] 开始计算稳定版 eval_loss（不 packing，优先保留 completion）...")
        manual_eval_metrics = manual_evaluate_loss(
            model=trainer.model,
            tokenizer=tokenizer,
            eval_dataset=loaded_eval_dataset,
            eval_max_length=args.eval_max_length,
            batch_size=args.per_device_eval_batch_size,
            completion_end_token=args.completion_end_token,
        )
        metrics.update(manual_eval_metrics)

    if is_main_process():
        metrics_path = Path(args.output_dir) / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        save_run_summary(
            args,
            train_dataset,
            loaded_eval_dataset,
            stage="post_train",
            metrics=metrics,
        )

        print("\n[训练完成]")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"adapter/model 已保存到: {args.output_dir}")

    if is_dist_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
