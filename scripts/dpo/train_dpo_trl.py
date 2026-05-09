import argparse
import hashlib
import inspect
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist


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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None


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
    parser = argparse.ArgumentParser(description="电商客服 DPO 偏好训练脚本")

    parser.add_argument("--train_path", type=str, required=True, help="训练集路径，支持 json/jsonl")
    parser.add_argument("--val_path", type=str, default=None, help="验证集路径")
    parser.add_argument("--validation_split_ratio", type=float, default=0.0)
    parser.add_argument("--dataset_num_proc", type=int, default=1)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--drop_empty_samples", action="store_true")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/opt/data/llz/hf_models/Qwen3-8B-Base",
        help="基座模型路径或模型名",
    )
    parser.add_argument(
        "--sft_adapter_path",
        type=str,
        required=True,
        help="SFT 阶段产出的 adapter 路径，DPO 将在此基础上继续训练",
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
        help="覆盖默认 chat template，统一使用简洁 ChatML 模板",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="DPO 阶段统一使用的 system prompt；建议与 SFT 保持一致",
    )
    parser.add_argument(
        "--force_replace_system",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否忽略原始 instruction，只保留统一的客服 system prompt",
    )
    parser.add_argument(
        "--completion_end_token",
        type=str,
        default="im_end",
        choices=["im_end", "eos", "both"],
        help=(
            "chosen/rejected 结尾监督符。im_end 使用 <|im_end|>；"
            "eos 使用 <|endoftext|>；both 同时使用两者。"
            "若 DPO 接在 EOS 短答 SFT 后，建议使用 eos。"
        ),
    )

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", type=str, default="sigmoid")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--truncation_mode", type=str, default="keep_end")
    parser.add_argument("--max_length", type=int, default=768)
    parser.add_argument("--max_prompt_length", type=int, default=640)
    parser.add_argument("--max_completion_length", type=int, default=64)
    parser.add_argument("--precompute_ref_log_probs", action="store_true")

    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none", help="none / tensorboard / wandb")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

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


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u0000", "")
    return text.strip()


def parse_prefixed_dialogue(text: str) -> List[Dict]:
    messages: List[Dict] = []
    current_role = None
    current_content: List[str] = []

    def flush():
        nonlocal current_role, current_content
        if current_role and current_content:
            content = "\n".join(x for x in current_content if x).strip()
            if content:
                messages.append({"role": current_role, "content": content})
        current_role = None
        current_content = []

    for raw_line in normalize_text(text).split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        matched = False
        for prefix, role in (
            ("用户：", "user"),
            ("用户:", "user"),
            ("客服：", "assistant"),
            ("客服:", "assistant"),
            ("系统：", "system"),
            ("系统:", "system"),
        ):
            if line.startswith(prefix):
                flush()
                current_role = role
                current_content = [line[len(prefix):].strip()]
                matched = True
                break

        if matched:
            continue

        if current_role is None:
            current_role = "user"
            current_content = [line]
        else:
            current_content.append(line)

    flush()
    return [m for m in messages if normalize_text(m.get("content"))]


def apply_consistent_system_prompt(messages: List[Dict], system_prompt: str, force_replace_system: bool = True) -> List[Dict]:
    system_prompt = normalize_text(system_prompt)
    if not system_prompt:
        return messages
    if not messages:
        return [{"role": "system", "content": system_prompt}]
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


def convert_row(
    row: Dict,
    system_prompt: str,
    force_replace_system: bool,
    completion_end_token: str = "im_end",
):
    if all(k in row for k in ("prompt", "chosen", "rejected")):
        prompt_text = normalize_text(row.get("prompt", ""))
        chosen_text = ensure_completion_suffix(row.get("chosen", ""), completion_end_token)
        rejected_text = ensure_completion_suffix(row.get("rejected", ""), completion_end_token)
        if not prompt_text or not chosen_text or not rejected_text or chosen_text == rejected_text:
            return None
        return {
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
        }

    input_text = normalize_text(row.get("input", ""))
    chosen_text = normalize_text(row.get("chosen", ""))
    rejected_text = normalize_text(row.get("rejected", ""))
    if not input_text or not chosen_text or not rejected_text or chosen_text == rejected_text:
        return None

    prompt_messages = parse_prefixed_dialogue(input_text)
    prompt_messages = apply_consistent_system_prompt(
        prompt_messages,
        system_prompt=system_prompt,
        force_replace_system=force_replace_system,
    )
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        return None

    return {
        "prompt": render_chatml(prompt_messages, add_generation_prompt=True),
        "chosen": ensure_completion_suffix(chosen_text, completion_end_token),
        "rejected": ensure_completion_suffix(rejected_text, completion_end_token),
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
        ds = ds.filter(
            lambda x: bool(str(x["prompt"]).strip())
            and bool(str(x["chosen"]).strip())
            and bool(str(x["rejected"]).strip()),
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
            "chosen_preview": sample["chosen"][:200],
            "rejected_preview": sample["rejected"][:200],
        }
        main_print("[数据预览]")
        main_print(json.dumps(preview, ensure_ascii=False, indent=2))

    return ds


def build_model_and_tokenizer(args):
    dtype, use_bf16, use_fp16 = maybe_get_dtype()

    tokenizer_source = args.sft_adapter_path if Path(args.sft_adapter_path).exists() else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if args.use_simple_chatml:
        tokenizer.chat_template = SIMPLE_CHATML_TEMPLATE
        main_print("[模板] 已覆盖为简洁 ChatML 模板。")

    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }

    if args.train_mode == "qlora":
        try:
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
        except ImportError as e:
            raise ImportError("QLoRA 模式需要 bitsandbytes 与 peft 的量化支持，请先安装 bitsandbytes。") from e

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

        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    model = PeftModel.from_pretrained(
        model,
        args.sft_adapter_path,
        is_trainable=True,
    )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    return model, tokenizer, use_bf16, use_fp16


def ensure_trl_model_compat(model):
    # Some TRL versions write to model.warnings_issued during trainer init,
    # but PEFT-wrapped Qwen models may not expose that attribute by default.
    if not hasattr(model, "warnings_issued") or getattr(model, "warnings_issued", None) is None:
        model.warnings_issued = {}

    base_candidates = [
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in base_candidates:
        if candidate is not None and (
            not hasattr(candidate, "warnings_issued") or getattr(candidate, "warnings_issued", None) is None
        ):
            candidate.warnings_issued = {}


def build_dpo_config(args, use_bf16: bool, use_fp16: bool, has_eval: bool):
    config_cls = DPOConfig if DPOConfig is not None else transformers.TrainingArguments
    sig = inspect.signature(config_cls.__init__)
    params = sig.parameters

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
        bf16=use_bf16,
        fp16=use_fp16,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )

    optional_values = {
        "deepspeed": args.deepspeed,
        "dataloader_num_workers": args.dataloader_num_workers,
        "ddp_find_unused_parameters": False,
        "gradient_checkpointing_kwargs": {"use_reentrant": False} if args.gradient_checkpointing else None,
        "logging_first_step": True,
        "remove_unused_columns": False,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "truncation_mode": args.truncation_mode,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
    }
    for key, value in optional_values.items():
        if key in params and value is not None:
            cfg[key] = value

    if has_eval:
        if "eval_steps" in params:
            cfg["eval_steps"] = args.eval_steps
        if "eval_strategy" in params:
            cfg["eval_strategy"] = "steps"
        elif "evaluation_strategy" in params:
            cfg["evaluation_strategy"] = "steps"
        if "load_best_model_at_end" in params:
            cfg["load_best_model_at_end"] = True
        if "metric_for_best_model" in params:
            cfg["metric_for_best_model"] = "eval_loss"
        if "greater_is_better" in params:
            cfg["greater_is_better"] = False
    else:
        if "eval_strategy" in params:
            cfg["eval_strategy"] = "no"
        elif "evaluation_strategy" in params:
            cfg["evaluation_strategy"] = "no"

    if "save_strategy" in params:
        cfg["save_strategy"] = "steps"

    return config_cls(**cfg), set(cfg.keys())


def build_trainer(model, tokenizer, train_dataset, eval_dataset, dpo_config, config_keys, args):
    sig = inspect.signature(DPOTrainer.__init__)
    params = sig.parameters

    kwargs = dict(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer

    direct_fallbacks = {
        "beta": args.beta,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "truncation_mode": args.truncation_mode,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
    }
    for key, value in direct_fallbacks.items():
        if key in params and key not in config_keys:
            kwargs[key] = value

    return DPOTrainer(**kwargs)


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.split("-")[-1])
    except Exception:
        return -1


def list_checkpoint_dirs(output_dir: str) -> List[Path]:
    base = Path(output_dir)
    if not base.exists():
        return []
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


def sync_root_adapter_from_latest_checkpoint(output_dir: str) -> Dict:
    base = Path(output_dir)
    checkpoints = list_checkpoint_dirs(output_dir)
    latest_checkpoint = checkpoints[-1] if checkpoints else None

    result = {
        "synced": False,
        "latest_checkpoint_dir": latest_checkpoint.name if latest_checkpoint is not None else None,
        "copied_files": [],
    }
    if latest_checkpoint is None:
        return result

    # Keep root adapter aligned with the final checkpoint so downstream
    # inference/eval can safely point to output_dir without ambiguity.
    candidate_files = [
        "adapter_model.safetensors",
        "adapter_config.json",
    ]
    for name in candidate_files:
        src = latest_checkpoint / name
        dst = base / name
        if src.exists():
            shutil.copy2(src, dst)
            result["copied_files"].append(name)

    result["synced"] = bool(result["copied_files"])
    return result


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
        "drop_empty_samples": args.drop_empty_samples,
        "model_name_or_path": args.model_name_or_path,
        "sft_adapter_path": args.sft_adapter_path,
        "trust_remote_code": args.trust_remote_code,
        "use_fast_tokenizer": args.use_fast_tokenizer,
        "train_mode": args.train_mode,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "gradient_checkpointing": args.gradient_checkpointing,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_total_limit": args.save_total_limit,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
        "output_dir": args.output_dir,
        "deepspeed": args.deepspeed,
        "use_simple_chatml": args.use_simple_chatml,
        "system_prompt": args.system_prompt,
        "force_replace_system": args.force_replace_system,
        "completion_end_token": args.completion_end_token,
        "beta": args.beta,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "truncation_mode": args.truncation_mode,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "precompute_ref_log_probs": args.precompute_ref_log_probs,
        "reference_model_strategy": "ref_model=None (use initial SFT policy as implicit reference)",
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

    model, tokenizer, use_bf16, use_fp16 = build_model_and_tokenizer(args)
    ensure_trl_model_compat(model)
    dpo_config, config_keys = build_dpo_config(args, use_bf16, use_fp16, loaded_eval_dataset is not None)

    save_run_summary(args, train_dataset, loaded_eval_dataset, stage="pre_train")
    save_dataset_preview(args.output_dir, train_dataset, loaded_eval_dataset)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=loaded_eval_dataset,
        dpo_config=dpo_config,
        config_keys=config_keys,
        args=args,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)

    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)

    metrics = dict(train_result.metrics)
    if loaded_eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics.update(eval_metrics)

    if is_main_process():
        sync_info = sync_root_adapter_from_latest_checkpoint(args.output_dir)
        if sync_info.get("synced"):
            print(
                f"[保存同步] 已将最新 checkpoint ({sync_info['latest_checkpoint_dir']}) 的 adapter 同步回根目录: "
                f"{', '.join(sync_info['copied_files'])}"
            )

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
