import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    "你是电商客服。只回答最后一句用户问题。答案必须简短、直接、保守，最多2句，"
    "不主动扩展，不重复寒暄。不确定时说需要帮您核实。"
)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").replace("\u0000", "").strip()


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def render_chatml(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    parts = []
    for msg in messages:
        role = normalize_text(msg.get("role", "")).lower()
        content = normalize_text(msg.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def ensure_completion_suffix(text: str, completion_end_token: str) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    stripped = re.sub(r"(<\|im_end\|>|<\|endoftext\|>)+\s*$", "", text).rstrip()
    if completion_end_token == "eos":
        suffix = "<|endoftext|>"
    elif completion_end_token == "both":
        suffix = "<|im_end|>\n<|endoftext|>"
    else:
        suffix = "<|im_end|>"
    return stripped + suffix + "\n"


def normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if not isinstance(messages, list):
        return []
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = normalize_text(msg.get("role", "")).lower()
        content = normalize_text(msg.get("content", ""))
        if role in {"system", "user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


def build_prompt(row: Dict[str, Any], system_prompt: str, force_replace_system: bool) -> str:
    prompt = row.get("prompt", "")
    if isinstance(prompt, str):
        text = normalize_text(prompt)
        if text:
            return text

    messages = normalize_messages(prompt)
    if not messages and "messages" in row:
        messages = normalize_messages(row.get("messages"))
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]

    if messages:
        if messages[0]["role"] == "system":
            if force_replace_system:
                messages[0] = {"role": "system", "content": normalize_text(system_prompt)}
        else:
            messages = [{"role": "system", "content": normalize_text(system_prompt)}] + messages
        return render_chatml(messages, add_generation_prompt=True)

    user_input = normalize_text(row.get("input", ""))
    if not user_input:
        raise ValueError("Cannot build prompt from row.")
    return render_chatml(
        [
            {"role": "system", "content": normalize_text(system_prompt)},
            {"role": "user", "content": user_input},
        ],
        add_generation_prompt=True,
    )


def truncate_prompt_completion(
    prompt_ids: List[int],
    completion_ids: List[int],
    max_length: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> Tuple[List[int], List[int]]:
    if max_completion_length > 0 and len(completion_ids) > max_completion_length:
        completion_ids = completion_ids[:max_completion_length]

    if max_prompt_length > 0 and len(prompt_ids) > max_prompt_length:
        prompt_ids = prompt_ids[-max_prompt_length:]

    if len(prompt_ids) + len(completion_ids) > max_length:
        keep_prompt_len = max(0, max_length - len(completion_ids))
        prompt_ids = prompt_ids[-keep_prompt_len:] if keep_prompt_len > 0 else []
        if len(prompt_ids) + len(completion_ids) > max_length:
            completion_ids = completion_ids[:max_length]
            prompt_ids = []
    return prompt_ids, completion_ids


@torch.no_grad()
def sequence_logprob(
    model,
    tokenizer,
    prompt_text: str,
    completion_text: str,
    max_length: int,
    max_prompt_length: int,
    max_completion_length: int,
) -> Tuple[float, int]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion_text, add_special_tokens=False)["input_ids"]
    prompt_ids, completion_ids = truncate_prompt_completion(
        prompt_ids,
        completion_ids,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
    )
    if not completion_ids:
        return float("-inf"), 0

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    label_tensor = torch.tensor([labels], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_tensor, device=model.device)

    logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits[:, :-1, :]
    shift_labels = label_tensor[:, 1:]
    valid_mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~valid_mask, 0)

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    total = token_log_probs.masked_fill(~valid_mask, 0.0).sum().item()
    count = valid_mask.sum().item()
    return float(total), int(count)


def load_model(base_model: str, adapter_path: str, trust_remote_code: bool, use_simple_chatml: bool):
    tokenizer_source = adapter_path if adapter_path.lower() not in {"none", "base", "null"} else base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_simple_chatml:
        tokenizer.chat_template = SIMPLE_CHATML_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    if adapter_path.lower() not in {"none", "base", "null"}:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DPO preference win rate by chosen/rejected log-prob.")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--force_replace_system", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--completion_end_token", choices=["im_end", "eos", "both"], default="eos")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=640)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=64)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_simple_chatml", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(
        args.base_model,
        args.adapter_path,
        trust_remote_code=args.trust_remote_code,
        use_simple_chatml=args.use_simple_chatml,
    )

    rows = load_json_or_jsonl(args.eval_path)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    results = []
    wins = ties = valid = 0
    margins = []
    norm_margins = []

    for idx, row in enumerate(tqdm(rows, desc="Preference eval")):
        try:
            prompt_text = build_prompt(row, args.system_prompt, args.force_replace_system)
            chosen = ensure_completion_suffix(row.get("chosen", ""), args.completion_end_token)
            rejected = ensure_completion_suffix(row.get("rejected", ""), args.completion_end_token)
            chosen_lp, chosen_tokens = sequence_logprob(
                model,
                tokenizer,
                prompt_text,
                chosen,
                args.max_length,
                args.max_prompt_length,
                args.max_completion_length,
            )
            rejected_lp, rejected_tokens = sequence_logprob(
                model,
                tokenizer,
                prompt_text,
                rejected,
                args.max_length,
                args.max_prompt_length,
                args.max_completion_length,
            )
        except Exception as exc:
            results.append({"id": idx, "error": str(exc)})
            continue

        if not math.isfinite(chosen_lp) or not math.isfinite(rejected_lp):
            results.append({"id": idx, "error": "non_finite_logprob"})
            continue

        valid += 1
        margin = chosen_lp - rejected_lp
        chosen_avg = chosen_lp / max(chosen_tokens, 1)
        rejected_avg = rejected_lp / max(rejected_tokens, 1)
        norm_margin = chosen_avg - rejected_avg
        margins.append(margin)
        norm_margins.append(norm_margin)
        if margin > 0:
            wins += 1
            result = "win"
        elif margin == 0:
            ties += 1
            result = "tie"
        else:
            result = "loss"

        results.append(
            {
                "id": idx,
                "result": result,
                "chosen_logprob": round(chosen_lp, 6),
                "rejected_logprob": round(rejected_lp, 6),
                "margin": round(margin, 6),
                "chosen_avg_logprob": round(chosen_avg, 6),
                "rejected_avg_logprob": round(rejected_avg, 6),
                "normalized_margin": round(norm_margin, 6),
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "noise_swapped": row.get("_noise_swapped"),
            }
        )

    summary = {
        "adapter_path": args.adapter_path,
        "eval_path": args.eval_path,
        "num_samples": len(rows),
        "valid_samples": valid,
        "preference_accuracy": round(wins / valid, 6) if valid else 0.0,
        "tie_rate": round(ties / valid, 6) if valid else 0.0,
        "mean_margin": round(sum(margins) / valid, 6) if valid else 0.0,
        "mean_normalized_margin": round(sum(norm_margins) / valid, 6) if valid else 0.0,
        "max_length": args.max_length,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "completion_end_token": args.completion_end_token,
    }

    (output_dir / "preference_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "preference_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
