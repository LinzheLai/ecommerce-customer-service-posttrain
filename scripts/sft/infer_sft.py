import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from peft import PeftModel
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
    "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。"
    "如果历史对话里没有明确信息，只能给出简短、保守回复。"
)

DEFAULT_BASE_MODEL = "/opt/data/llz/hf_models/Qwen3-8B-Base"
DEFAULT_SFT_ADAPTER = "/opt/data/llz/ecommerce-customer-service-posttrain/outputs/sft/qwen3_8b_rank16_zero2"
DEFAULT_TEST_PATH = "/opt/data/llz/ecommerce-customer-service-posttrain/data/processed_5000/taobao_messages_test.json"
DEFAULT_OUTPUT_PATH = "/opt/data/llz/ecommerce-customer-service-posttrain/outputs/sft/batch_infer/taobao_messages_test_random20_5000.json"

ROLE_TAIL_RE = re.compile(r"(?:^|\s)(?:user|assistant|system)\s*(?:\n|:|：)", flags=re.IGNORECASE)
NOISE_UNICODE_RE = re.compile(r"[\u0E00-\u0E7F\U00010000-\U0010FFFF]")
LONG_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]{4,}")
MULTI_SPACE_RE = re.compile(r"\s+")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified batch inference for base / SFT / DPO comparison."
    )
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=DEFAULT_SFT_ADAPTER,
        help="LoRA adapter path. Set empty / none / null / base to run the pure base model.",
    )
    parser.add_argument("--test_path", type=str, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--max_output_chars", type=int, default=56)
    parser.add_argument(
        "--use_simple_chatml",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="If set, run single-prompt inference instead of random batch inference.",
    )
    parser.add_argument(
        "--sample_indices_path",
        type=str,
        default=None,
        help="Optional JSON / JSONL file describing which dataset indices to reuse.",
    )
    parser.add_argument(
        "--save_sample_indices_path",
        type=str,
        default=None,
        help="Optional path to save the chosen dataset indices for later comparison runs.",
    )
    parser.add_argument(
        "--model_label",
        type=str,
        default=None,
        help="Optional label written to output meta, e.g. base / sft / dpo.",
    )
    return parser.parse_args()


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_optional_path(path_text: Optional[str]) -> Optional[str]:
    text = normalize_text(path_text)
    if not text:
        return None
    if text.lower() in {"none", "null", "base", "no_adapter"}:
        return None
    return text


def load_json_or_jsonl_raw(path: str) -> Any:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if file_path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    payload = load_json_or_jsonl_raw(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list payload: {path}")
    return payload


def ensure_unique_indices(indices: Sequence[int]) -> List[int]:
    seen = set()
    unique: List[int] = []
    for idx in indices:
        if idx in seen:
            raise ValueError(f"Duplicate sample index found: {idx}")
        seen.add(idx)
        unique.append(idx)
    return unique


def normalize_message_signature(messages: Sequence[Dict[str, Any]]) -> Tuple[Tuple[str, str], ...]:
    signature: List[Tuple[str, str]] = []
    for message in messages:
        role = normalize_text(message.get("role", "")).lower()
        if role == "system":
            continue
        content = normalize_text(message.get("content", ""))
        signature.append((role, content))
    return tuple(signature)


def build_dataset_signature_maps(
    samples: Sequence[Dict[str, Any]],
    system_prompt: str,
) -> Tuple[Dict[Tuple[Any, ...], List[int]], Dict[Tuple[str, str], List[int]], Dict[str, List[int]]]:
    history_map: Dict[Tuple[Any, ...], List[int]] = {}
    prompt_ref_map: Dict[Tuple[str, str], List[int]] = {}
    prompt_only_map: Dict[str, List[int]] = {}

    for idx, sample in enumerate(samples):
        prompt_messages, prompt_text, reference = extract_prompt_and_reference(sample, system_prompt)
        history_signature = (normalize_message_signature(prompt_messages), reference)
        prompt_ref_signature = (prompt_text, reference)

        history_map.setdefault(history_signature, []).append(idx)
        prompt_ref_map.setdefault(prompt_ref_signature, []).append(idx)
        prompt_only_map.setdefault(prompt_text, []).append(idx)

    return history_map, prompt_ref_map, prompt_only_map


def pop_first_available(indices: Sequence[int], used_indices: set[int]) -> Optional[int]:
    for idx in indices:
        if idx not in used_indices:
            return idx
    return None


def resolve_legacy_result_indices(
    results: Sequence[Dict[str, Any]],
    samples: Sequence[Dict[str, Any]],
    system_prompt: str,
) -> Optional[List[int]]:
    if not results:
        return None

    history_map, prompt_ref_map, prompt_only_map = build_dataset_signature_maps(samples, system_prompt)
    used_indices: set[int] = set()
    resolved: List[int] = []

    for item in results:
        if not isinstance(item, dict):
            return None

        result_messages = item.get("messages")
        prompt_text = normalize_text(item.get("prompt", ""))
        reference = normalize_text(item.get("reference", ""))
        candidate_index: Optional[int] = None

        if isinstance(result_messages, list):
            history_signature = (normalize_message_signature(result_messages), reference)
            candidate_index = pop_first_available(history_map.get(history_signature, []), used_indices)

        if candidate_index is None and prompt_text:
            candidate_index = pop_first_available(
                prompt_ref_map.get((prompt_text, reference), []),
                used_indices,
            )

        if candidate_index is None and prompt_text:
            candidate_index = pop_first_available(prompt_only_map.get(prompt_text, []), used_indices)

        if candidate_index is None:
            return None

        used_indices.add(candidate_index)
        resolved.append(candidate_index)

    return resolved


def load_sample_indices(
    path: str,
    samples: Optional[Sequence[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
) -> List[int]:
    payload = load_json_or_jsonl_raw(path)
    indices: Optional[List[int]] = None

    if isinstance(payload, dict):
        if isinstance(payload.get("meta"), dict):
            meta_indices = payload["meta"].get("sample_indices")
            if isinstance(meta_indices, list):
                indices = meta_indices

        if indices is None and isinstance(payload.get("indices"), list):
            indices = payload["indices"]

        if indices is None and isinstance(payload.get("sample_indices"), list):
            indices = payload["sample_indices"]

        if indices is None and isinstance(payload.get("results"), list):
            result_indices = [item.get("source_index") for item in payload["results"]]
            if result_indices and all(isinstance(idx, int) for idx in result_indices):
                indices = result_indices

        if (
            indices is None
            and samples is not None
            and system_prompt is not None
            and isinstance(payload.get("results"), list)
        ):
            indices = resolve_legacy_result_indices(payload["results"], samples, system_prompt)

    elif isinstance(payload, list):
        if payload and all(isinstance(item, int) for item in payload):
            indices = list(payload)
        elif payload and all(isinstance(item, dict) and isinstance(item.get("source_index"), int) for item in payload):
            indices = [item["source_index"] for item in payload]

    if not indices:
        raise ValueError(f"Could not parse sample indices from: {path}")

    return ensure_unique_indices([int(idx) for idx in indices])


def save_sample_indices(path: str, test_path: str, seed: int, indices: Sequence[int]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "test_path": test_path,
        "seed": seed,
        "num_samples": len(indices),
        "indices": list(indices),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_model_label(adapter_path: Optional[str], explicit_label: Optional[str]) -> str:
    if explicit_label:
        return normalize_text(explicit_label)
    if not adapter_path:
        return "base"
    return Path(adapter_path).name or "adapter"


def load_model(base_model: str, adapter_path: Optional[str], use_simple_chatml: bool):
    tokenizer_source = adapter_path or base_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=False,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if use_simple_chatml:
        tokenizer.chat_template = SIMPLE_CHATML_TEMPLATE

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer, tokenizer_source


def get_stop_ids(tokenizer):
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0 and im_end_id != tokenizer.unk_token_id:
        stop_ids.append(im_end_id)
    return list(dict.fromkeys(stop_ids))


def clean_prediction(text: str, max_output_chars: int = 56) -> str:
    text = normalize_text(text)
    cut_markers = [
        "<|im_end|>",
        "<|im_start|>user",
        "<|im_start|>assistant",
        "<|im_start|>system",
        "\nuser\n",
        "\nassistant\n",
        "\nsystem\n",
        "</s>",
        "<|endoftext|>",
    ]
    for marker in cut_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    role_tail = ROLE_TAIL_RE.search(text)
    if role_tail:
        text = text[:role_tail.start()].strip()

    text = text.replace("ForCanBeConvertedToForeach", " ")
    text = NOISE_UNICODE_RE.sub("", text)
    text = LONG_LATIN_TOKEN_RE.sub("", text)
    text = MULTI_SPACE_RE.sub(" ", text).strip()

    if max_output_chars > 0 and len(text) > max_output_chars:
        text = text[:max_output_chars].rstrip()
    return text


def extract_prompt_and_reference(
    sample: Dict[str, Any],
    default_system_prompt: str,
) -> Tuple[List[Dict[str, str]], str, str]:
    if "messages" in sample:
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("messages sample format is invalid.")
        if messages[-1].get("role") != "assistant":
            raise ValueError("The last message must be assistant.")

        reference = normalize_text(messages[-1].get("content", ""))
        prompt_messages = messages[:-1]

        if prompt_messages and prompt_messages[0].get("role") == "system":
            prompt_messages[0] = {"role": "system", "content": default_system_prompt}
        else:
            prompt_messages = [{"role": "system", "content": default_system_prompt}] + prompt_messages

        prompt_text = normalize_text(prompt_messages[-1].get("content", "")) if prompt_messages else ""
        return prompt_messages, prompt_text, reference

    instruction = normalize_text(sample.get("instruction", "")) or default_system_prompt
    user_input = normalize_text(sample.get("input", ""))
    reference = normalize_text(sample.get("output", ""))

    prompt_messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    return prompt_messages, user_input, reference


def build_single_prompt_messages(prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": normalize_text(system_prompt)},
        {"role": "user", "content": normalize_text(prompt)},
    ]


def move_inputs_to_model_device(model, tokenizer_inputs):
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(model.parameters()).device
    return tokenizer_inputs.to(model_device)


def generate_one(model, tokenizer, prompt_messages: List[Dict[str, str]], args) -> str:
    text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    inputs = move_inputs_to_model_device(model, inputs)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=get_stop_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    if args.do_sample:
        generate_kwargs.update(
            dict(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        )
    else:
        generate_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)

    gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return clean_prediction(pred, max_output_chars=args.max_output_chars)


def run_single_prompt(args, model, tokenizer):
    prompt_messages = build_single_prompt_messages(args.prompt, args.system_prompt)
    prediction = generate_one(model, tokenizer, prompt_messages, args)

    print("\n[Prompt]")
    print(args.prompt)
    print("\n[Response]")
    print(prediction)


def select_sample_indices(args, num_samples: int) -> List[int]:
    if args.sample_indices_path:
        raise RuntimeError("Dataset samples must be provided when loading sample indices.")

    sample_size = min(args.num_samples, num_samples)
    rng = random.Random(args.seed)
    return rng.sample(range(num_samples), k=sample_size)


def select_sample_indices_from_samples(args, samples: Sequence[Dict[str, Any]]) -> List[int]:
    if args.sample_indices_path:
        indices = load_sample_indices(
            args.sample_indices_path,
            samples=samples,
            system_prompt=args.system_prompt,
        )
        for idx in indices:
            if idx < 0 or idx >= len(samples):
                raise IndexError(
                    f"Sample index {idx} is out of range for dataset size {len(samples)}."
                )
        return indices

    return select_sample_indices(args, len(samples))


def run_batch_eval(args, model, tokenizer, tokenizer_source: str, adapter_path: Optional[str]):
    samples = load_json_or_jsonl(args.test_path)
    if not samples:
        raise ValueError(f"Test set is empty: {args.test_path}")

    selected_indices = select_sample_indices_from_samples(args, samples)
    if args.save_sample_indices_path:
        save_sample_indices(args.save_sample_indices_path, args.test_path, args.seed, selected_indices)

    results = []
    print(f"\n[INFO] Running batch inference on {len(selected_indices)} samples from {args.test_path}\n")

    for rank, source_index in enumerate(selected_indices, start=1):
        sample = samples[source_index]
        prompt_messages, prompt_text, reference = extract_prompt_and_reference(sample, args.system_prompt)
        prediction = generate_one(model, tokenizer, prompt_messages, args)

        record = {
            "id": rank,
            "source_index": source_index,
            "prompt": prompt_text,
            "prediction": prediction,
            "reference": reference,
            "messages": prompt_messages,
        }
        results.append(record)

        print(f"===== Sample {rank} / source_index={source_index} =====")
        print("[Prompt]")
        print(prompt_text)
        print("\n[Prediction]")
        print(prediction)
        print("\n[Reference]")
        print(reference)
        print()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "model_label": build_model_label(adapter_path, args.model_label),
            "base_model": args.base_model,
            "adapter_path": adapter_path,
            "tokenizer_source": tokenizer_source,
            "test_path": args.test_path,
            "num_samples": len(results),
            "seed": args.seed,
            "system_prompt": args.system_prompt,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature if args.do_sample else None,
            "top_p": args.top_p if args.do_sample else None,
            "top_k": args.top_k if args.do_sample else None,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_output_chars": args.max_output_chars,
            "sample_indices_path": normalize_optional_path(args.sample_indices_path),
            "save_sample_indices_path": normalize_optional_path(args.save_sample_indices_path),
            "sample_indices": selected_indices,
        },
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[INFO] Results saved to: {output_path}")


def main():
    args = parse_args()
    adapter_path = normalize_optional_path(args.adapter_path)
    model, tokenizer, tokenizer_source = load_model(
        args.base_model,
        adapter_path,
        args.use_simple_chatml,
    )

    if args.prompt:
        run_single_prompt(args, model, tokenizer)
    else:
        run_batch_eval(args, model, tokenizer, tokenizer_source, adapter_path)


if __name__ == "__main__":
    main()
