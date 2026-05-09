import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_ADAPTER_PATH = str(BASE_DIR / "outputs" / "sft" / "qwen3_8b_rank16_zero2")
DEFAULT_EVAL_PATH = str(BASE_DIR / "data" / "processed_5000" / "taobao_messages_test.json")
DEFAULT_OUTPUT_DIR = str(BASE_DIR / "outputs" / "sft" / "eval_qwen3_8b_test")

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

ROLE_TAIL_RE = re.compile(r"(?:^|\s)(?:user|assistant|system)\s*(?:\n|:|\uFF1A)", flags=re.IGNORECASE)
NOISE_UNICODE_RE = re.compile(r"[\u0E00-\u0E7F\U00010000-\U0010FFFF]")
LONG_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]{4,}")
MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def load_json_or_jsonl(path: str) -> List[Dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def extract_prompt_and_reference(sample: Dict, system_prompt: str) -> Tuple[List[Dict], str]:
    """
    返回：prompt_messages, reference
    支持两类格式：
    1) messages：默认将最后一条 assistant 作为 reference，并统一覆盖 system prompt
    2) instruction/input/output：统一使用传入的 system_prompt
    """
    if "messages" in sample:
        msgs = sample["messages"]
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError("messages 样本格式不合法")
        if msgs[-1].get("role") != "assistant":
            raise ValueError("messages 样本最后一条不是 assistant")

        reference = normalize_text(msgs[-1].get("content", ""))
        prompt_messages = msgs[:-1]

        # 统一覆盖 system prompt
        if prompt_messages and prompt_messages[0].get("role") == "system":
            prompt_messages[0] = {"role": "system", "content": normalize_text(system_prompt)}
        else:
            prompt_messages = [{"role": "system", "content": normalize_text(system_prompt)}] + prompt_messages

        return prompt_messages, reference

    user_input = normalize_text(sample.get("input", ""))
    reference = normalize_text(sample.get("output", ""))

    prompt_messages = [
        {"role": "system", "content": normalize_text(system_prompt)},
        {"role": "user", "content": user_input},
    ]
    return prompt_messages, reference


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def char_f1(pred: str, ref: str) -> float:
    pred_counter = Counter(list(pred))
    ref_counter = Counter(list(ref))
    common = sum((pred_counter & ref_counter).values())
    if common == 0:
        return 0.0
    precision = common / max(len(pred), 1)
    recall = common / max(len(ref), 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def rouge_l_char(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    lcs = lcs_length(list(pred), list(ref))
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    beta = 1.2
    return ((1 + beta ** 2) * prec * rec) / (rec + beta ** 2 * prec + 1e-12)


def exact_match(pred: str, ref: str) -> int:
    return int(pred.strip() == ref.strip())


def get_stop_ids(tokenizer):
    stop_ids = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id and im_end_id >= 0:
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


def generate_one(
    model,
    tokenizer,
    prompt_messages: List[Dict],
    max_new_tokens: int = 24,
    do_sample: bool = False,
    temperature: float = 0.6,
    top_p: float = 0.85,
    top_k: int = 50,
    repetition_penalty: float = 1.12,
    no_repeat_ngram_size: int = 4,
    max_output_chars: int = 56,
) -> str:
    text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        eos_token_id=get_stop_ids(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
    )
    if no_repeat_ngram_size and no_repeat_ngram_size > 0:
        generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    if do_sample:
        generate_kwargs.update(
            dict(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        )
    else:
        generate_kwargs.update(dict(do_sample=False))

    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)

    gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    pred = tokenizer.decode(gen_ids, skip_special_tokens=False)
    return clean_prediction(pred, max_output_chars=max_output_chars)


def build_badcases(results: List[Dict], top_k: int = 100) -> List[Dict]:
    valid = [x for x in results if "prediction" in x]
    valid = sorted(valid, key=lambda x: (x["char_f1"], x["rouge_l_char"], x["exact_match"]))
    return valid[:top_k]


def main():
    parser = argparse.ArgumentParser(description="SFT 测试集批量评估脚本")
    parser.add_argument("--adapter_path", type=str, default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--eval_path", type=str, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_new_tokens", type=int, default=24)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.12)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=4)
    parser.add_argument("--max_output_chars", type=int, default=56)
    parser.add_argument("--badcase_top_k", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps(
        {
            "adapter_path": args.adapter_path,
            "eval_path": args.eval_path,
            "output_dir": args.output_dir,
            "max_samples": args.max_samples,
        },
        ensure_ascii=False,
        indent=2,
    ))

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, use_fast=False, trust_remote_code=True)
    tokenizer.chat_template = SIMPLE_CHATML_TEMPLATE

    model = AutoPeftModelForCausalLM.from_pretrained(
        args.adapter_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    samples = load_json_or_jsonl(args.eval_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    results, em_list, f1_list, rouge_list = [], [], [], []
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        prompt_messages, reference = extract_prompt_and_reference(sample, args.system_prompt)
        prediction = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt_messages=prompt_messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            max_output_chars=args.max_output_chars,
        )

        em = exact_match(prediction, reference)
        f1 = char_f1(prediction, reference)
        rouge_l = rouge_l_char(prediction, reference)
        em_list.append(em)
        f1_list.append(f1)
        rouge_list.append(rouge_l)

        results.append({
            "id": idx,
            "prompt_messages": prompt_messages,
            "reference": reference,
            "prediction": prediction,
            "exact_match": em,
            "char_f1": round(f1, 6),
            "rouge_l_char": round(rouge_l, 6),
        })

    summary = {
        "num_samples": len(results),
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "top_k": args.top_k if args.do_sample else None,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_output_chars": args.max_output_chars,
        "exact_match": round(sum(em_list) / max(len(em_list), 1), 6),
        "char_f1": round(sum(f1_list) / max(len(f1_list), 1), 6),
        "rouge_l_char": round(sum(rouge_list) / max(len(rouge_list), 1), 6),
    }
    badcases = build_badcases(results, top_k=args.badcase_top_k)

    (output_dir / "predictions.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "badcases.json").write_text(json.dumps(badcases, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "summary": summary,
        "predictions_path": str(output_dir / "predictions.json"),
        "summary_path": str(output_dir / "summary.json"),
        "badcases_path": str(output_dir / "badcases.json"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
