import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

TASK_PREFIX = "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。如果历史对话里没有明确信息，只能给出简短、保守回复"

DEFAULT_PAIRS = [
    (
        "data/processed_5000/taobao_sft_train.json",
        "data/processed_5000/taobao_messages_train.json",
    ),
    (
        "data/processed_5000/taobao_sft_dev.json",
        "data/processed_5000/taobao_messages_dev.json",
    ),
    (
        "data/processed_5000/taobao_sft_test.json",
        "data/processed_5000/taobao_messages_test.json",
    ),
]


def basic_clean(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\ufeff", "")
    text = text.replace(":", "：")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \u3000]{2,}", " ", text)
    return text.strip()


def strip_task_prefix(text: str) -> str:
    text = basic_clean(text)
    pattern = r"^\s*请根据下面的电商客服历史对话[，,]生成下一句合适的客服回复。\s*"
    text = re.sub(pattern, "", text, count=1)
    return text.strip()


def parse_turns(input_text: str) -> List[Tuple[str, str]]:
    dialogue = strip_task_prefix(input_text)
    parts = re.split(r"(用户：|客服：)", dialogue)
    turns: List[Tuple[str, str]] = []

    i = 1
    while i < len(parts):
        marker = parts[i]
        content = parts[i + 1] if i + 1 < len(parts) else ""
        role = "user" if marker == "用户：" else "assistant"
        content = basic_clean(content)
        if content:
            turns.append((role, content))
        i += 2

    return turns


def load_records(path: Path) -> List[Dict]:
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def save_json(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def convert(records: List[Dict]) -> List[Dict]:
    out = []
    default_sys = "你是电商平台客服，请基于历史对话给出准确、礼貌、可执行的回复。"

    for rec in records:
        instruction = basic_clean(rec.get("instruction", default_sys)) or default_sys
        output_text = basic_clean(rec.get("output", ""))
        if not output_text:
            continue

        turns = parse_turns(rec.get("input", ""))

        messages = [{"role": "system", "content": instruction}]
        for role, content in turns:
            messages.append({"role": role, "content": content})
        messages.append({"role": "assistant", "content": output_text})

        out.append({"messages": messages})

    return out


def process_one(input_path: str, output_path: str) -> Dict:
    records = load_records(Path(input_path))
    out = convert(records)
    save_json(Path(output_path), out)
    return {
        "input": input_path,
        "output": output_path,
        "total": len(out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.input and args.output:
        result = process_one(args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = []
    for input_path, output_path in DEFAULT_PAIRS:
        result = process_one(input_path, output_path)
        summary.append(result)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()