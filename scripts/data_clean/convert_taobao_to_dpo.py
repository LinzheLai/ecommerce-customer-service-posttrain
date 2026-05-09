from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DIR = BASE_DIR / "data" / "cleaned"

DEFAULT_TRAIN_TASKS = (
    (
        CLEAN_DIR / "train_clean_top5000_raw.txt",
        BASE_DIR / "data" / "processed_5000" / "taobao_dpo_train.json",
    ),
    (
        CLEAN_DIR / "train_clean_top10000_raw.txt",
        BASE_DIR / "data" / "processed_10000" / "taobao_dpo_train.json",
    ),
    (
        CLEAN_DIR / "train_clean_top23000_raw.txt",
        BASE_DIR / "data" / "processed_23000" / "taobao_dpo_train.json",
    ),
)

PREFERENCE_INSTRUCTION = "你是电商平台客服偏好优化数据构建器，请偏向更准确、相关、礼貌且可执行的回复。"


def clean_text(text: str) -> str:
    return "" if text is None else str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_raw_line(line: str) -> Tuple[int, List[str], str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 3:
        raise ValueError(f"bad raw line: {line!r}")
    label = int(parts[0])
    context = [clean_text(x) for x in parts[1:-1] if x.strip()]
    response = clean_text(parts[-1])
    return label, context, response


def prompt_from_context(context: Sequence[str]) -> str:
    lines: List[str] = []
    for idx, utt in enumerate(context):
        role = "用户" if idx % 2 == 0 else "客服"
        lines.append(f"{role}：{utt}")
    return "\n".join(lines)


def load_groups(raw_path: Path) -> List[Tuple[List[str], str, str]]:
    groups: Dict[Tuple[str, ...], Dict[int, str]] = {}
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            label, context, response = parse_raw_line(line)
            if label not in (0, 1):
                continue
            key = tuple(context)
            bucket = groups.setdefault(key, {})
            bucket[label] = response

    pairs: List[Tuple[List[str], str, str]] = []
    for context_key, bucket in groups.items():
        if 1 not in bucket or 0 not in bucket:
            continue
        chosen = bucket[1]
        rejected = bucket[0]
        if not chosen or not rejected or chosen == rejected:
            continue
        pairs.append((list(context_key), chosen, rejected))
    return pairs


def convert_one(raw_path: Path, output_path: Path) -> Dict[str, object]:
    pairs = load_groups(raw_path)
    payload = [
        {
            "instruction": PREFERENCE_INSTRUCTION,
            "input": prompt_from_context(context),
            "chosen": chosen,
            "rejected": rejected,
        }
        for context, chosen, rejected in pairs
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input": str(raw_path),
        "output": str(output_path),
        "samples": len(payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert repaired raw DPO pairs into training JSON.")
    parser.add_argument("--input", type=str, default=None, help="Single raw input path")
    parser.add_argument("--output", type=str, default=None, help="Single json output path")
    args = parser.parse_args()

    if args.input or args.output:
        if not (args.input and args.output):
            raise ValueError("single-file mode requires both --input and --output")
        summary = convert_one(Path(args.input), Path(args.output))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    summary = [convert_one(raw_path, output_path) for raw_path, output_path in DEFAULT_TRAIN_TASKS]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
