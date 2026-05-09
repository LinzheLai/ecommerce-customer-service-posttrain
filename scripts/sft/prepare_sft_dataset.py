"""
将原始 instruction / input / output 格式数据
转换为 TRL SFTTrainer 可直接使用的 conversational prompt-completion 格式。

输入样例:
{
  "instruction": "...",
  "input": "...",
  "output": "..."
}

输出样例(JSONL):
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "completion": [
    {"role": "assistant", "content": "..."}
  ]
}
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    # 保留多轮对话中的换行，仅做首尾空白清理
    return str(text).replace("\r\n", "\n").replace("\r", "\n").strip()


def validate_sample(sample: Dict, idx: int) -> Tuple[bool, str]:
    required = ["instruction", "input", "output"]
    for key in required:
        if key not in sample:
            return False, f"sample[{idx}] 缺少字段: {key}"

    if not normalize_text(sample["input"]):
        return False, f"sample[{idx}] input 为空"
    if not normalize_text(sample["output"]):
        return False, f"sample[{idx}] output 为空"

    return True, ""


def convert_sample(sample: Dict) -> Dict:
    instruction = normalize_text(sample["instruction"])
    user_input = normalize_text(sample["input"])
    output = normalize_text(sample["output"])

    # 保留 instruction 作为 system 角色，更贴合 TRL 的 conversational prompt-completion 格式
    prompt = []
    if instruction:
        prompt.append({"role": "system", "content": instruction})
    prompt.append({"role": "user", "content": user_input})

    completion = [{"role": "assistant", "content": output}]

    return {
        "prompt": prompt,
        "completion": completion,
    }


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="原始 json 文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--val_ratio", type=float, default=0.02, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", action="store_true", help="是否对完全重复样本去重")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("输入文件必须是 JSON 数组。")

    raw_count = len(data)
    valid_rows = []
    invalid_rows = []

    for idx, sample in enumerate(data):
        ok, msg = validate_sample(sample, idx)
        if not ok:
            invalid_rows.append(msg)
            continue
        valid_rows.append(sample)

    dedup_removed = 0
    if args.deduplicate:
        uniq = []
        seen = set()
        for sample in valid_rows:
            key = (
                normalize_text(sample["instruction"]),
                normalize_text(sample["input"]),
                normalize_text(sample["output"]),
            )
            if key in seen:
                dedup_removed += 1
                continue
            seen.add(key)
            uniq.append(sample)
        valid_rows = uniq

    random.seed(args.seed)
    random.shuffle(valid_rows)

    converted = [convert_sample(x) for x in valid_rows]

    if args.val_ratio > 0:
        val_size = max(1, int(len(converted) * args.val_ratio))
    else:
        val_size = 0

    val_rows = converted[:val_size]
    train_rows = converted[val_size:]

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "stats.json"
    invalid_path = output_dir / "invalid_samples.txt"

    write_jsonl(train_path, train_rows)
    if val_rows:
        write_jsonl(val_path, val_rows)

    stats = {
        "raw_count": raw_count,
        "valid_count": len(converted),
        "invalid_count": len(invalid_rows),
        "dedup_removed": dedup_removed,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "input_path": str(input_path),
        "train_path": str(train_path),
        "val_path": str(val_path) if val_rows else None,
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if invalid_rows:
        with invalid_path.open("w", encoding="utf-8") as f:
            for row in invalid_rows:
                f.write(row + "\n")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if train_rows:
        print("\n[train sample]")
        print(json.dumps(train_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
