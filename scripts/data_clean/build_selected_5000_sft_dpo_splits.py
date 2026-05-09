from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DEFAULT_PREVIEW = BASE_DIR / "data" / "cleaned" / "train_clean_top5000_preview.jsonl"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "selected_5000_sft_dpo"

SFT_INSTRUCTION = (
    "你是电商客服。只回答当前最后一句用户问题，不要扩展其他信息。"
    "如果历史对话里没有明确信息，只能给出简短、保守回复"
)
SFT_TASK_PREFIX = "请根据下面的电商客服历史对话，生成下一句合适的客服回复。"
DPO_INSTRUCTION = "你是电商平台客服偏好优化数据构建器，请偏向更准确、相关、礼貌且可执行的回复。"

SPLIT_TARGETS = {
    "train": 3000,
    "validation": 1000,
    "test": 1000,
}


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["selection_rank"] = line_no
            rows.append(row)
    return rows


def raw_line(label: int, context: Sequence[str], response: str) -> str:
    return "\t".join([str(label), *context, response])


def prompt_from_context(context: Sequence[str]) -> str:
    lines: List[str] = []
    for idx, utt in enumerate(context):
        role = "用户" if idx % 2 == 0 else "客服"
        lines.append(f"{role}：{utt}")
    return "\n".join(lines)


def sft_input_from_context(context: Sequence[str]) -> str:
    prompt = prompt_from_context(context)
    if not prompt.endswith("客服："):
        prompt = f"{prompt}\n客服："
    return f"{SFT_TASK_PREFIX}\n\n{prompt}"


def to_sft(row: Dict[str, object]) -> Dict[str, str]:
    context = list(row["context"])
    return {
        "instruction": SFT_INSTRUCTION,
        "input": sft_input_from_context(context),
        "output": str(row["chosen"]),
    }


def to_dpo(row: Dict[str, object]) -> Dict[str, str]:
    context = list(row["context"])
    return {
        "instruction": DPO_INSTRUCTION,
        "input": prompt_from_context(context),
        "chosen": str(row["chosen"]),
        "rejected": str(row["rejected"]),
    }


def allocate_by_score_tiers(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    """Deterministically stratify by category while spreading score ranks across splits."""
    total_target = sum(SPLIT_TARGETS.values())
    if len(rows) != total_target:
        raise ValueError(f"expected {total_target} rows, got {len(rows)}")

    by_category: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_category[str(row.get("category") or "other")].append(row)

    assignments = {name: [] for name in SPLIT_TARGETS}
    pattern = ["train", "train", "train", "validation", "test"]

    for category in sorted(by_category):
        bucket = sorted(
            by_category[category],
            key=lambda x: (-float(x.get("score", 0.0)), int(x["selection_rank"])),
        )
        for idx, row in enumerate(bucket):
            assignments[pattern[idx % len(pattern)]].append(row)

    # The category buckets in top5000 are multiples of the 60/20/20 ratio today.
    # Keep a small balancing pass so the script remains exact if the source changes.
    for split_rows in assignments.values():
        split_rows.sort(key=lambda x: int(x["selection_rank"]))

    while True:
        oversized = [s for s, target in SPLIT_TARGETS.items() if len(assignments[s]) > target]
        undersized = [s for s, target in SPLIT_TARGETS.items() if len(assignments[s]) < target]
        if not oversized and not undersized:
            break
        if not oversized or not undersized:
            raise RuntimeError("could not balance split sizes")

        src = max(oversized, key=lambda s: len(assignments[s]) - SPLIT_TARGETS[s])
        dst = max(undersized, key=lambda s: SPLIT_TARGETS[s] - len(assignments[s]))
        moved = assignments[src].pop()
        assignments[dst].append(moved)
        assignments[dst].sort(key=lambda x: int(x["selection_rank"]))

    return assignments


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def build(preview_path: Path, out_dir: Path) -> Dict[str, object]:
    rows = read_jsonl(preview_path)
    splits = allocate_by_score_tiers(rows)

    summary: Dict[str, object] = {
        "source_preview": str(preview_path),
        "output_dir": str(out_dir),
        "split_targets": SPLIT_TARGETS,
        "total_samples": len(rows),
        "split_stats": {},
        "notes": {
            "unit": "Each sample is one context with one chosen response and one rejected response.",
            "sft": "Use sft/*.json; each split contains only chosen responses.",
            "dpo": "Use dpo/*.json; each split contains chosen/rejected preference pairs.",
            "raw_pairs": "Use raw_pairs/*.txt when a downstream script expects original label-tab format.",
            "positive_only": "Use positive_only/*.txt when a downstream script expects original format but only label=1 rows.",
        },
    }

    for split, split_rows in splits.items():
        raw_lines: List[str] = []
        positive_lines: List[str] = []
        for row in split_rows:
            context = list(row["context"])
            chosen = str(row["chosen"])
            rejected = str(row["rejected"])
            positive_lines.append(raw_line(1, context, chosen))
            raw_lines.append(raw_line(1, context, chosen))
            raw_lines.append(raw_line(0, context, rejected))

        write_text_lines(out_dir / "raw_pairs" / f"{split}.txt", raw_lines)
        write_text_lines(out_dir / "positive_only" / f"{split}.txt", positive_lines)
        write_json(out_dir / "sft" / f"{split}.json", [to_sft(row) for row in split_rows])
        write_json(out_dir / "dpo" / f"{split}.json", [to_dpo(row) for row in split_rows])
        write_jsonl(out_dir / "preview" / f"{split}.jsonl", split_rows)

        summary["split_stats"][split] = {
            "samples": len(split_rows),
            "raw_pair_lines": len(raw_lines),
            "category_distribution": dict(Counter(str(row.get("category") or "other") for row in split_rows)),
            "avg_score": round(sum(float(row.get("score", 0.0)) for row in split_rows) / len(split_rows), 4),
            "rank_min": min(int(row["selection_rank"]) for row in split_rows),
            "rank_max": max(int(row["selection_rank"]) for row in split_rows),
        }

    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact 3000/1000/1000 SFT and DPO splits from selected top5000.")
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summary = build(args.preview, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
