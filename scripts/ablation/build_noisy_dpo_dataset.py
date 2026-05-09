import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def save_json_or_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create noisy DPO preference data by swapping chosen/rejected in a fixed ratio."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--noise_ratio", type=float, required=True, help="0.1 means swap 10% pairs.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 <= args.noise_ratio <= 1.0:
        raise ValueError("--noise_ratio must be in [0, 1]")

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    rows = load_json_or_jsonl(input_path)

    rng = random.Random(args.seed)
    n = len(rows)
    noisy_n = int(round(n * args.noise_ratio))
    noisy_indices = set(rng.sample(range(n), noisy_n)) if noisy_n > 0 else set()

    output_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        item = dict(row)
        item["_noise_ratio"] = args.noise_ratio
        item["_noise_swapped"] = i in noisy_indices
        item["_source_index"] = i
        if i in noisy_indices:
            item["chosen"], item["rejected"] = row.get("rejected", ""), row.get("chosen", "")
        output_rows.append(item)

    save_json_or_jsonl(output_rows, output_path)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_samples": n,
        "noise_ratio": args.noise_ratio,
        "num_swapped": noisy_n,
        "seed": args.seed,
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
