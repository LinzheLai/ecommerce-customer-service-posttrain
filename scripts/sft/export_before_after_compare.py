import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export before/after customer-service inference comparison as Markdown."
    )
    parser.add_argument("--base_json", type=str, required=True, help="Base model inference JSON.")
    parser.add_argument("--sft_json", type=str, required=True, help="SFT model inference JSON.")
    parser.add_argument("--output_md", type=str, required=True, help="Output Markdown path.")
    parser.add_argument(
        "--max_rows",
        type=int,
        default=10,
        help="Maximum number of rows to export.",
    )
    parser.add_argument(
        "--keyword",
        type=str,
        default=None,
        help="Optional keyword filter, for example: refund, return, policy.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported inference result format: {path}")


def item_key(item: Dict[str, Any]) -> Tuple[str, str]:
    source_index = item.get("source_index")
    if isinstance(source_index, int):
        return ("source_index", str(source_index))
    prompt = str(item.get("prompt", "")).strip()
    return ("prompt", prompt)


def normalize_cell(text: Any, limit: int = 120) -> str:
    value = "" if text is None else str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = " ".join(value.split())
    value = value.replace("|", "\\|")
    if len(value) > limit:
        return value[: limit - 3] + "..."
    return value


def contains_keyword(item: Dict[str, Any], keyword: str) -> bool:
    haystack = " ".join(
        str(item.get(name, "")) for name in ("prompt", "prediction", "reference")
    ).lower()
    return keyword.lower() in haystack


def align_rows(
    base_results: Iterable[Dict[str, Any]],
    sft_results: Iterable[Dict[str, Any]],
    keyword: str | None,
    max_rows: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    base_map = {item_key(item): item for item in base_results}
    rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for sft_item in sft_results:
        key = item_key(sft_item)
        base_item = base_map.get(key)
        if not base_item:
            continue
        if keyword and not (contains_keyword(base_item, keyword) or contains_keyword(sft_item, keyword)):
            continue
        rows.append((base_item, sft_item))
        if len(rows) >= max_rows:
            break
    return rows


def write_markdown(path: Path, rows: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SFT Before/After Comparison",
        "",
        "| # | Query | Reference | Before SFT | After SFT |",
        "|---:|---|---|---|---|",
    ]
    for idx, (base_item, sft_item) in enumerate(rows, start=1):
        query = normalize_cell(sft_item.get("prompt") or base_item.get("prompt"))
        reference = normalize_cell(sft_item.get("reference") or base_item.get("reference"))
        before = normalize_cell(base_item.get("prediction"))
        after = normalize_cell(sft_item.get("prediction"))
        lines.append(f"| {idx} | {query} | {reference} | {before} | {after} |")
    if not rows:
        lines.append("| - | No aligned rows found. | - | - | - |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_results = load_results(Path(args.base_json))
    sft_results = load_results(Path(args.sft_json))
    rows = align_rows(base_results, sft_results, args.keyword, args.max_rows)
    write_markdown(Path(args.output_md), rows)
    print(f"rows: {len(rows)}")
    print(f"output_md: {args.output_md}")


if __name__ == "__main__":
    main()
