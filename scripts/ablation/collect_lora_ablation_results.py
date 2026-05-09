import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def read_peak_gpu_memory_mb(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None

    peak = None
    with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("memory_used_mb", "")
            try:
                memory = int(float(raw.strip()))
            except ValueError:
                continue
            peak = memory if peak is None else max(peak, memory)
    return peak


def collect_one(exp_dir: Path) -> Dict[str, Any]:
    meta = load_json(exp_dir / "ablation_meta.json")
    run_summary = load_json(exp_dir / "run_summary.json")
    metrics = load_json(exp_dir / "metrics.json")
    eval_summary = load_json(exp_dir / "eval_test" / "summary.json")
    time_summary = load_json(exp_dir / "time_summary.json")

    final_metrics = run_summary.get("final_metrics") or {}

    train_loss = (
        metrics.get("train_loss")
        or final_metrics.get("train_loss")
        or run_summary.get("train_loss")
    )
    eval_loss = (
        metrics.get("eval_loss")
        or final_metrics.get("eval_loss")
        or run_summary.get("eval_loss")
    )
    train_runtime = (
        metrics.get("train_runtime")
        or final_metrics.get("train_runtime")
        or time_summary.get("train_wall_time_seconds")
    )

    exact_match = eval_summary.get("exact_match")
    char_f1 = eval_summary.get("char_f1")
    rouge_l_char = eval_summary.get("rouge_l_char")

    peak_memory = read_peak_gpu_memory_mb(exp_dir / "gpu_usage.csv")
    qlora_baseline_peak = None

    row = {
        "experiment": meta.get("name") or exp_dir.name,
        "description": meta.get("description", ""),
        "train_mode": meta.get("train_mode") or run_summary.get("train_mode"),
        "lora_r": meta.get("lora_r") or run_summary.get("lora_r"),
        "lora_alpha": meta.get("lora_alpha") or run_summary.get("lora_alpha"),
        "target_modules": meta.get("target_modules") or ",".join(run_summary.get("target_modules", [])),
        "train_loss": as_float(train_loss),
        "eval_loss": as_float(eval_loss),
        "task_accuracy_proxy_exact_match": as_float(exact_match),
        "char_f1": as_float(char_f1),
        "rouge_l_char": as_float(rouge_l_char),
        "train_runtime_seconds": as_float(train_runtime),
        "peak_gpu_memory_mb": peak_memory,
        "checkpoint": run_summary.get("latest_checkpoint_dir", ""),
        "output_dir": str(exp_dir),
    }
    row["_qlora_baseline_peak"] = qlora_baseline_peak
    return row


def add_derived_metrics(rows: List[Dict[str, Any]]) -> None:
    baseline = next((r for r in rows if r.get("experiment") == "rank16_qv_qlora"), None)
    baseline_peak = as_float(baseline.get("peak_gpu_memory_mb")) if baseline else None
    baseline_f1 = as_float(baseline.get("char_f1")) if baseline else None

    for row in rows:
        peak = as_float(row.get("peak_gpu_memory_mb"))
        f1 = as_float(row.get("char_f1"))

        if baseline_peak and peak is not None:
            row["peak_memory_vs_rank16_qv_qlora_pct"] = (peak - baseline_peak) / baseline_peak * 100
        else:
            row["peak_memory_vs_rank16_qv_qlora_pct"] = None

        if baseline_f1 and f1 is not None:
            row["char_f1_vs_rank16_qv_qlora_pct"] = (f1 - baseline_f1) / baseline_f1 * 100
        else:
            row["char_f1_vs_rank16_qv_qlora_pct"] = None

    lora_row = next((r for r in rows if r.get("experiment") == "rank16_qv_lora"), None)
    qlora_row = next((r for r in rows if r.get("experiment") == "rank16_qv_qlora"), None)
    if lora_row and qlora_row:
        lora_peak = as_float(lora_row.get("peak_gpu_memory_mb"))
        qlora_peak = as_float(qlora_row.get("peak_gpu_memory_mb"))
        if lora_peak and qlora_peak:
            qlora_row["memory_saving_vs_lora_pct"] = (lora_peak - qlora_peak) / lora_peak * 100
            lora_row["memory_saving_vs_lora_pct"] = 0.0


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        "experiment",
        "description",
        "train_mode",
        "lora_r",
        "lora_alpha",
        "target_modules",
        "train_loss",
        "eval_loss",
        "task_accuracy_proxy_exact_match",
        "char_f1",
        "rouge_l_char",
        "train_runtime_seconds",
        "peak_gpu_memory_mb",
        "peak_memory_vs_rank16_qv_qlora_pct",
        "char_f1_vs_rank16_qv_qlora_pct",
        "memory_saving_vs_lora_pct",
        "checkpoint",
        "output_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        ("experiment", "Experiment"),
        ("train_mode", "Mode"),
        ("lora_r", "Rank"),
        ("target_modules", "Target Modules"),
        ("train_loss", "Train Loss"),
        ("eval_loss", "Eval Loss"),
        ("task_accuracy_proxy_exact_match", "Exact Match"),
        ("char_f1", "Char F1"),
        ("train_runtime_seconds", "Train Sec"),
        ("peak_gpu_memory_mb", "Peak Mem MB"),
        ("peak_memory_vs_rank16_qv_qlora_pct", "Mem vs r16 qv"),
        ("char_f1_vs_rank16_qv_qlora_pct", "F1 vs r16 qv"),
        ("memory_saving_vs_lora_pct", "QLoRA Saving"),
    ]

    lines = []
    lines.append("# LoRA Ablation Summary")
    lines.append("")
    lines.append("Fixed settings: same dataset, same short-answer system prompt, completion ending with EOS, deterministic evaluation.")
    lines.append("")
    lines.append("| " + " | ".join(title for _, title in columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if key.endswith("_pct") and value is not None:
                values.append(f"{float(value):.2f}%")
            else:
                values.append(fmt(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    lines.append("Interview note:")
    lines.append("")
    lines.append("- Use `rank16_qv_qlora` as the main baseline.")
    lines.append("- Rank ablation compares `rank4_qv_qlora`, `rank16_qv_qlora`, and `rank64_qv_qlora`.")
    lines.append("- Target-module ablation compares `rank16_qv_qlora` with `rank16_all_qlora`.")
    lines.append("- QLoRA-vs-LoRA compares `rank16_qv_qlora` with `rank16_qv_lora`.")
    lines.append("- `Exact Match` is a strict task-accuracy proxy for this dataset; use `Char F1` and manual badcase review as softer quality signals.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LoRA ablation metrics.")
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    ablation_root = Path(args.ablation_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for exp_dir in sorted(p for p in ablation_root.iterdir() if p.is_dir() and p.name != "reports"):
        if not (exp_dir / "ablation_meta.json").exists():
            continue
        rows.append(collect_one(exp_dir))

    add_derived_metrics(rows)

    csv_path = output_dir / "lora_ablation_summary.csv"
    md_path = output_dir / "lora_ablation_summary.md"
    json_path = output_dir / "lora_ablation_summary.json"

    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "num_experiments": len(rows),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "json_path": str(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
