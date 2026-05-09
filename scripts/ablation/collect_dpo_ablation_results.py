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
    summary = load_json(exp_dir / "run_summary.json")
    metrics = load_json(exp_dir / "metrics.json")
    task_eval = load_json(exp_dir / "eval_task" / "summary.json")
    pref_eval = load_json(exp_dir / "eval_preference" / "preference_summary.json")
    time_summary = load_json(exp_dir / "time_summary.json")
    final = summary.get("final_metrics") or {}

    def metric(name: str):
        return metrics.get(name, final.get(name))

    row = {
        "experiment": meta.get("name") or exp_dir.name,
        "group": meta.get("group", ""),
        "description": meta.get("description", ""),
        "beta": meta.get("beta", summary.get("beta")),
        "noise_ratio": meta.get("noise_ratio", 0.0),
        "train_path": summary.get("train_path", meta.get("train_path", "")),
        "train_loss": as_float(metric("train_loss")),
        "eval_loss": as_float(metric("eval_loss")),
        "eval_rewards_accuracies": as_float(metric("eval_rewards/accuracies")),
        "eval_rewards_margins": as_float(metric("eval_rewards/margins")),
        "preference_accuracy": as_float(pref_eval.get("preference_accuracy")),
        "preference_mean_margin": as_float(pref_eval.get("mean_margin")),
        "preference_mean_normalized_margin": as_float(pref_eval.get("mean_normalized_margin")),
        "task_exact_match": as_float(task_eval.get("exact_match")),
        "task_char_f1": as_float(task_eval.get("char_f1")),
        "task_rouge_l_char": as_float(task_eval.get("rouge_l_char")),
        "train_runtime_seconds": as_float(metric("train_runtime") or time_summary.get("train_wall_time_seconds")),
        "peak_gpu_memory_mb": read_peak_gpu_memory_mb(exp_dir / "gpu_usage.csv"),
        "checkpoint": summary.get("latest_checkpoint_dir", ""),
        "output_dir": str(exp_dir),
    }
    return row


def collect_sft_baseline(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    task_eval = load_json(path / "eval_task" / "summary.json")
    pref_eval = load_json(path / "eval_preference" / "preference_summary.json")
    return {
        "experiment": "sft_only",
        "group": "baseline",
        "description": "SFT-only baseline without DPO",
        "beta": None,
        "noise_ratio": 0.0,
        "train_path": "",
        "train_loss": None,
        "eval_loss": None,
        "eval_rewards_accuracies": None,
        "eval_rewards_margins": None,
        "preference_accuracy": as_float(pref_eval.get("preference_accuracy")),
        "preference_mean_margin": as_float(pref_eval.get("mean_margin")),
        "preference_mean_normalized_margin": as_float(pref_eval.get("mean_normalized_margin")),
        "task_exact_match": as_float(task_eval.get("exact_match")),
        "task_char_f1": as_float(task_eval.get("char_f1")),
        "task_rouge_l_char": as_float(task_eval.get("rouge_l_char")),
        "train_runtime_seconds": None,
        "peak_gpu_memory_mb": None,
        "checkpoint": "",
        "output_dir": str(path),
    }


def add_relative_metrics(rows: List[Dict[str, Any]]) -> None:
    sft = next((r for r in rows if r.get("experiment") == "sft_only"), None)
    clean_beta005 = next((r for r in rows if r.get("experiment") == "beta005_clean"), None)

    for row in rows:
        if sft:
            for key in ("preference_accuracy", "task_exact_match", "task_char_f1"):
                base = as_float(sft.get(key))
                value = as_float(row.get(key))
                row[f"{key}_vs_sft"] = value - base if base is not None and value is not None else None
        if clean_beta005:
            base_pref = as_float(clean_beta005.get("preference_accuracy"))
            value_pref = as_float(row.get("preference_accuracy"))
            row["preference_accuracy_vs_beta005_clean"] = (
                value_pref - base_pref if base_pref is not None and value_pref is not None else None
            )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    columns = [
        "experiment",
        "group",
        "beta",
        "noise_ratio",
        "train_loss",
        "eval_loss",
        "eval_rewards_accuracies",
        "eval_rewards_margins",
        "preference_accuracy",
        "preference_accuracy_vs_sft",
        "preference_accuracy_vs_beta005_clean",
        "preference_mean_normalized_margin",
        "task_exact_match",
        "task_exact_match_vs_sft",
        "task_char_f1",
        "task_char_f1_vs_sft",
        "task_rouge_l_char",
        "train_runtime_seconds",
        "peak_gpu_memory_mb",
        "description",
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
        ("group", "Group"),
        ("beta", "Beta"),
        ("noise_ratio", "Noise"),
        ("eval_loss", "DPO Eval Loss"),
        ("eval_rewards_accuracies", "Reward Acc"),
        ("preference_accuracy", "Pref Win"),
        ("preference_accuracy_vs_sft", "Pref vs SFT"),
        ("task_exact_match", "Task EM"),
        ("task_char_f1", "Task F1"),
        ("train_runtime_seconds", "Train Sec"),
        ("peak_gpu_memory_mb", "Peak Mem MB"),
    ]

    lines = [
        "# DPO Ablation Summary",
        "",
        "Fixed settings: same SFT adapter, same DPO data schema, EOS completion ending, deterministic task evaluation.",
        "",
        "| " + " | ".join(title for _, title in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if key == "noise_ratio" and value is not None:
                values.append(f"{float(value):.0%}")
            elif key.endswith("_vs_sft") and value is not None:
                values.append(f"{float(value):+.4f}")
            else:
                values.append(fmt(value).replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "How to read:",
            "",
            "- `Pref Win` is the log-prob preference win rate: chosen log-prob > rejected log-prob.",
            "- `Reward Acc` comes from TRL DPO eval metrics.",
            "- `Task EM/F1` are generated-answer metrics on the same customer-service test set.",
            "- Compare beta 0.05/0.1/0.3 for alignment strength vs task quality drift.",
            "- Compare noise 10%/30% with the clean beta=0.05 run for preference-data quality sensitivity.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect DPO ablation metrics.")
    parser.add_argument("--ablation_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    root = Path(args.ablation_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    sft = collect_sft_baseline(root / "sft_only")
    if sft:
        rows.append(sft)

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"reports", "data", "sft_only"}):
        if (exp_dir / "ablation_meta.json").exists():
            rows.append(collect_one(exp_dir))

    add_relative_metrics(rows)

    csv_path = out / "dpo_ablation_summary.csv"
    md_path = out / "dpo_ablation_summary.md"
    json_path = out / "dpo_ablation_summary.json"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "num_rows": len(rows),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "json_path": str(json_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
