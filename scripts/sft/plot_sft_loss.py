import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SFT training loss from Hugging Face trainer_state.json."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default="outputs/sft/qwen3_8b_rank16_zero2",
        help="SFT output directory, or a checkpoint directory that contains trainer_state.json.",
    )
    parser.add_argument(
        "--trainer_state",
        type=str,
        default=None,
        help="Explicit trainer_state.json path. Overrides --run_dir auto discovery.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Output PNG path. Defaults to <run_dir>/loss_curve.png.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Optional CSV path for exported loss points.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="SFT Training Loss",
        help="Figure title.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
        help="Moving average window for train loss. Use 1 to disable smoothing.",
    )
    return parser.parse_args()


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_trainer_state(run_dir: Path) -> Path:
    direct = run_dir / "trainer_state.json"
    if direct.exists():
        return direct

    checkpoints = [
        path
        for path in run_dir.glob("checkpoint-*")
        if path.is_dir() and (path / "trainer_state.json").exists()
    ]
    if not checkpoints:
        raise FileNotFoundError(
            f"No trainer_state.json found under {run_dir}. "
            "Pass --trainer_state explicitly if it lives elsewhere."
        )
    return max(checkpoints, key=checkpoint_step) / "trainer_state.json"


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def read_history(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    history = payload.get("log_history", [])
    if not isinstance(history, list):
        raise ValueError(f"Invalid log_history in {path}")
    return [item for item in history if isinstance(item, dict)]


def export_csv(
    path: Path,
    train_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
) -> None:
    eval_by_step = {int(row["step"]): row for row in eval_rows if "step" in row}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "epoch",
                "train_loss",
                "eval_loss",
                "learning_rate",
                "mean_token_accuracy",
            ],
        )
        writer.writeheader()
        for row in train_rows:
            step = int(row["step"])
            eval_row = eval_by_step.get(step, {})
            writer.writerow(
                {
                    "step": step,
                    "epoch": row.get("epoch", ""),
                    "train_loss": row.get("loss", ""),
                    "eval_loss": eval_row.get("eval_loss", ""),
                    "learning_rate": row.get("learning_rate", ""),
                    "mean_token_accuracy": row.get("mean_token_accuracy", ""),
                }
            )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    trainer_state = Path(args.trainer_state) if args.trainer_state else find_trainer_state(run_dir)
    output_png = Path(args.output_png) if args.output_png else run_dir / "loss_curve.png"
    output_csv: Optional[Path] = Path(args.output_csv) if args.output_csv else None

    history = read_history(trainer_state)
    train_rows = [row for row in history if "step" in row and "loss" in row]
    eval_rows = [row for row in history if "step" in row and "eval_loss" in row]
    if not train_rows:
        raise ValueError(f"No training loss points found in {trainer_state}")

    steps = [int(row["step"]) for row in train_rows]
    epochs = [float(row.get("epoch", 0.0)) for row in train_rows]
    train_loss = [float(row["loss"]) for row in train_rows]
    plotted_loss = moving_average(train_loss, args.smooth_window)

    import matplotlib.pyplot as plt

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=160)
    label = "train loss"
    if args.smooth_window > 1:
        label += f" (moving avg {args.smooth_window})"
    ax.plot(steps, plotted_loss, color="#2563eb", linewidth=2.0, label=label)

    if args.smooth_window > 1:
        ax.plot(steps, train_loss, color="#93c5fd", linewidth=1.0, alpha=0.55, label="raw train loss")

    if eval_rows:
        eval_steps = [int(row["step"]) for row in eval_rows]
        eval_loss = [float(row["eval_loss"]) for row in eval_rows]
        ax.scatter(eval_steps, eval_loss, color="#dc2626", s=30, label="eval loss", zorder=3)

    ax.set_title(args.title)
    ax.set_xlabel("global step")
    ax.set_ylabel("cross entropy loss")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    secondary = ax.secondary_xaxis(
        "top",
        functions=(
            lambda step: step / max(steps) * max(epochs) if max(steps) else step,
            lambda epoch: epoch / max(epochs) * max(steps) if max(epochs) else epoch,
        ),
    )
    secondary.set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    if output_csv:
        export_csv(output_csv, train_rows, eval_rows)

    print(f"trainer_state: {trainer_state}")
    print(f"loss_png: {output_png}")
    if output_csv:
        print(f"loss_csv: {output_csv}")
    print(f"points: {len(train_rows)}")
    print(f"first_loss: {train_loss[0]:.6f}")
    print(f"last_loss: {train_loss[-1]:.6f}")


if __name__ == "__main__":
    main()
