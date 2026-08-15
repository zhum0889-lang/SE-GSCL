"""Run a lightweight local P1 capacity sweep while cloud jobs continue."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _int_values(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _float_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=str(ROOT / "data" / "CWRU"))
    parser.add_argument(
        "--text-cache",
        default=str(
            ROOT / "results" / "semantic_cache" / "qwen25_05b_bearing4_v1"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "local_p1_sweep"),
    )
    parser.add_argument("--dataset", default="cwru4")
    parser.add_argument("--domains", default="0,1")
    parser.add_argument("--branch-dims", default="16,32,64")
    parser.add_argument("--semantic-dims", default="128,256")
    parser.add_argument("--num-tokens", default="32")
    parser.add_argument("--learning-rates", default="0.001")
    parser.add_argument("--encoder-kernels", default="7,15,31")
    parser.add_argument("--initial-epochs", type=int, default=5)
    parser.add_argument("--continual-epochs", type=int, default=5)
    parser.add_argument("--max-windows-per-file", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-per-class", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _score(report: dict[str, object]) -> tuple[float, float, float]:
    summary = report["sequence_summary"]["balanced_accuracy"]
    return (
        float(summary["final_average_accuracy"]),
        float(summary["average_incremental_accuracy"]),
        float(summary["average_forgetting"]),
    )


def main() -> int:
    args = parse_args()
    branch_dims = _int_values(args.branch_dims)
    semantic_dims = _int_values(args.semantic_dims)
    token_counts = _int_values(args.num_tokens)
    learning_rates = _float_values(args.learning_rates)
    if not all((branch_dims, semantic_dims, token_counts, learning_rates)):
        raise ValueError("Sweep value lists must not be empty.")

    data_root = Path(args.data_root)
    text_cache = Path(args.text_cache)
    if args.execute and not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")
    if args.execute and not (text_cache / "text_embeddings.npz").is_file():
        raise FileNotFoundError(f"Text cache not found: {text_cache}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    commands: list[str] = []
    combinations = itertools.product(
        branch_dims,
        semantic_dims,
        token_counts,
        learning_rates,
    )
    for branch_dim, semantic_dim, num_tokens, learning_rate in combinations:
        run_id = (
            f"b{branch_dim}_d{semantic_dim}_t{num_tokens}_"
            f"lr{learning_rate:g}"
        )
        output_dir = output_root / run_id
        report_path = output_dir / "p1_report.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "train_p1_global.py"),
            "--dataset", args.dataset,
            "--data-root", str(data_root),
            "--text-cache", str(text_cache),
            "--output-dir", str(output_dir),
            "--domains", args.domains,
            "--strategy", "full",
            "--window-size", "1024",
            "--step-size", "1024",
            "--max-windows-per-file", str(args.max_windows_per_file),
            "--semantic-dim", str(semantic_dim),
            "--num-tokens", str(num_tokens),
            "--branch-dim", str(branch_dim),
            "--encoder-kernels", args.encoder_kernels,
            "--batch-size", str(args.batch_size),
            "--initial-epochs", str(args.initial_epochs),
            "--continual-epochs", str(args.continual_epochs),
            "--replay-per-class", str(args.replay_per_class),
            "--learning-rate", str(learning_rate),
            "--lambda-cc", "0.1",
            "--lambda-dec", "0.001",
            "--lambda-rel", "1.0",
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        commands.append(subprocess.list2cmdline(command))
        if args.execute and (args.force or not report_path.is_file()):
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"RUN {run_id}")
            with (output_dir / "run.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            final, incremental, forgetting = _score(report)
            rows.append(
                {
                    "run_id": run_id,
                    "branch_dim": branch_dim,
                    "semantic_dim": semantic_dim,
                    "num_tokens": num_tokens,
                    "learning_rate": learning_rate,
                    "final_balanced_accuracy": final,
                    "incremental_balanced_accuracy": incremental,
                    "average_forgetting": forgetting,
                    "continual_trainable_parameters": report["training_config"][
                        "continual_trainable_parameters"
                    ],
                }
            )

    (output_root / "commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    if rows:
        rows.sort(
            key=lambda row: (
                -float(row["final_balanced_accuracy"]),
                float(row["average_forgetting"]),
            )
        )
        with (output_root / "summary.csv").open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(rows, indent=2))
    print(
        f"{'Executed' if args.execute else 'Planned'} {len(commands)} local runs "
        f"under {output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
