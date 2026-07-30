"""Run the paper-level P1 continual-learning baseline and ablation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matrix",
        default=str(ROOT / "configs" / "experiments" / "paper_matrix.json"),
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--text-cache")
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "paper_matrix" / "p1"),
    )
    parser.add_argument("--seeds", help="Comma-separated override.")
    parser.add_argument("--jobs", help="Comma-separated job IDs.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_matrix(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"seeds", "datasets", "p1_common", "p1_jobs"}
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Paper matrix is missing keys: {missing}")
    job_ids = [str(row["id"]) for row in value["p1_jobs"]]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("P1 job IDs must be unique.")
    return value


def selected_seeds(matrix: dict[str, Any], override: str | None) -> list[int]:
    values = matrix["seeds"] if not override else override.split(",")
    seeds = [int(value) for value in values]
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Seeds must be a non-empty unique list.")
    return seeds


def selected_jobs(
    matrix: dict[str, Any],
    requested: str | None,
) -> list[dict[str, Any]]:
    jobs = [dict(row) for row in matrix["p1_jobs"]]
    if not requested:
        return jobs
    wanted = {value.strip() for value in requested.split(",") if value.strip()}
    known = {str(row["id"]) for row in jobs}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown P1 jobs: {unknown}")
    return [row for row in jobs if str(row["id"]) in wanted]


def resolve_job_settings(
    common: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    resolved = dict(common)
    resolved.update(dict(job.get("overrides", {})))
    return resolved


def build_train_command(
    *,
    python_bin: str,
    dataset: str,
    data_root: Path,
    text_cache: Path,
    output_dir: Path,
    dataset_config: dict[str, Any],
    job: dict[str, Any],
    common: dict[str, Any],
    seed: int,
    device: str,
) -> list[str]:
    settings = resolve_job_settings(common, job)
    domains = ",".join(str(value) for value in dataset_config["domain_order"])
    return [
        python_bin,
        str(ROOT / "scripts" / "train_p1_global.py"),
        "--dataset",
        dataset,
        "--data-root",
        str(data_root),
        "--text-cache",
        str(text_cache),
        "--output-dir",
        str(output_dir),
        "--domains",
        domains,
        "--strategy",
        str(job["strategy"]),
        "--window-size",
        str(dataset_config["window_size"]),
        "--step-size",
        str(dataset_config["step_size"]),
        "--max-windows-per-file",
        str(dataset_config["max_windows_per_file"]),
        "--semantic-dim",
        str(settings["semantic_dim"]),
        "--num-tokens",
        str(settings["num_tokens"]),
        "--batch-size",
        str(settings["batch_size"]),
        "--initial-epochs",
        str(settings["initial_epochs"]),
        "--continual-epochs",
        str(settings["continual_epochs"]),
        "--replay-per-class",
        str(dataset_config["replay_per_class"]),
        "--learning-rate",
        str(settings["learning_rate"]),
        "--lambda-cc",
        str(settings["lambda_cc"]),
        "--lambda-dec",
        str(settings["lambda_dec"]),
        "--lambda-rel",
        str(settings["lambda_rel"]),
        "--seed",
        str(seed),
        "--device",
        device,
    ]


def command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.matrix)
    if args.dataset not in matrix["datasets"]:
        raise ValueError(f"Dataset {args.dataset!r} is not in the paper matrix.")
    dataset_config = dict(matrix["datasets"][args.dataset])
    data_root = Path(args.data_root)
    text_cache = Path(
        args.text_cache or str(dataset_config["global_text_cache"])
    )
    if args.execute:
        if not data_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {data_root}")
        if not (text_cache / "text_embeddings.npz").is_file():
            raise FileNotFoundError(
                "Frozen text cache is missing: "
                f"{text_cache / 'text_embeddings.npz'}"
            )

    seeds = selected_seeds(matrix, args.seeds)
    jobs = selected_jobs(matrix, args.jobs)
    dataset_root = Path(args.output_root) / args.dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    resolved = {
        "matrix": str(Path(args.matrix).resolve()),
        "matrix_version": matrix.get("version"),
        "dataset": args.dataset,
        "dataset_config": dataset_config,
        "data_root": str(data_root.resolve()),
        "text_cache": str(text_cache.resolve()),
        "seeds": seeds,
        "jobs": jobs,
        "device": args.device,
    }
    (dataset_root / "resolved_matrix.json").write_text(
        json.dumps(resolved, indent=2),
        encoding="utf-8",
    )

    commands: list[str] = []
    for seed in seeds:
        for job in jobs:
            output_dir = (
                dataset_root / f"seed_{seed}" / str(job["id"])
            )
            report_path = output_dir / "p1_report.json"
            command = build_train_command(
                python_bin=sys.executable,
                dataset=args.dataset,
                data_root=data_root,
                text_cache=text_cache,
                output_dir=output_dir,
                dataset_config=dataset_config,
                job=job,
                common=dict(matrix["p1_common"]),
                seed=seed,
                device=args.device,
            )
            commands.append(command_text(command))
            if not args.execute:
                continue
            if report_path.is_file() and not args.force:
                print(f"SKIP complete: {report_path}")
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            job_record = {
                "dataset": args.dataset,
                "seed": seed,
                "job": job,
                "resolved_settings": resolve_job_settings(
                    dict(matrix["p1_common"]),
                    job,
                ),
                "command": command_text(command),
            }
            (output_dir / "paper_job.json").write_text(
                json.dumps(job_record, indent=2),
                encoding="utf-8",
            )
            print(f"RUN {args.dataset} seed={seed} job={job['id']}")
            with (output_dir / "run.log").open(
                "w",
                encoding="utf-8",
            ) as log:
                subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
    (dataset_root / "commands.txt").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    print(
        f"{'Executed' if args.execute else 'Planned'} "
        f"{len(commands)} P1 jobs under {dataset_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
