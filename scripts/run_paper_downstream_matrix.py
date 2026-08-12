"""Run paper-level P2/P3 ablations from completed P1 full checkpoints."""

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
    parser.add_argument("--p1-root", default=str(ROOT / "results" / "paper_matrix" / "p1"))
    parser.add_argument("--global-text-cache")
    parser.add_argument("--symptom-text-cache")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "results" / "paper_matrix"),
    )
    parser.add_argument("--seeds", help="Comma-separated override.")
    parser.add_argument("--p2-jobs", help="Comma-separated P2 job IDs.")
    parser.add_argument("--p3-jobs", help="Comma-separated P3 job IDs.")
    parser.add_argument("--stage", choices=("p2", "p3", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--prompt-epochs", type=int, default=10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _csv_values(value: str | None) -> set[str] | None:
    if not value:
        return None
    rows = {item.strip() for item in value.split(",") if item.strip()}
    return rows or None


def _select_jobs(
    rows: list[dict[str, Any]],
    requested: str | None,
) -> list[dict[str, Any]]:
    wanted = _csv_values(requested)
    if wanted is None:
        return [dict(row) for row in rows]
    known = {str(row["id"]) for row in rows}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(f"Unknown downstream jobs: {unknown}")
    return [dict(row) for row in rows if str(row["id"]) in wanted]


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(value) for value in command)


def _dataset_root_candidates(dataset: str) -> list[Path]:
    if dataset.startswith("cwru"):
        names = ("CWRU", "cwru")
    elif dataset.startswith("multidomain"):
        names = ("MultiDomainBearing", "multidomainbearing")
    else:
        names = ("HUSTbearing", "HUSTBearing", "hustbearing")
    bases = (ROOT / "data", ROOT.parent / "data")
    return [base / name for base in bases for name in names]


def _add_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def build_p2_command(
    *,
    python_bin: str,
    data_root: Path,
    p1_dir: Path,
    global_cache: Path,
    symptom_cache: Path,
    output_dir: Path,
    job: dict[str, Any],
    device: str,
) -> list[str]:
    command = [
        python_bin,
        str(ROOT / "scripts" / "evaluate_p2_local.py"),
        "--data-root",
        str(data_root),
        "--global-text-cache",
        str(global_cache),
        "--symptom-text-cache",
        str(symptom_cache),
        "--p1-dir",
        str(p1_dir),
        "--output-dir",
        str(output_dir),
        "--adapter-epochs",
        str(job.get("adapter_epochs", 15)),
        "--batch-size",
        "64",
        "--learning-rate",
        "0.001",
        "--top-tokens",
        "4",
        "--local-temperature",
        "0.1",
        "--local-weight",
        str(job.get("local_weight", 0.3)),
        "--physics-weight",
        "1.0",
        "--anchor-weight",
        "0.1",
        "--residual-scale",
        "0.2",
        "--ranking-weight",
        "0.5",
        "--early-stopping-patience",
        "3",
        "--device",
        device,
    ]
    _add_flag(command, bool(job.get("physics_guided")), "--physics-guided")
    _add_flag(command, bool(job.get("semantic_guard")), "--semantic-guard")
    _add_flag(command, bool(job.get("adaptive_fusion")), "--adaptive-fusion")
    return command


def build_p3_command(
    *,
    python_bin: str,
    job: dict[str, Any],
    p2_dir: Path,
    p31_dir: Path,
    model: Path,
    output_dir: Path,
    device: str,
    dtype: str,
    prompt_epochs: int,
    seed: int,
    local_files_only: bool,
) -> list[str]:
    job_type = str(job["type"])
    common = [
        "--model",
        str(model),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--dtype",
        dtype,
    ]
    if job_type == "structured_text":
        command = [
            python_bin,
            str(ROOT / "scripts" / "run_p3_frozen_qwen.py"),
            "--p2-dir",
            str(p2_dir),
            *common,
            "--max-samples",
            "0",
            "--batch-size",
            "4",
        ]
    elif job_type == "continuous_prompt":
        command = [
            python_bin,
            str(ROOT / "scripts" / "train_p31_continuous_prompt.py"),
            "--p2-dir",
            str(p2_dir),
            *common,
            "--epochs",
            str(prompt_epochs),
            "--batch-size",
            "1",
            "--gradient-accumulation",
            "8",
            "--auxiliary-weight",
            str(job.get("auxiliary_weight", 0.5)),
            "--num-prompt-tokens",
            "4",
            "--adapter-rank",
            "64",
            "--seed",
            str(seed),
            "--max-train-samples",
            "0",
            "--max-validation-samples",
            "0",
            "--max-test-samples",
            "0",
        ]
    elif job_type == "continuous_explanation":
        command = [
            python_bin,
            str(ROOT / "scripts" / "run_p32_continuous_explanations.py"),
            "--p2-dir",
            str(p2_dir),
            "--p31-dir",
            str(p31_dir),
            *common,
            "--max-samples",
            "0",
            "--batch-size",
            "4",
        ]
        _add_flag(
            command,
            bool(job.get("unlock_diagnosis")),
            "--unlock-diagnosis",
        )
    else:
        raise ValueError(f"Unsupported P3 job type: {job_type}")
    _add_flag(command, local_files_only, "--local-files-only")
    return command


def _run_job(
    command: list[str],
    output_dir: Path,
    report_name: str,
    job_record: dict[str, Any],
    *,
    execute: bool,
    force: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "paper_job.json").write_text(
        json.dumps(job_record | {"command": _command_text(command)}, indent=2),
        encoding="utf-8",
    )
    if not execute:
        return
    report = output_dir / report_name
    if report.is_file() and not force:
        print(f"SKIP complete: {report}")
        return
    with (output_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if not report.is_file():
        raise RuntimeError(
            "Downstream subprocess exited without producing its report: "
            f"{report}. Inspect {output_dir / 'run.log'}."
        )


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.matrix)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if args.dataset not in matrix["datasets"]:
        raise ValueError(f"Dataset {args.dataset!r} is not in the matrix.")
    dataset_config = dict(matrix["datasets"][args.dataset])
    seeds = [
        int(value)
        for value in (
            args.seeds.split(",") if args.seeds else matrix["seeds"]
        )
    ]
    p2_jobs = _select_jobs(matrix["p2_jobs"], args.p2_jobs)
    p3_jobs = _select_jobs(matrix["p3_jobs"], args.p3_jobs)
    data_root = Path(args.data_root)
    p1_root = Path(args.p1_root) / args.dataset
    global_cache = Path(
        args.global_text_cache or dataset_config["global_text_cache"]
    )
    symptom_cache = Path(
        args.symptom_text_cache or dataset_config["symptom_text_cache"]
    )
    model = Path(args.model)
    output_root = Path(args.output_root)

    if args.execute:
        if not data_root.exists():
            existing = [
                path
                for path in _dataset_root_candidates(args.dataset)
                if path.exists()
            ]
            hint = (
                "\nDetected candidate(s):\n  "
                + "\n  ".join(str(path) for path in existing)
                if existing
                else ""
            )
            raise FileNotFoundError(
                f"Dataset root not found: {data_root}{hint}\n"
                "Pass the correct directory with --data-root."
            )
        required = [
            global_cache / "text_embeddings.npz",
            symptom_cache / "symptom_embeddings.npz",
            model / "config.json",
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing downstream inputs: {missing}")
        if args.dataset.startswith("multidomain"):
            physics_jobs = [
                str(job["id"])
                for job in p2_jobs
                if bool(job.get("physics_guided"))
            ]
            if physics_jobs:
                raise ValueError(
                    "MultiDomainBearing contains multiple bearing families, but "
                    "validated kinematic ratios have not yet been registered for "
                    "every family. Refusing physics-guided P2 jobs "
                    f"{physics_jobs} rather than silently using an unrelated "
                    "bearing prior. Run the P1 continual-learning matrix first, "
                    "or register verified per-bearing kinematics before P2/P3."
                )

    command_rows: list[str] = []
    for seed in seeds:
        p1_dir = p1_root / f"seed_{seed}" / "se_gscl_full"
        if args.execute and not (p1_dir / "p1_report.json").is_file():
            raise FileNotFoundError(f"Missing full P1 checkpoint: {p1_dir}")
        full_p2_dir = output_root / "p2" / args.dataset / f"seed_{seed}" / "se_gscl_full"
        if args.stage in {"p2", "all"}:
            for job in p2_jobs:
                output_dir = output_root / "p2" / args.dataset / f"seed_{seed}" / str(job["id"])
                command = build_p2_command(
                    python_bin=sys.executable,
                    data_root=data_root,
                    p1_dir=p1_dir,
                    global_cache=global_cache,
                    symptom_cache=symptom_cache,
                    output_dir=output_dir,
                    job=job,
                    device=args.device,
                )
                command_rows.append(_command_text(command))
                print(f"P2 {args.dataset} seed={seed} job={job['id']}")
                _run_job(
                    command,
                    output_dir,
                    "p2_report.json",
                    {"dataset": args.dataset, "seed": seed, "job": job},
                    execute=args.execute,
                    force=args.force,
                )

        if args.stage in {"p3", "all"}:
            if args.execute and not (full_p2_dir / "p2_report.json").is_file():
                raise FileNotFoundError(f"Missing full P2 output: {full_p2_dir}")
            full_p31_dir = (
                output_root
                / "p3"
                / args.dataset
                / f"seed_{seed}"
                / "continuous_full"
            )
            for job in p3_jobs:
                output_dir = output_root / "p3" / args.dataset / f"seed_{seed}" / str(job["id"])
                command = build_p3_command(
                    python_bin=sys.executable,
                    job=job,
                    p2_dir=full_p2_dir,
                    p31_dir=full_p31_dir,
                    model=model,
                    output_dir=output_dir,
                    device=args.device,
                    dtype=args.dtype,
                    prompt_epochs=args.prompt_epochs,
                    seed=seed,
                    local_files_only=args.local_files_only,
                )
                command_rows.append(_command_text(command))
                report_name = (
                    "p3_report.json"
                    if job["type"] == "structured_text"
                    else (
                        "p31_report.json"
                        if job["type"] == "continuous_prompt"
                        else "p32_report.json"
                    )
                )
                print(f"P3 {args.dataset} seed={seed} job={job['id']}")
                _run_job(
                    command,
                    output_dir,
                    report_name,
                    {"dataset": args.dataset, "seed": seed, "job": job},
                    execute=args.execute,
                    force=args.force,
                )

    command_file = output_root / f"{args.dataset}_downstream_commands.txt"
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text(
        "\n".join(command_rows) + "\n",
        encoding="utf-8",
    )
    print(
        f"{'Executed' if args.execute else 'Planned'} "
        f"{len(command_rows)} downstream jobs; commands={command_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
