"""Run selected, resumable model/category benchmarks with local MLflow tracking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import cache
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.dataset_audit import read_config
from industrialguard.mlops.tracking import (
    canonical_json,
    configure_local_mlflow,
    dataset_model_macro_rows,
    experiment_id,
    latest_attempt_state,
    metric_scalars,
    next_attempt_number,
    redact_config,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_CONFIG = ROOT / "configs" / "dataset.yaml"
MODEL_CONFIGS = {
    name: ROOT / "configs" / "models" / f"{name}.yaml"
    for name in ("padim", "patchcore", "efficientad")
}
TRACKED_PACKAGES = (
    "torch",
    "torchvision",
    "anomalib",
    "numpy",
    "scipy",
    "scikit-learn",
    "mlflow",
    "dvc",
    "ruff",
)


def environment_metadata(configs: list[Path], seed: int) -> dict[str, object]:
    """Capture portable versions and hashes, deliberately omitting user/host data."""
    versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in configs
    }
    try:
        import torch

        device = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        }
    except ImportError:
        device = {"cuda_available": False, "cuda_version": None}
    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine_architecture": platform.machine(),
        "packages": versions,
        "seed": seed,
        "git_commit_sha": _git_commit_sha(),
        "source_archive_sha256": os.environ.get(
            "INDUSTRIALGUARD_SOURCE_ARCHIVE_SHA256"
        ),
        "config_sha256": hashes,
        "torch_device_capability": device,
    }


def _git_commit_sha() -> str | None:
    """Return the source revision without recording a local checkout path."""
    declared_version = os.environ.get("INDUSTRIALGUARD_CODE_VERSION")
    if declared_version:
        return declared_version.strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _code_version() -> str | None:
    """Identify a source archive when a Colab ZIP has no Git metadata."""
    commit = _git_commit_sha()
    if commit:
        return commit
    archive_sha = os.environ.get("INDUSTRIALGUARD_SOURCE_ARCHIVE_SHA256")
    return f"archive-sha256:{archive_sha}" if archive_sha else None


def _sha256_file(path: Path) -> str:
    """Hash a potentially large resource without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )


def _safe_text(text: str) -> str:
    """Avoid persisting local Windows paths in captured command output."""
    return re.sub(r"(?i)\b[A-Z]:\\[^\r\n\"']+", "<local-path>", text)


@cache
def _resource_fingerprints(
    dataset: str, model: str, teacher_checkpoint: str | None, penalty_dir: str | None
) -> dict[str, object]:
    """Fingerprint dataset inventory and external EfficientAD inputs without paths."""
    inventory_path = ROOT / "reports" / "dataset_audit" / f"{dataset}_inventory.csv"
    result: dict[str, object] = {
        "dataset_inventory_sha256": hashlib.sha256(
            inventory_path.read_bytes()
        ).hexdigest()
        if inventory_path.is_file()
        else None
    }
    dataset_config = read_config(DATASET_CONFIG)["datasets"][dataset]
    archive = Path(dataset_config["archive"])
    if not archive.is_absolute():
        archive = ROOT / archive
    result["dataset_archive_sha256"] = (
        _sha256_file(archive) if archive.is_file() else None
    )
    source_digest = hashlib.sha256()
    source_roots = (
        ROOT / "src" / "industrialguard" / "data",
        ROOT / "src" / "industrialguard" / "models",
        ROOT / "src" / "industrialguard" / "evaluation",
        ROOT / "src" / "industrialguard" / "mlops",
        ROOT / "scripts" / "train_model.py",
        ROOT / "scripts" / "evaluate_model.py",
        ROOT / "scripts" / "run_benchmark.py",
    )
    source_files: list[Path] = []
    for source in source_roots:
        source_files.extend(source.rglob("*.py") if source.is_dir() else [source])
    for source in sorted(source_files):
        source_digest.update(source.relative_to(ROOT).as_posix().encode())
        source_digest.update(source.read_bytes())
    result["implementation_sha256"] = source_digest.hexdigest()
    package_versions: dict[str, str | None] = {}
    for package in TRACKED_PACKAGES:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    result["environment_sha256"] = hashlib.sha256(
        canonical_json(
            {"python": platform.python_version(), "packages": package_versions}
        ).encode()
    ).hexdigest()
    if model == "efficientad":
        teacher_path = Path(teacher_checkpoint) if teacher_checkpoint else None
        result["teacher_checkpoint_sha256"] = (
            _sha256_file(teacher_path)
            if teacher_path and teacher_path.is_file()
            else None
        )
        penalty_root = Path(penalty_dir) if penalty_dir else None
        if penalty_root and penalty_root.is_dir():
            entries = sorted(
                f"{path.relative_to(penalty_root).as_posix()}:{path.stat().st_size}"
                for path in penalty_root.rglob("*")
                if path.is_file()
            )
            result["penalty_image_count"] = len(entries)
            result["penalty_manifest_sha256"] = hashlib.sha256(
                "\n".join(entries).encode()
            ).hexdigest()
        else:
            result["penalty_image_count"] = None
            result["penalty_manifest_sha256"] = None
    return result


def _make_run_config(
    model_name: str,
    dataset_name: str,
    category: str,
    args,
    output_dir: Path,
) -> dict:
    config_path = (
        Path(args.config)
        if args.config and model_name == args.model
        else MODEL_CONFIGS[model_name]
    )
    base = read_config(config_path)
    base.update(
        dataset=dataset_name,
        categories=[category],
        image_size=args.image_size or int(base["image_size"]),
        batch_size=1
        if model_name == "efficientad"
        else (args.batch_size or int(base["batch_size"])),
        seed=args.seed,
        device=args.device,
        checkpoint_dir=str(output_dir / "checkpoints"),
        threshold_method=args.threshold_method
        or str(base.get("threshold_method", "normal_quantile")),
        threshold_quantile=args.threshold_quantile
        if args.threshold_quantile is not None
        else float(base.get("threshold_quantile", 0.995)),
        mad_multiplier=args.mad_multiplier
        if args.mad_multiplier is not None
        else float(base.get("mad_multiplier", 3.5)),
        run_kind=args.run_kind,
    )
    return base


def _aggregate(run_root: Path, report_root: Path) -> None:
    rows = []
    failures = []
    for status_path in run_root.glob("*/*/status.json"):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("state") == "completed":
            metrics_path = Path(status["metrics_path"])
            if metrics_path.is_file():
                report = json.loads(metrics_path.read_text(encoding="utf-8"))
                for row in report.get("per_category", []):
                    row.update(dataset=status["dataset"], model_name=status["model"])
                    row.update(
                        {
                            f"runtime_{key}": value
                            for key, value in report.get("runtime", {}).items()
                        }
                    )
                    row["training_duration_seconds"] = status.get(
                        "training_duration_seconds"
                    )
                    rows.append(row)
        elif status.get("state") == "failed":
            failures.append(
                {
                    key: status.get(key)
                    for key in (
                        "experiment_id",
                        "dataset",
                        "category",
                        "model",
                        "error",
                    )
                }
            )
    report_root.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with (report_root / "per_category.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    macro_rows = dataset_model_macro_rows(rows)
    macro_fields = sorted({key for row in macro_rows for key in row})
    with (report_root / "dataset_macro.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=macro_fields)
        if macro_fields:
            writer.writeheader()
            writer.writerows(macro_rows)
    _write_json(report_root / "failures.json", {"failed_experiments": failures})


def main() -> int:
    benchmark_defaults = read_config(ROOT / "configs" / "benchmark.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("mpdd", "mvtec", "all"))
    parser.add_argument(
        "--model", required=True, choices=("padim", "patchcore", "efficientad", "all")
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--category", action="append", help="Repeat to select categories"
    )
    selection.add_argument("--all-categories", action="store_true")
    parser.add_argument(
        "--device",
        default=benchmark_defaults.get("device", "auto"),
        help="auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--batch-size", type=int, default=benchmark_defaults.get("batch_size", 8)
    )
    parser.add_argument(
        "--image-size", type=int, default=benchmark_defaults.get("image_size", 256)
    )
    parser.add_argument("--seed", type=int, default=benchmark_defaults.get("seed", 42))
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--config",
        help="One model config override for a single-model, single-category experiment",
    )
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument(
        "--run-kind",
        choices=("smoke_test", "baseline", "tuning", "final_benchmark"),
        default="baseline",
    )
    parser.add_argument("--threshold-method", choices=("normal_quantile", "mad", "f1"))
    parser.add_argument("--threshold-quantile", type=float)
    parser.add_argument("--mad-multiplier", type=float)
    parser.add_argument("--validation-manifest")
    parser.add_argument("--validation-root")
    parser.add_argument(
        "--allow-massive",
        action="store_true",
        help="Explicitly allow more than three model/category runs per invocation",
    )
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--imagenet-penalty-dir")
    parser.add_argument(
        "--allow-pretrained-download",
        action="store_true",
        help="Permit torchvision to download missing ImageNet backbone weights",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    if bool(args.validation_manifest) != bool(args.validation_root):
        parser.error(
            "--validation-manifest and --validation-root must be supplied together"
        )
    if args.threshold_method == "f1" and not args.validation_manifest:
        parser.error(
            "F1 threshold selection requires an independent labeled validation manifest"
        )
    if args.run_kind == "smoke_test" and args.limit_per_split is None:
        parser.error("smoke_test runs require --limit-per-split")
    if args.run_kind != "smoke_test" and args.limit_per_split is not None:
        parser.error("--limit-per-split is reserved for runs tagged smoke_test")
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")
    if args.max_train_images is not None and args.max_train_images < 1:
        parser.error("--max-train-images must be positive")

    # Category metadata is static; inspect it without expanding required data
    # path environment variables so CLI safety guards run before preflight.
    dataset_specs = json.loads(DATASET_CONFIG.read_text(encoding="utf-8"))["datasets"]
    datasets = list(dataset_specs) if args.dataset == "all" else [args.dataset]
    model_names = list(MODEL_CONFIGS) if args.model == "all" else [args.model]
    custom_config = None
    if args.config:
        if args.model == "all" or args.all_categories or len(args.category or []) != 1:
            parser.error("--config requires one --model and exactly one --category")
        custom_config = read_config(args.config)
        if custom_config.get("model") != args.model:
            parser.error("--config model must match --model")
    configured_threshold_methods = {
        args.threshold_method
        or str(
            custom_config.get("threshold_method", "normal_quantile")
            if custom_config is not None
            else read_config(MODEL_CONFIGS[name]).get(
                "threshold_method", "normal_quantile"
            )
        )
        for name in model_names
    }
    if "f1" in configured_threshold_methods and not args.validation_manifest:
        parser.error(
            "F1 threshold selection requires an independent labeled validation manifest"
        )
    categories_by_dataset = {}
    for dataset in datasets:
        known = dataset_specs[dataset]["categories"]
        categories = known if args.all_categories else args.category
        unknown = sorted(set(categories) - set(known))
        if unknown:
            raise SystemExit(f"Unknown {dataset} categories: {unknown}")
        categories_by_dataset[dataset] = categories
    planned_runs = sum(
        len(categories) for categories in categories_by_dataset.values()
    ) * len(model_names)
    if planned_runs > 3 and not args.allow_massive:
        parser.error(
            f"This invocation plans {planned_runs} runs; stage the work or explicitly pass --allow-massive."
        )
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.image_size is not None and args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.model in {"efficientad", "all"} and (
        not args.teacher_checkpoint or not args.imagenet_penalty_dir
    ):
        print(
            "EfficientAD runs will be recorded as failed unless --teacher-checkpoint and --imagenet-penalty-dir are supplied.",
            file=sys.stderr,
        )

    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit(
            "MLflow is required: install with `python -m pip install -e .[mlops]`"
        ) from exc

    run_root = ROOT / "runs" / "benchmark"
    run_root.mkdir(parents=True, exist_ok=True)
    tracking = configure_local_mlflow(ROOT, mlflow)
    invocation_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    failed_count = 0

    for dataset in datasets:
        with mlflow.start_run(
            run_name=f"benchmark-{dataset}-{invocation_id}"
        ) as parent:
            mlflow.set_tags(
                {
                    "run_role": "benchmark_dataset_parent",
                    "dataset": dataset,
                    "phase": "7",
                    "run_kind": args.run_kind,
                    "threshold_method": args.threshold_method or "normal_quantile",
                    "code_version": _code_version() or "unknown",
                }
            )
            for category in categories_by_dataset[dataset]:
                for model_name in model_names:
                    placeholder = run_root / "_id"
                    resources = _resource_fingerprints(
                        dataset,
                        model_name,
                        args.teacher_checkpoint,
                        args.imagenet_penalty_dir,
                    )
                    run_config = _make_run_config(
                        model_name, dataset, category, args, placeholder
                    )
                    if model_name == "efficientad" and args.max_steps is not None:
                        run_config["max_steps"] = args.max_steps
                    run_config["resource_fingerprints"] = resources
                    if model_name == "efficientad":
                        run_config["teacher_checkpoint"] = args.teacher_checkpoint
                        run_config["imagenet_penalty_dir"] = args.imagenet_penalty_dir
                    run_id = experiment_id(
                        dataset=dataset,
                        category=category,
                        model=model_name,
                        seed=args.seed,
                        config=redact_config(run_config),
                    )
                    experiment_dir = run_root / run_id
                    prior_state = latest_attempt_state(experiment_dir)
                    if prior_state == "completed":
                        print(f"SKIP completed {run_id}")
                        continue
                    attempt = next_attempt_number(experiment_dir)
                    attempt_dir = experiment_dir / f"attempt-{attempt:03d}"
                    if attempt_dir.exists():
                        raise RuntimeError(
                            f"Refusing to overwrite existing attempt: {attempt_dir}"
                        )
                    config_path = attempt_dir / "config.json"
                    run_config = _make_run_config(
                        model_name, dataset, category, args, attempt_dir
                    )
                    if model_name == "efficientad" and args.max_steps is not None:
                        run_config["max_steps"] = args.max_steps
                    run_config["resource_fingerprints"] = resources
                    if model_name == "efficientad":
                        run_config["teacher_checkpoint"] = args.teacher_checkpoint
                        run_config["imagenet_penalty_dir"] = args.imagenet_penalty_dir
                    safe_config = redact_config(run_config)
                    _write_json(config_path, run_config)
                    _write_json(attempt_dir / "config.redacted.json", safe_config)
                    env_path = attempt_dir / "environment.json"
                    _write_json(
                        env_path,
                        environment_metadata(
                            [
                                Path(args.config)
                                if args.config and model_name == args.model
                                else MODEL_CONFIGS[model_name],
                                DATASET_CONFIG,
                            ],
                            args.seed,
                        ),
                    )
                    status = {
                        "experiment_id": run_id,
                        "dataset": dataset,
                        "category": category,
                        "model": model_name,
                        "state": "running",
                        "attempt": attempt,
                        "started_at": datetime.now(UTC).isoformat(),
                        "parent_mlflow_run_id": parent.info.run_id,
                    }
                    _write_json(attempt_dir / "status.json", status)
                    run_name = (
                        f"{dataset}-{category}-{model_name}-s{args.seed}-{run_id[-6:]}"
                    )
                    print(f"RUN {run_name} ({attempt_dir.relative_to(ROOT)})")
                    train_log = attempt_dir / "train.log"
                    eval_log = attempt_dir / "evaluate.log"
                    metrics_path = attempt_dir / "metrics" / f"{category}.json"
                    with mlflow.start_run(run_name=run_name, nested=True) as child:
                        mlflow.set_tags(
                            {
                                "run_role": "benchmark_model_category",
                                "experiment_id": run_id,
                                "dataset": dataset,
                                "category": category,
                                "model": model_name,
                                "model_version": "1",
                                "parent_run_id": parent.info.run_id,
                                "run_kind": args.run_kind,
                                "threshold_method": run_config["threshold_method"],
                                "code_version": _code_version() or "unknown",
                            }
                        )
                        mlflow.log_params(
                            {
                                "seed": args.seed,
                                "backbone": run_config.get(
                                    "backbone", "efficientad-pdn"
                                ),
                                "input_resolution": run_config["image_size"],
                                "batch_size": run_config["batch_size"],
                                "config_sha256": hashlib.sha256(
                                    canonical_json(safe_config).encode()
                                ).hexdigest(),
                                "hyperparameters": canonical_json(
                                    safe_config.get("hyperparameters", {})
                                )[:450],
                            }
                        )
                        try:
                            if model_name == "efficientad" and (
                                not args.teacher_checkpoint
                                or not args.imagenet_penalty_dir
                            ):
                                raise ValueError(
                                    "EfficientAD requires explicit --teacher-checkpoint and --imagenet-penalty-dir paths"
                                )
                            if (
                                model_name in {"padim", "patchcore"}
                                and run_config.get("pretrained", True)
                                and not args.allow_pretrained_download
                            ):
                                backbone = run_config.get("backbone")
                                weight_name = (
                                    {
                                        "resnet18": "resnet18-f37072fd.pth",
                                        "resnet34": "resnet34-b627a593.pth",
                                        "resnet50": "resnet50-11ad3fa6.pth",
                                        "wide_resnet50_2": "wide_resnet50_2-9ba9bcbe.pth",
                                    }.get(backbone)
                                    if isinstance(backbone, str)
                                    else None
                                )
                                cached_weights = (
                                    Path(
                                        os.environ.get(
                                            "TORCH_HOME",
                                            str(Path.home() / ".cache" / "torch"),
                                        )
                                    )
                                    / "hub"
                                    / "checkpoints"
                                    / str(weight_name)
                                )
                                if not weight_name or not cached_weights.is_file():
                                    raise FileNotFoundError(
                                        "Pretrained torchvision weights are not cached; pass --allow-pretrained-download to permit an explicit download"
                                    )
                            _write_json(config_path, run_config)
                            train_command = [
                                sys.executable,
                                str(ROOT / "scripts" / "train_model.py"),
                                "--config",
                                str(config_path),
                            ]
                            if args.limit_per_split is not None:
                                train_command.extend(
                                    ("--limit-per-split", str(args.limit_per_split))
                                )
                            if args.max_train_images is not None:
                                train_command.extend(
                                    ("--max-train-images", str(args.max_train_images))
                                )
                            if args.allow_pretrained_download and model_name in {
                                "padim",
                                "patchcore",
                            }:
                                train_command.append("--allow-pretrained-download")
                            eval_command = [
                                sys.executable,
                                str(ROOT / "scripts" / "evaluate_model.py"),
                                "--config",
                                str(config_path),
                                "--output-dir",
                                str(attempt_dir / "metrics"),
                            ]
                            if args.limit_per_split is not None:
                                eval_command.extend(
                                    ("--limit-per-split", str(args.limit_per_split))
                                )
                            if args.validation_manifest and args.validation_root:
                                eval_command.extend(
                                    (
                                        "--validation-manifest",
                                        args.validation_manifest,
                                        "--validation-root",
                                        args.validation_root,
                                    )
                                )
                            for command in (train_command, eval_command):
                                if (
                                    args.teacher_checkpoint
                                    and "efficientad" in model_name
                                ):
                                    command.extend(
                                        (
                                            "--teacher-checkpoint",
                                            args.teacher_checkpoint,
                                        )
                                    )
                                if (
                                    args.imagenet_penalty_dir
                                    and "efficientad" in model_name
                                ):
                                    command.extend(
                                        (
                                            "--imagenet-penalty-dir",
                                            args.imagenet_penalty_dir,
                                        )
                                    )
                            train_started = time.perf_counter()
                            train_result = subprocess.run(
                                train_command,
                                cwd=ROOT,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=False,
                            )
                            training_seconds = time.perf_counter() - train_started
                            train_log.write_text(
                                _safe_text(train_result.stdout), encoding="utf-8"
                            )
                            if train_result.returncode:
                                raise RuntimeError(
                                    f"Training subprocess exited {train_result.returncode}: {train_result.stdout[-1600:]}"
                                )
                            eval_result = subprocess.run(
                                eval_command,
                                cwd=ROOT,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=False,
                            )
                            eval_log.write_text(
                                _safe_text(eval_result.stdout), encoding="utf-8"
                            )
                            if eval_result.returncode:
                                raise RuntimeError(
                                    f"Evaluation subprocess exited {eval_result.returncode}: {eval_result.stdout[-1600:]}"
                                )
                            report = json.loads(
                                metrics_path.read_text(encoding="utf-8")
                            )
                            category_row = report["per_category"][0]
                            calibration_sha256 = report.get("calibration_data_sha256")
                            if calibration_sha256:
                                mlflow.set_tag(
                                    "calibration_data_sha256", calibration_sha256
                                )
                            mlflow.log_metrics(metric_scalars(category_row))
                            mlflow.log_metrics(
                                {
                                    f"runtime_{key}": float(value)
                                    for key, value in report.get("runtime", {}).items()
                                    if isinstance(value, (float, int))
                                }
                            )
                            mlflow.log_metric(
                                "training_duration_seconds", training_seconds
                            )
                            mlflow.log_metric(
                                "training_subprocess_duration_seconds",
                                training_seconds,
                            )
                            training_metadata_path = (
                                attempt_dir
                                / "checkpoints"
                                / model_name
                                / dataset
                                / category
                                / "training_metadata.json"
                            )
                            training_metadata = json.loads(
                                training_metadata_path.read_text(encoding="utf-8")
                            )
                            fit_duration = training_metadata.get(
                                "training_duration_seconds"
                            )
                            if isinstance(fit_duration, (float, int)):
                                mlflow.log_metric(
                                    "model_fit_duration_seconds", float(fit_duration)
                                )
                            fit_peak_memory = training_metadata.get(
                                "training_peak_memory_bytes"
                            )
                            if isinstance(fit_peak_memory, (float, int)):
                                mlflow.log_metric(
                                    "training_peak_memory_bytes", float(fit_peak_memory)
                                )
                            mlflow.log_artifact(
                                str(
                                    attempt_dir
                                    / "checkpoints"
                                    / model_name
                                    / dataset
                                    / category
                                    / "model.pt"
                                ),
                                artifact_path="checkpoint",
                            )
                            mlflow.log_artifact(
                                str(
                                    attempt_dir
                                    / "checkpoints"
                                    / model_name
                                    / dataset
                                    / category
                                    / "training_metadata.json"
                                ),
                                artifact_path="metadata",
                            )
                            mlflow.log_artifact(
                                str(metrics_path), artifact_path="metrics"
                            )
                            mlflow.log_artifact(
                                str(metrics_path.parent / "summary.json"),
                                artifact_path="metrics",
                            )
                            mlflow.log_artifact(
                                str(env_path), artifact_path="reproducibility"
                            )
                            mlflow.log_artifact(
                                str(attempt_dir / "config.redacted.json"),
                                artifact_path="configuration",
                            )
                            for log_path in (train_log, eval_log):
                                mlflow.log_artifact(str(log_path), artifact_path="logs")
                            images_dir = metrics_path.parent / "images"
                            if images_dir.is_dir():
                                mlflow.log_artifacts(
                                    str(images_dir),
                                    artifact_path="representative_images",
                                )
                            status.update(
                                state="completed",
                                finished_at=datetime.now(UTC).isoformat(),
                                metrics_path=str(metrics_path),
                                training_duration_seconds=training_seconds,
                                model_fit_duration_seconds=fit_duration,
                                mlflow_run_id=child.info.run_id,
                            )
                            print(f"DONE {run_name}")
                        except Exception as exc:  # noqa: BLE001 - isolate each failed experiment
                            failed_count += 1
                            safe_error = _safe_text(f"{type(exc).__name__}: {exc}")
                            status.update(
                                state="failed",
                                finished_at=datetime.now(UTC).isoformat(),
                                error=safe_error,
                                mlflow_run_id=child.info.run_id,
                            )
                            mlflow.set_tag("experiment_state", "failed")
                            mlflow.set_tag("failure_reason", safe_error[:450])
                            for path in (
                                train_log,
                                eval_log,
                                env_path,
                                attempt_dir / "config.redacted.json",
                            ):
                                if path.is_file():
                                    mlflow.log_artifact(
                                        str(path), artifact_path="failure_diagnostics"
                                    )
                            print(f"FAILED {run_name}: {exc}", file=sys.stderr)
                        finally:
                            _write_json(attempt_dir / "status.json", status)
                    if status["state"] == "failed" and args.fail_fast:
                        _aggregate(
                            run_root, ROOT / "reports" / "benchmarks" / "aggregates"
                        )
                        return 1
            _aggregate(run_root, ROOT / "reports" / "benchmarks" / "aggregates")
    print(f"Benchmark selection finished; failures={failed_count}; tracking={tracking}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
