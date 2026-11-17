"""Privacy-conscious experiment identifiers and local MLflow setup."""

from __future__ import annotations

import hashlib
import importlib
import json
import urllib.parse
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    """Serialize configuration deterministically for fingerprints and artifacts."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def experiment_id(
    *,
    dataset: str,
    category: str,
    model: str,
    seed: int,
    config: dict[str, Any],
) -> str:
    """Stable ID for one fully specified model/category experiment."""
    identity = {
        "dataset": dataset,
        "category": category,
        "model": model,
        "seed": seed,
        "config": config,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:12]
    return f"{dataset}-{category}-{model}-s{seed}-{digest}"


def redact_config(value: Any, key: str = "") -> Any:
    """Remove local data/resource paths and secret-like values from artifacts."""
    private_keys = {
        "archive",
        "teacher_checkpoint",
        "imagenet_penalty_dir",
        "cache_dir",
        "checkpoint_dir",
    }
    if isinstance(value, dict):
        return {k: redact_config(v, k.lower()) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_config(item, key) for item in value]
    if key in private_keys:
        return "<local path omitted>" if value else None
    if any(
        token in key.lower()
        for token in ("password", "secret", "token", "credential", "api_key")
    ):
        return "<redacted>" if value else None
    return value


def tracking_uri(project_root: Path) -> str:
    """Return a persistent, local-only SQLite-backed MLflow tracking URI."""
    database = (project_root / "runs" / "mlflow" / "mlflow.db").resolve()
    database.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{database.as_posix()}"


def configure_local_mlflow(project_root: Path, mlflow_module=None) -> str:
    """Configure a persistent local SQLite store and local artifact root only."""
    if mlflow_module is None:
        mlflow_module = importlib.import_module("mlflow")
    root = (project_root / "runs" / "mlflow").resolve()
    root.mkdir(parents=True, exist_ok=True)
    uri = tracking_uri(project_root)
    mlflow_module.set_tracking_uri(uri)
    experiment_name = "IndustrialGuard-Phase4-Benchmark"
    existing = mlflow_module.get_experiment_by_name(experiment_name)
    if existing is None:
        artifact = (root / "artifacts").as_uri()
        mlflow_module.create_experiment(
            experiment_name,
            artifact_location=urllib.parse.quote(artifact, safe=":/"),
        )
    mlflow_module.set_experiment(experiment_name)
    return uri


def latest_attempt_state(experiment_dir: Path) -> str | None:
    """Read latest immutable attempt state used by benchmark resume logic."""
    attempts = sorted(experiment_dir.glob("attempt-*/status.json"))
    if not attempts:
        return None
    return json.loads(attempts[-1].read_text(encoding="utf-8")).get("state")


def next_attempt_number(experiment_dir: Path) -> int:
    """Choose a new attempt directory without overwriting any past attempt."""
    return 1 + len(list(experiment_dir.glob("attempt-*")))


def metric_scalars(row: dict[str, Any]) -> dict[str, float]:
    """Select defined numeric metrics suitable for MLflow scalar logging."""
    output: dict[str, float] = {}
    for key, value in row.items():
        if (
            isinstance(value, (float, int))
            and not isinstance(value, bool)
            and key
            not in {
                "category",
                "test_images",
                "normal_images",
                "anomalous_images",
            }
        ):
            output[key] = float(value)
    return output


def dataset_model_macro_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate categories separately by dataset and model, skipping undefined metrics."""
    suffixes = {
        "auroc",
        "average_precision",
        "aupro",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
    }
    runtime_keys = (
        "runtime_latency_ms_per_image",
        "runtime_fps",
        "runtime_peak_memory_bytes",
        "training_duration_seconds",
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["model_name"]), []).append(row)
    output = []
    for (dataset, model), group in sorted(grouped.items()):
        keys = {
            key for row in group for key in row if key.rsplit("_", 1)[-1] in suffixes
        }
        keys.update(key for key in runtime_keys if any(key in row for row in group))
        means = {}
        for key in sorted(keys):
            values = [
                row[key]
                for row in group
                if isinstance(row.get(key), (int, float))
                and not isinstance(row.get(key), bool)
            ]
            means[key] = sum(values) / len(values) if values else None
        output.append(
            {
                "dataset": dataset,
                "model": model,
                "category_count": len(group),
                **means,
            }
        )
    return output
