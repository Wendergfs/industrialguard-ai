from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrialguard.mlops.reproducibility import seed_everything
from industrialguard.mlops.tracking import (
    configure_local_mlflow,
    dataset_model_macro_rows,
    experiment_id,
    latest_attempt_state,
    metric_scalars,
    next_attempt_number,
    redact_config,
    tracking_uri,
)


def test_experiment_identifier_is_stable_and_configuration_sensitive():
    common = {
        "dataset": "mpdd",
        "category": "bracket_black",
        "model": "padim",
        "seed": 42,
    }
    first = experiment_id(**common, config={"image_size": 256})
    assert first == experiment_id(**common, config={"image_size": 256})
    assert first != experiment_id(**common, config={"image_size": 512})


def test_redaction_removes_local_resource_paths_and_secrets():
    clean = redact_config(
        {
            "teacher_checkpoint": "C:/Users/person/private.pth",
            "hyperparameters": {"learning_rate": 1e-4},
            "api_token": "secret-value",
        }
    )
    assert "person" not in json.dumps(clean)
    assert "secret-value" not in json.dumps(clean)
    assert clean["hyperparameters"]["learning_rate"] == 1e-4


def test_checkpoint_resume_never_overwrites_prior_attempts(tmp_path: Path):
    experiment = tmp_path / "stable-id"
    assert latest_attempt_state(experiment) is None
    attempt = experiment / "attempt-001"
    attempt.mkdir(parents=True)
    (attempt / "status.json").write_text('{"state":"failed"}', encoding="utf-8")
    assert latest_attempt_state(experiment) == "failed"
    assert next_attempt_number(experiment) == 2
    (attempt / "status.json").write_text('{"state":"completed"}', encoding="utf-8")
    assert latest_attempt_state(experiment) == "completed"


def test_mlflow_logs_to_persistent_local_store(tmp_path: Path):
    mlflow = pytest.importorskip("mlflow")
    uri = configure_local_mlflow(tmp_path, mlflow)
    assert uri.startswith("sqlite:///")
    assert Path(uri.removeprefix("sqlite:///")).is_file()
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"image_auroc":0.9}', encoding="utf-8")
    with mlflow.start_run(run_name="unit-test-run"):
        mlflow.log_params({"dataset": "mpdd", "seed": 42})
        mlflow.log_metric("image_auroc", 0.9)
        mlflow.log_artifact(str(artifact))
    experiment = mlflow.get_experiment_by_name("IndustrialGuard-Phase4-Benchmark")
    run = mlflow.search_runs(experiment_ids=[experiment.experiment_id]).iloc[0]
    assert run["params.dataset"] == "mpdd"
    assert run["metrics.image_auroc"] == pytest.approx(0.9)


def test_metric_logging_skips_undefined_values():
    assert metric_scalars(
        {"image_auroc": 0.75, "pixel_auroc": None, "test_images": 5}
    ) == {"image_auroc": 0.75}


def test_seed_everything_repeats_torch_values():
    import torch

    seed_everything(7)
    first = torch.rand(4)
    seed_everything(7)
    assert torch.equal(first, torch.rand(4))


def test_dvc_metadata_is_not_ignored_and_archives_are_not_targets():
    root = Path(__file__).resolve().parents[1]
    ignored = (root / ".gitignore").read_text(encoding="utf-8")
    lines = [line.strip() for line in ignored.splitlines()]
    assert "*.dvc" not in lines
    assert ".dvc/" not in lines
    setup_script = (root / "scripts" / "setup_dvc.py").read_text(encoding="utf-8")
    assert '"reports/dataset_audit"' in setup_script
    assert '"configs"' in setup_script
    assert "*.zip" not in setup_script
    assert "*.tar.xz" not in setup_script


def test_tracking_uri_is_project_local(tmp_path: Path):
    uri = tracking_uri(tmp_path)
    assert uri.startswith("sqlite:///")
    assert str(tmp_path).replace("\\", "/") in uri


def test_macro_aggregation_keeps_datasets_separate_and_skips_undefined():
    rows = [
        {
            "dataset": "mpdd",
            "model_name": "padim",
            "category": "a",
            "image_auroc": 0.8,
            "pixel_aupro": None,
        },
        {
            "dataset": "mpdd",
            "model_name": "padim",
            "category": "b",
            "image_auroc": 1.0,
            "pixel_aupro": 0.4,
        },
        {
            "dataset": "mvtec",
            "model_name": "padim",
            "category": "screw",
            "image_auroc": 0.5,
            "pixel_aupro": 0.2,
        },
    ]
    result = dataset_model_macro_rows(rows)
    assert len(result) == 2
    mpdd = next(row for row in result if row["dataset"] == "mpdd")
    mvtec = next(row for row in result if row["dataset"] == "mvtec")
    assert mpdd["category_count"] == 2
    assert mpdd["image_auroc"] == pytest.approx(0.9)
    assert mpdd["pixel_aupro"] == pytest.approx(0.4)
    assert mvtec["image_auroc"] == pytest.approx(0.5)


def test_initialized_dvc_config_has_no_remote():
    root = Path(__file__).resolve().parents[1]
    config = root / ".dvc" / "config"
    if config.exists():
        contents = config.read_text(encoding="utf-8")
        assert "[remote" not in contents
        assert "analytics = false" in contents
