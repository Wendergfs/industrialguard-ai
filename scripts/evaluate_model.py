"""Evaluate fitted checkpoints on unchanged official test records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.data_module import IndustrialDataModule
from industrialguard.data.dataset_audit import read_config
from industrialguard.evaluation.metrics import aggregate_by_category
from industrialguard.evaluation.runner import evaluate_model
from industrialguard.mlops.reproducibility import seed_everything
from industrialguard.models.registry import create_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint", help="Use one explicit checkpoint for a single category"
    )
    parser.add_argument("--dataset", choices=("mpdd", "mvtec"))
    parser.add_argument("--category", action="append")
    parser.add_argument("--device")
    parser.add_argument(
        "--limit-per-split",
        type=int,
        help="Smoke-only cap; capped metrics are not benchmark metrics",
    )
    parser.add_argument("--image-size", type=int)
    parser.add_argument(
        "--output-dir", help="Optional run-specific metrics/artifact directory"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing evaluation outputs"
    )
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(int(config.get("seed", 42)))
    model_name = config["model"]
    dataset_name = args.dataset or config["dataset"]
    categories = args.category or config["categories"]
    if args.checkpoint and len(categories) != 1:
        parser.error("--checkpoint can be used with exactly one --category")
    device_value = args.device or config.get("device", "auto")
    device = torch.device(
        "cuda"
        if device_value == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_value == "auto"
        else device_value
    )
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(config.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_root.is_absolute():
        checkpoint_root = project_root / checkpoint_root
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else project_root
        / "reports"
        / "model_development"
        / "results"
        / model_name
        / dataset_name
    )
    output.mkdir(parents=True, exist_ok=True)
    prior_outputs = [output / f"{category}.json" for category in categories]
    prior_outputs.append(output / "summary.json")
    if not args.overwrite and any(path.exists() for path in prior_outputs):
        existing = next(path for path in prior_outputs if path.exists())
        parser.error(
            f"Evaluation output already exists: {existing}. Choose a new --output-dir or pass --overwrite."
        )
    rows = []
    for category in categories:
        kwargs = dict(config.get("hyperparameters", {}))
        if model_name in {"padim", "patchcore"}:
            # Checkpoints already contain the backbone state; avoid weight downloads here.
            kwargs.update(
                backbone_name=config["backbone"],
                pretrained=False,
                seed=int(config.get("seed", 42)),
            )
        if model_name == "efficientad":
            kwargs["teacher_checkpoint"] = config.get("teacher_checkpoint")
        kwargs["device"] = device
        model = create_model(model_name, **kwargs)
        model.model_config.update(
            {
                "dataset": dataset_name,
                "category": category,
                "input_resolution": args.image_size or int(config["image_size"]),
                "seed": int(config.get("seed", 42)),
                "batch_size": int(config["batch_size"]),
                "model_version": "1",
            }
        )
        checkpoint = (
            Path(args.checkpoint)
            if args.checkpoint
            else (checkpoint_root / model_name / dataset_name / category / "model.pt")
        )
        model.load_state(checkpoint, map_location=device)
        model.eval()
        data = IndustrialDataModule(
            dataset_name,
            [category],
            image_size=args.image_size or int(config["image_size"]),
            batch_size=int(config["batch_size"]),
            num_workers=int(config.get("num_workers", 0)),
            seed=int(config.get("seed", 42)),
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            cache_dir=config.get("cache_dir", ".cache/datasets"),
            limit_per_split=args.limit_per_split,
        )
        data.setup()
        report = evaluate_model(
            model,
            data,
            device=device,
            threshold_quantile=float(config.get("threshold_quantile", 0.995)),
        )
        report["limited_smoke_run"] = args.limit_per_split is not None
        _save_representative_images(model, data, category, device, output / "images")
        result_path = output / f"{category}.json"
        result_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        rows.extend(report["per_category"])
        print(f"Saved per-category test metrics: {result_path}")
    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "category_count": len(rows),
        "aggregation": "macro mean across categories; test samples are not pooled",
        "dataset_macro_mean": aggregate_by_category(rows)[dataset_name]["macro_mean"],
        "limited_smoke_run": args.limit_per_split is not None,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _save_representative_images(
    model, data, category: str, device, output: Path
) -> None:
    """Save one normal and one anomalous input, mask, and predicted map."""
    output.mkdir(parents=True, exist_ok=True)
    dataset = data._datasets["test"]
    selected: dict[str, dict] = {}
    for index, record in enumerate(dataset.records):
        kind = "anomalous" if record.image_type == "anomalous" else "normal"
        if kind not in selected:
            selected[kind] = dataset[index]
        if len(selected) == 2:
            break
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    model.eval()
    with torch.inference_mode():
        for kind, sample in selected.items():
            image = sample["image"].unsqueeze(0).to(device)
            prediction = model.predict(image)
            rgb = (
                (sample["image"].cpu() * std + mean)
                .clamp(0, 1)
                .permute(1, 2, 0)
                .numpy()
                * 255
            ).astype(np.uint8)
            mask = (sample["mask"][0].numpy() > 0).astype(np.uint8) * 255
            heat = prediction.anomaly_map[0, 0].detach().float().cpu().numpy()
            low, high = float(heat.min()), float(heat.max())
            scaled = (
                np.zeros_like(heat, dtype=np.uint8)
                if high <= low
                else ((heat - low) / (high - low) * 255).astype(np.uint8)
            )
            stem = f"{category}_{kind}"
            Image.fromarray(rgb).save(output / f"{stem}_image.png")
            Image.fromarray(mask).save(output / f"{stem}_mask.png")
            Image.fromarray(scaled).save(output / f"{stem}_predicted_map.png")


if __name__ == "__main__":
    main()
