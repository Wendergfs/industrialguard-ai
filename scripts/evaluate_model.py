"""Evaluate fitted checkpoints on unchanged official test records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.data_module import IndustrialDataModule
from industrialguard.data.dataset_audit import read_config
from industrialguard.data.validation_manifest import ValidationManifestDataset
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
        "--validation-manifest",
        help="CSV manifest for an independent labeled calibration set",
    )
    parser.add_argument(
        "--validation-root",
        help="Root directory for relative paths in --validation-manifest",
    )
    parser.add_argument(
        "--threshold-method",
        choices=("normal_quantile", "mad", "f1"),
        help="Calibration strategy; f1 requires independent labeled validation",
    )
    parser.add_argument("--threshold-quantile", type=float)
    parser.add_argument("--mad-multiplier", type=float)
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing evaluation outputs"
    )
    args = parser.parse_args()
    if bool(args.validation_manifest) != bool(args.validation_root):
        parser.error(
            "--validation-manifest and --validation-root must be supplied together"
        )
    config = read_config(args.config)
    threshold_method = args.threshold_method or str(
        config.get("threshold_method", "normal_quantile")
    )
    if threshold_method == "f1" and not args.validation_manifest:
        parser.error(
            "F1 threshold selection requires an independent labeled validation manifest"
        )
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
        calibration_loader = None
        if args.validation_manifest and args.validation_root:
            calibration_dataset = ValidationManifestDataset(
                args.validation_manifest,
                args.validation_root,
                image_size=args.image_size or int(config["image_size"]),
                categories={category},
            )
            calibration_loader = DataLoader(
                calibration_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
                num_workers=0,
            )
        report = evaluate_model(
            model,
            data,
            device=device,
            threshold_quantile=(
                args.threshold_quantile
                if args.threshold_quantile is not None
                else float(config.get("threshold_quantile", 0.995))
            ),
            threshold_method=threshold_method,
            mad_multiplier=(
                args.mad_multiplier
                if args.mad_multiplier is not None
                else float(config.get("mad_multiplier", 3.5))
            ),
            calibration_loader=calibration_loader,
        )
        report["limited_smoke_run"] = args.limit_per_split is not None
        _save_outcome_diagnostics(
            model, data, report, category, device, output / "images"
        )
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


def _save_outcome_diagnostics(
    model, data, report: dict, category: str, device, output: Path, limit: int = 5
) -> None:
    """Save map, mask, and overlay diagnostics for image FP/FN/TP/TN outcomes."""
    output.mkdir(parents=True, exist_ok=True)
    dataset = data._datasets["test"]
    sample_indices = {
        f"{record.dataset}:{record.category}:{record.image_path}": index
        for index, record in enumerate(dataset.records)
    }
    row = next(item for item in report["per_category"] if item["category"] == category)
    outcomes: dict[str, list[dict]] = {name: [] for name in ("fp", "fn", "tp", "tn")}
    for item in row["per_image"]:
        outcomes[item["image_outcome"]].append(item)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    image_threshold = float(row["image_threshold"])
    pixel_threshold = float(row["pixel_threshold"])
    palette: list[int] = []
    for value in range(256):
        palette.extend((value, min(255, value * 2), max(0, 255 - value * 2)))
    palette.extend([0] * (768 - len(palette)))
    model.eval()
    with torch.inference_mode():
        for outcome, items in outcomes.items():
            for item in items[:limit]:
                index = sample_indices[item["sample_id"]]
                sample = dataset[index]
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
                predicted_mask = (heat >= pixel_threshold).astype(np.uint8) * 255
                overlay = rgb.copy()
                positive = predicted_mask > 0
                overlay[positive] = (
                    0.55 * overlay[positive] + 0.45 * np.array([255, 0, 0])
                ).astype(np.uint8)
                stem = f"{category}_{outcome}_{index:04d}"
                Image.fromarray(rgb).save(output / f"{stem}_image.png")
                Image.fromarray(mask).save(output / f"{stem}_ground_truth.png")
                np.save(output / f"{stem}_raw_map.npy", heat.astype(np.float32))
                Image.fromarray(scaled).save(output / f"{stem}_normalized_map.png")
                heatmap = Image.fromarray(scaled).convert("P")
                heatmap.putpalette(palette)
                heatmap.save(output / f"{stem}_heatmap.png")
                Image.fromarray(predicted_mask).save(
                    output / f"{stem}_predicted_mask.png"
                )
                Image.fromarray(overlay).save(output / f"{stem}_overlay.png")
                (output / f"{stem}_metrics.json").write_text(
                    json.dumps(
                        {
                            **item,
                            "image_threshold": image_threshold,
                            "pixel_threshold": pixel_threshold,
                            "raw_map_min": low,
                            "raw_map_max": high,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )


if __name__ == "__main__":
    main()
