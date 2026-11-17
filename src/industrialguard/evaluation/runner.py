"""Leakage-safe validation thresholding and test evaluation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from .metrics import (
    aggregate_by_category,
    measure_inference,
    pixel_ranking_metrics,
    ranking_metrics,
    segmentation_metrics,
    select_threshold,
    threshold_sweep,
)


def _collect(
    model, loader, device: str | torch.device
) -> dict[str, dict[str, list[Any]]]:
    collected: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"labels": [], "scores": [], "maps": [], "masks": [], "sample_ids": []}
    )
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            output = model.predict(images)
            maps = output.anomaly_map.detach().float().cpu().numpy()
            scores = output.image_score.detach().float().cpu().numpy()
            labels = batch["label"].detach().cpu().numpy()
            masks = batch["mask"].detach().cpu().numpy()
            for index, category in enumerate(batch["category"]):
                bucket = collected[category]
                bucket["labels"].append(int(labels[index]))
                bucket["scores"].append(float(scores[index]))
                bucket["maps"].append(maps[index, 0])
                bucket["masks"].append(masks[index, 0].astype(bool))
                sample_ids = batch.get("sample_id")
                bucket["sample_ids"].append(
                    str(sample_ids[index])
                    if sample_ids is not None
                    else str(len(bucket["labels"]) - 1)
                )
    return collected


def evaluate_model(
    model,
    data_module,
    *,
    device: str | torch.device = "cpu",
    threshold_quantile: float = 0.995,
    threshold_method: str = "normal_quantile",
    mad_multiplier: float = 3.5,
    calibration_loader=None,
) -> dict[str, Any]:
    """Calibrate per category on validation data and score the official test once.

    ``f1`` requires a separate labeled calibration loader containing both
    normal and anomalous images. It must never be constructed from the official
    test split. Normal-only quantile and MAD strategies use normal calibration
    observations and do not require anomaly labels.
    """
    loader = (
        data_module.val_dataloader()
        if calibration_loader is None
        else calibration_loader
    )
    validation = _collect(model, loader, device)
    thresholds: dict[str, dict[str, Any]] = {}
    for category, validation_data in validation.items():
        labels = np.asarray(validation_data["labels"], dtype=np.uint8)
        scores = np.asarray(validation_data["scores"], dtype=np.float64)
        maps = np.stack(validation_data["maps"])
        masks = np.stack(validation_data["masks"])
        thresholds[category] = {
            "image": select_threshold(
                labels,
                scores,
                method=threshold_method,
                quantile=threshold_quantile,
                mad_multiplier=mad_multiplier,
            ),
            "pixel": select_threshold(
                masks.reshape(-1).astype(np.uint8),
                maps.reshape(-1),
                method=threshold_method,
                quantile=threshold_quantile,
                mad_multiplier=mad_multiplier,
            ),
            "selection": threshold_method,
            "quantile": threshold_quantile,
            "mad_multiplier": mad_multiplier if threshold_method == "mad" else None,
            "validation_normal_images": int((labels == 0).sum()),
            "validation_anomalous_images": int((labels == 1).sum()),
        }
        image_candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 101)))
        pixel_values = maps.reshape(-1)
        pixel_candidates = np.unique(np.quantile(pixel_values, np.linspace(0, 1, 101)))
        thresholds[category]["validation_image_curve"] = threshold_sweep(
            labels, scores, image_candidates
        )
        thresholds[category]["validation_pixel_curve"] = threshold_sweep(
            masks.reshape(-1).astype(np.uint8), pixel_values, pixel_candidates
        )
    expected_categories = getattr(data_module, "categories", None)
    if expected_categories is not None:
        missing_calibration = sorted(set(expected_categories) - set(thresholds))
        if missing_calibration:
            raise ValueError(
                "Independent calibration data is missing categories: "
                + ", ".join(missing_calibration)
            )
    # Freeze all thresholds before the test loader is read or any test score is
    # computed. This keeps the evaluation boundary explicit in the code path.
    test = _collect(model, data_module.test_dataloader(), device)
    rows = []
    for category, data in test.items():
        if category not in thresholds:
            raise ValueError(
                f"No validation calibration samples for category {category}"
            )
        labels = np.asarray(data["labels"], dtype=np.uint8)
        scores = np.asarray(data["scores"], dtype=np.float64)
        masks = np.stack(data["masks"])
        maps = np.stack(data["maps"])
        image_metrics = ranking_metrics(labels, scores)
        pixel_metrics = pixel_ranking_metrics(masks, maps)
        image_segmentation = segmentation_metrics(
            labels, scores, thresholds[category]["image"]
        )
        pixel_segmentation = segmentation_metrics(
            masks, maps, thresholds[category]["pixel"]
        )
        image_threshold = thresholds[category]["image"]
        pixel_threshold = thresholds[category]["pixel"]
        per_image = []
        for index, sample_id in enumerate(data["sample_ids"]):
            image_prediction = bool(scores[index] >= image_threshold)
            image_label = bool(labels[index])
            image_outcome = (
                "tp"
                if image_label and image_prediction
                else "fn"
                if image_label
                else "fp"
                if image_prediction
                else "tn"
            )
            pixel_counts = segmentation_metrics(
                masks[index], maps[index], pixel_threshold
            )
            per_image.append(
                {
                    "sample_id": sample_id,
                    "label": int(labels[index]),
                    "image_score": float(scores[index]),
                    "image_prediction": int(image_prediction),
                    "image_outcome": image_outcome,
                    "pixel_tp": pixel_counts["tp"],
                    "pixel_fp": pixel_counts["fp"],
                    "pixel_fn": pixel_counts["fn"],
                    "pixel_tn": pixel_counts["tn"],
                    "pixel_precision": pixel_counts["precision"],
                    "pixel_recall": pixel_counts["recall"],
                    "pixel_dice": pixel_counts["dice"],
                    "pixel_iou": pixel_counts["iou"],
                }
            )
        result = {
            "dataset": data_module.dataset_name,
            "category": category,
            "test_images": len(labels),
            "per_image": per_image,
            "normal_images": int((labels == 0).sum()),
            "anomalous_images": int((labels == 1).sum()),
            "image_threshold": image_threshold,
            "pixel_threshold": pixel_threshold,
            **{f"image_{key}": value for key, value in image_metrics.items()},
            **{f"pixel_{key}": value for key, value in pixel_metrics.items()},
            **{f"image_{key}": value for key, value in image_segmentation.items()},
            **{
                f"segmentation_{key}": value
                for key, value in pixel_segmentation.items()
            },
        }
        rows.append(result)
    first_batch = next(iter(data_module.test_dataloader()))["image"][:1].to(device)
    runtime = measure_inference(model, first_batch)
    return {
        "model": model.metadata(),
        "dataset": data_module.dataset_name,
        "calibration_data_sha256": getattr(
            getattr(loader, "dataset", None), "data_sha256", None
        ),
        "threshold_source": (
            "independent labeled validation"
            if threshold_method == "f1"
            else "normal validation only"
        ),
        "thresholds_by_category": thresholds,
        "per_category": rows,
        "dataset_macro_mean": aggregate_by_category(rows)[data_module.dataset_name][
            "macro_mean"
        ],
        "runtime": runtime,
    }
