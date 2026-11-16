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
)


def _collect(
    model, loader, device: str | torch.device
) -> dict[str, dict[str, list[Any]]]:
    collected: dict[str, dict[str, list[Any]]] = defaultdict(
        lambda: {"labels": [], "scores": [], "maps": [], "masks": []}
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
    return collected


def evaluate_model(
    model,
    data_module,
    *,
    device: str | torch.device = "cpu",
    threshold_quantile: float = 0.995,
) -> dict[str, Any]:
    """Calibrate each category on normal validation only and score official test."""
    validation = _collect(model, data_module.val_dataloader(), device)
    test = _collect(model, data_module.test_dataloader(), device)
    rows = []
    thresholds: dict[str, dict[str, Any]] = {}
    for category, validation_data in validation.items():
        labels = np.asarray(validation_data["labels"], dtype=np.uint8)
        scores = np.asarray(validation_data["scores"], dtype=np.float64)
        maps = np.stack(validation_data["maps"])
        thresholds[category] = {
            "image": select_threshold(labels, scores, quantile=threshold_quantile),
            "pixel": select_threshold(
                np.zeros(maps.size, dtype=np.uint8),
                maps.reshape(-1),
                quantile=threshold_quantile,
            ),
            "selection": "validation normal-score quantile",
            "quantile": threshold_quantile,
        }
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
        result = {
            "dataset": data_module.dataset_name,
            "category": category,
            "test_images": len(labels),
            "normal_images": int((labels == 0).sum()),
            "anomalous_images": int((labels == 1).sum()),
            "image_threshold": thresholds[category]["image"],
            "pixel_threshold": thresholds[category]["pixel"],
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
        "threshold_source": "validation only; no test samples used for threshold selection",
        "thresholds_by_category": thresholds,
        "per_category": rows,
        "dataset_macro_mean": aggregate_by_category(rows)[data_module.dataset_name][
            "macro_mean"
        ],
        "runtime": runtime,
    }
