"""Image, pixel, segmentation, and runtime evaluation metrics."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch


def ranking_metrics(labels, scores) -> dict[str, float | None]:
    """Calculate AUROC and average precision, returning None for one-class data."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    prediction = np.asarray(scores, dtype=np.float64).reshape(-1)
    valid = np.isfinite(prediction)
    target, prediction = target[valid], prediction[valid]
    if not len(target) or len(np.unique(target)) < 2:
        return {"auroc": None, "average_precision": None}
    return {
        "auroc": float(roc_auc_score(target, prediction)),
        "average_precision": float(average_precision_score(target, prediction)),
    }


def segmentation_metrics(masks, maps, threshold: float) -> dict[str, float | None]:
    """Compute binary Dice/IoU/precision/recall/F1 at a supplied threshold."""
    truth = np.asarray(masks, dtype=bool).reshape(-1)
    predicted = np.asarray(maps, dtype=np.float64).reshape(-1) >= threshold
    tp = int(np.logical_and(truth, predicted).sum())
    fp = int(np.logical_and(~truth, predicted).sum())
    fn = int(np.logical_and(truth, ~predicted).sum())
    has_positive = tp + fn > 0
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if has_positive else None
    f1 = 2 * tp / (2 * tp + fp + fn) if has_positive else None
    iou = tp / (tp + fp + fn) if has_positive else None
    tn = int(np.logical_and(~truth, ~predicted).sum())
    return {
        "dice": f1,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _mean_region_overlap(
    predicted: np.ndarray, regions: list[tuple[int, np.ndarray]]
) -> float:
    return float(
        np.mean(
            [
                np.logical_and(predicted[image_index], region).sum() / region.sum()
                for image_index, region in regions
            ]
        )
    )


def au_pro(
    masks, maps, *, max_fpr: float = 0.3, threshold_steps: int = 200
) -> float | None:
    """AUPRO implementation with connected components per image."""
    from scipy import ndimage

    masks_array = np.asarray(masks, dtype=bool)
    maps_array = np.asarray(maps, dtype=np.float64)
    if not 0 < max_fpr <= 1 or threshold_steps < 2:
        raise ValueError("max_fpr must be in (0, 1] and threshold_steps at least 2")
    if masks_array.shape != maps_array.shape:
        raise ValueError("masks and maps must have the same shape")
    if not masks_array.any() or masks_array.all():
        return None
    negatives = int((~masks_array).sum())
    regions: list[tuple[int, np.ndarray]] = []
    for image_index, mask in enumerate(masks_array):
        labels, count = ndimage.label(mask)
        regions.extend((image_index, labels == index) for index in range(1, count + 1))
    if not regions or negatives == 0:
        return None
    candidates = np.quantile(maps_array, np.linspace(0.0, 1.0, threshold_steps))
    false_positive_rates, overlaps = [], []
    for threshold in candidates:
        predicted = maps_array >= threshold
        false_positive_rates.append(
            float(np.logical_and(predicted, ~masks_array).sum()) / negatives
        )
        overlaps.append(_mean_region_overlap(predicted, regions))
    order = np.argsort(false_positive_rates)
    fpr = np.asarray(false_positive_rates)[order]
    pro = np.asarray(overlaps)[order]
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_pro = pro[unique_indices]
    if unique_fpr[0] > 0:
        unique_fpr = np.insert(unique_fpr, 0, 0.0)
        unique_pro = np.insert(unique_pro, 0, unique_pro[0])
    keep = unique_fpr <= max_fpr
    x, y = unique_fpr[keep], unique_pro[keep]
    x = np.append(x, max_fpr)
    y = np.append(y, float(np.interp(max_fpr, unique_fpr, unique_pro)))
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) * 0.5) / max_fpr)


def pixel_ranking_metrics(
    masks, maps, *, max_fpr: float = 0.3
) -> dict[str, float | None]:
    """Calculate pixel AUROC/AP and AUPRO independently of any threshold."""
    truth = np.asarray(masks, dtype=np.uint8).reshape(-1)
    scores = np.asarray(maps, dtype=np.float64).reshape(-1)
    metrics = ranking_metrics(truth, scores)
    metrics["aupro"] = au_pro(masks, maps, max_fpr=max_fpr)
    return metrics


def select_threshold(
    labels, scores, *, method: str = "normal_quantile", quantile: float = 0.995
) -> float:
    """Choose a threshold from validation scores only.

    ``normal_quantile`` controls the validation-normal false-positive tail.
    ``f1`` is available only when validation labels contain both classes.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if len(values) != len(target) or not len(values) or not np.isfinite(values).all():
        raise ValueError(
            "validation labels/scores must be finite, non-empty, and aligned"
        )
    if method == "normal_quantile":
        normal_scores = values[target == 0]
        if not len(normal_scores):
            raise ValueError(
                "normal_quantile thresholding requires normal validation samples"
            )
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        return float(np.quantile(normal_scores, quantile))
    if method == "f1":
        if len(np.unique(target)) < 2:
            raise ValueError(
                "F1 threshold selection requires both classes in validation"
            )
        candidates = np.unique(values)
        best_threshold, best_f1 = float(candidates[0]), -1.0
        for threshold in candidates:
            score = segmentation_metrics(target, values, float(threshold))["f1"]
            if score is not None and score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)
        return best_threshold
    raise ValueError("method must be 'normal_quantile' or 'f1'")


def aggregate_by_category(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize metrics by dataset, then macro-average available category values."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["dataset"], []).append(row)
    result = {}
    metric_suffixes = {
        "auroc",
        "average_precision",
        "aupro",
        "dice",
        "iou",
        "precision",
        "recall",
        "f1",
    }
    for dataset, dataset_rows in grouped.items():
        metrics = sorted(
            {
                key
                for row in dataset_rows
                for key, value in row.items()
                if (key.rsplit("_", 1)[-1] in metric_suffixes)
                and (isinstance(value, (int, float)) or value is None)
            }
        )
        means = {}
        for metric in metrics:
            values = [
                row[metric]
                for row in dataset_rows
                if isinstance(row.get(metric), (int, float))
            ]
            means[metric] = float(np.mean(values)) if values else None
        result[dataset] = {
            "category_count": len(dataset_rows),
            "macro_mean": means,
            "per_category": dataset_rows,
        }
    return result


def measure_inference(
    model, batch: torch.Tensor, *, warmup: int = 3, iterations: int = 20
) -> dict[str, float | int | None]:
    """Measure synchronized per-image latency, FPS, and CUDA peak allocation."""
    device = batch.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model.predict(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            model.predict(batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    count = iterations * batch.shape[0]
    latency_ms = elapsed * 1000 / count
    return {
        "latency_ms_per_image": latency_ms,
        "fps": count / elapsed,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
