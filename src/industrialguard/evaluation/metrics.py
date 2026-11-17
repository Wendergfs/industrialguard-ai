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


def au_pro(
    masks, maps, *, max_fpr: float = 0.3, threshold_steps: int = 200
) -> float | None:
    """Exact rank-based AUPRO up to ``max_fpr`` with per-image components.

    ``threshold_steps`` is retained for call compatibility; the curve now uses
    every distinct map score so tied thresholds and sparse low-FPR regions are
    handled without quantile-grid approximation.
    """
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
    del threshold_steps
    region_labels = np.zeros(masks_array.shape, dtype=np.int64)
    next_region = 1
    structure = np.ones((3, 3), dtype=np.uint8)
    for image_index, mask in enumerate(masks_array):
        labels, count = ndimage.label(mask, structure=structure)
        if count:
            foreground = labels > 0
            region_labels[image_index, foreground] = (
                labels[foreground] + next_region - 1
            )
            next_region += count
    region_count = next_region - 1
    if not region_count or negatives == 0:
        return None
    labels_flat = region_labels.reshape(-1)
    scores_flat = maps_array.reshape(-1)
    region_sizes = np.bincount(labels_flat, minlength=region_count + 1)
    background = labels_flat == 0
    fp_change = background.astype(np.float64)
    pro_change = np.zeros_like(scores_flat, dtype=np.float64)
    foreground = ~background
    pro_change[foreground] = 1.0 / region_sizes[labels_flat[foreground]]

    order = np.argsort(-scores_flat, kind="mergesort")
    sorted_scores = scores_flat[order]
    fp_cumulative = np.cumsum(fp_change[order]) / int(background.sum())
    pro_cumulative = np.cumsum(pro_change[order]) / region_count
    # Include one point per distinct threshold, after all tied scores enter.
    ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(sorted_scores) - 1
    ]
    fpr = np.r_[0.0, fp_cumulative[ends]]
    pro = np.r_[0.0, pro_cumulative[ends]]

    last_below_limit = np.flatnonzero(fpr <= max_fpr)
    if not len(last_below_limit):
        x = np.array([0.0, max_fpr])
        y = np.array([0.0, 0.0])
    else:
        last = int(last_below_limit[-1])
        x = fpr[: last + 1]
        y = pro[: last + 1]
        if x[-1] < max_fpr:
            next_index = last + 1
            if next_index < len(fpr):
                interpolated = float(
                    y[-1]
                    + (pro[next_index] - y[-1])
                    * (max_fpr - x[-1])
                    / (fpr[next_index] - x[-1])
                )
            else:
                interpolated = float(y[-1])
            x = np.append(x, max_fpr)
            y = np.append(y, interpolated)
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
    labels,
    scores,
    *,
    method: str = "normal_quantile",
    quantile: float = 0.995,
    mad_multiplier: float = 3.5,
) -> float:
    """Choose a threshold from validation scores only.

    ``normal_quantile`` controls the validation-normal false-positive tail.
    ``f1`` is available only when validation labels contain both classes.
    """
    values, target = _validated_scores(labels, scores)
    if method == "normal_quantile":
        normal_scores = values[target == 0]
        if not len(normal_scores):
            raise ValueError(
                "normal_quantile thresholding requires normal validation samples"
            )
        if not 0 < quantile < 1:
            raise ValueError("quantile must be in (0, 1)")
        return float(np.quantile(normal_scores, quantile))
    if method == "mad":
        normal_scores = values[target == 0]
        if not len(normal_scores):
            raise ValueError("MAD thresholding requires normal validation samples")
        if mad_multiplier <= 0 or not np.isfinite(mad_multiplier):
            raise ValueError("mad_multiplier must be positive and finite")
        median = float(np.median(normal_scores))
        mad = float(np.median(np.abs(normal_scores - median)))
        scaled_mad = 1.4826 * mad
        # A zero MAD means the calibration observations are tied. Put the
        # threshold just above that tie so identical normal scores stay normal.
        if scaled_mad == 0:
            return float(np.nextafter(median, np.inf))
        return median + mad_multiplier * scaled_mad
    if method == "f1":
        if len(np.unique(target)) < 2:
            raise ValueError(
                "F1 threshold selection requires both classes in validation"
            )
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        sorted_positive = target[order].astype(np.int64)
        prefix_positive = np.concatenate(
            (np.zeros(1, dtype=np.int64), np.cumsum(sorted_positive))
        )
        starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
        true_positive = int(target.sum()) - prefix_positive[starts]
        predicted_positive = len(values) - starts
        false_positive = predicted_positive - true_positive
        false_negative = int(target.sum()) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        f1 = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=denominator > 0,
        )
        # Candidates are ascending, so ties select the lower threshold and
        # retain the higher-recall operating point.
        return float(sorted_values[starts[int(np.argmax(f1))]])
    raise ValueError("method must be 'normal_quantile', 'mad', or 'f1'")


def _validated_scores(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    """Validate aligned binary labels and finite scores used for calibration."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    raw_target = np.asarray(labels).reshape(-1)
    if (
        len(values) != len(raw_target)
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isin(raw_target, (0, 1)).all()
    ):
        raise ValueError(
            "validation labels/scores must be finite, non-empty, binary, and aligned"
        )
    return values, raw_target.astype(np.uint8, copy=False)


def threshold_sweep(labels, scores, thresholds) -> list[dict[str, float | int | None]]:
    """Evaluate fixed thresholds efficiently for validation PR/F1 analysis.

    Sorting scores once keeps pixel-level sweeps practical for multi-million
    pixel arrays. Callers choose thresholds from calibration data before
    evaluating a held-out test split.
    """
    values, target = _validated_scores(labels, scores)
    candidates = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    if not len(candidates) or not np.isfinite(candidates).all():
        raise ValueError("thresholds must be non-empty and finite")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    prefix_positive = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(target[order], dtype=np.int64))
    )
    starts = np.searchsorted(sorted_values, candidates, side="left")
    total_positive = int(target.sum())
    total_negative = len(target) - total_positive
    true_positive = total_positive - prefix_positive[starts]
    false_positive = (len(values) - starts) - true_positive
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.full(len(candidates), np.nan),
        where=precision_denominator > 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.full(len(candidates), np.nan),
        where=recall_denominator > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.full(len(candidates), np.nan),
        where=f1_denominator > 0,
    )
    iou_denominator = true_positive + false_positive + false_negative
    iou = np.divide(
        true_positive,
        iou_denominator,
        out=np.full(len(candidates), np.nan),
        where=iou_denominator > 0,
    )
    rows = []
    for index, threshold in enumerate(candidates):
        rows.append(
            {
                "threshold": float(threshold),
                "tp": int(true_positive[index]),
                "fp": int(false_positive[index]),
                "fn": int(false_negative[index]),
                "tn": int(true_negative[index]),
                "precision": None
                if np.isnan(precision[index])
                else float(precision[index]),
                "recall": None if np.isnan(recall[index]) else float(recall[index]),
                "f1": None if np.isnan(f1[index]) else float(f1[index]),
                "dice": None if np.isnan(f1[index]) else float(f1[index]),
                "iou": None if np.isnan(iou[index]) else float(iou[index]),
            }
        )
    return rows


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
