"""Anomaly-map smoothing and image-score calculation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_smooth(anomaly_map: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
    """Apply separable Gaussian smoothing to Bx1xHxW maps."""
    if sigma <= 0:
        return anomaly_map
    radius = max(1, round(3 * sigma))
    coords = torch.arange(
        -radius, radius + 1, device=anomaly_map.device, dtype=anomaly_map.dtype
    )
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    channels = anomaly_map.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = F.pad(anomaly_map, (radius, radius, 0, 0), mode="reflect")
    smoothed = F.conv2d(padded, horizontal, groups=channels)
    padded = F.pad(smoothed, (0, 0, radius, radius), mode="reflect")
    return F.conv2d(padded, vertical, groups=channels)


def image_scores(anomaly_map: torch.Tensor, reduction: str = "max") -> torch.Tensor:
    """Reduce pixel maps to one anomaly score per image."""
    flat = anomaly_map.flatten(start_dim=1)
    if reduction == "max":
        return flat.amax(dim=1)
    if reduction == "mean":
        return flat.mean(dim=1)
    raise ValueError("reduction must be 'max' or 'mean'")
