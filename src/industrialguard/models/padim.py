"""PaDiM patch-wise Gaussian embedding model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .backbones import aggregate_features, build_backbone
from .base import AnomalyOutput, BaseAnomalyModel
from .postprocessing import gaussian_smooth, image_scores


def _images(batch: Any) -> torch.Tensor:
    return batch["image"] if isinstance(batch, dict) else batch


class PaDiM(BaseAnomalyModel):
    """Estimate a regularized Gaussian distribution at every feature location."""

    model_name = "padim"
    projection_indices: torch.Tensor
    mean: torch.Tensor
    precision: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        feature_dim: int = 50,
        covariance_epsilon: float = 1e-4,
        patch_batch_size: int = 256,
        smoothing_sigma: float = 2.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if feature_dim < 1 or covariance_epsilon <= 0 or patch_batch_size < 1:
            raise ValueError(
                "feature_dim, covariance_epsilon, and patch_batch_size must be positive"
            )
        self.backbone = backbone or build_backbone(backbone_name, pretrained)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.feature_dim = feature_dim
        self.covariance_epsilon = covariance_epsilon
        self.patch_batch_size = patch_batch_size
        self.smoothing_sigma = smoothing_sigma
        self.seed = seed
        self.register_buffer("projection_indices", torch.empty(0, dtype=torch.long))
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("precision", torch.empty(0))
        self.model_config = {
            "backbone": backbone_name,
            "pretrained": pretrained,
            "feature_dim": feature_dim,
            "covariance_epsilon": covariance_epsilon,
            "patch_batch_size": patch_batch_size,
            "smoothing_sigma": smoothing_sigma,
            "seed": seed,
        }
        self.to(device)

    def _features(self, images: torch.Tensor) -> torch.Tensor:
        output = self.backbone(images)
        if isinstance(output, torch.Tensor):
            output = {"features": output}
        return aggregate_features(output)

    @torch.no_grad()
    def fit(self, train_loader, validation_loader=None, **kwargs) -> PaDiM:
        del validation_loader
        max_train_images = kwargs.get("max_train_images")
        self.backbone.eval()
        total = 0
        feature_sum = None
        cross_sum = None
        with torch.no_grad():
            for batch in train_loader:
                images = _images(batch).to(self.device)
                if (
                    isinstance(batch, dict)
                    and "label" in batch
                    and torch.any(batch["label"] != 0)
                ):
                    raise ValueError(
                        "PaDiM may only be fitted with normal training images"
                    )
                if max_train_images is not None:
                    remaining = max_train_images - total
                    if remaining <= 0:
                        break
                    images = images[:remaining]
                features = self._features(images)
                batch_size, channels, height, width = features.shape
                if self.projection_indices.numel() == 0:
                    generator = torch.Generator(device="cpu").manual_seed(self.seed)
                    selected_dim = min(self.feature_dim, channels)
                    indices = torch.randperm(channels, generator=generator)[
                        :selected_dim
                    ]
                    self.projection_indices = indices.to(self.device)
                    self.feature_dim = selected_dim
                patches = (
                    features[:, self.projection_indices]
                    .permute(0, 2, 3, 1)
                    .reshape(batch_size, height * width, self.feature_dim)
                    .to(torch.float64)
                )
                batch_sum = patches.sum(dim=0)
                batch_cross = torch.einsum("bpi,bpj->pij", patches, patches)
                feature_sum = (
                    batch_sum if feature_sum is None else feature_sum + batch_sum
                )
                cross_sum = (
                    batch_cross if cross_sum is None else cross_sum + batch_cross
                )
                total += batch_size
        if total == 0 or feature_sum is None or cross_sum is None:
            raise ValueError("PaDiM received no training images")
        mean = feature_sum / total
        if total > 1:
            covariance = (
                cross_sum - total * torch.einsum("pi,pj->pij", mean, mean)
            ) / (total - 1)
        else:
            covariance = torch.zeros_like(cross_sum)
        identity = torch.eye(
            self.feature_dim, dtype=covariance.dtype, device=covariance.device
        )
        scale = (
            covariance.diagonal(dim1=-2, dim2=-1)
            .mean(dim=-1)
            .clamp_min(self.covariance_epsilon)
        )
        covariance = (
            covariance + (self.covariance_epsilon * scale)[:, None, None] * identity
        )
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info):
            jitter = self.covariance_epsilon * 10
            for _ in range(5):
                covariance = covariance + jitter * identity
                cholesky, info = torch.linalg.cholesky_ex(covariance)
                if not torch.any(info):
                    break
                jitter *= 10
            if torch.any(info):
                raise RuntimeError(
                    "PaDiM covariance remained non-positive-definite after regularization"
                )
        self.mean = mean.to(dtype=torch.float32, device=self.device)
        self.precision = torch.cholesky_inverse(cholesky).to(
            dtype=torch.float32, device=self.device
        )
        self.fitted = True
        return self

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyOutput:
        if not self.fitted:
            raise RuntimeError("PaDiM must be fitted before prediction")
        images = images.to(self.device)
        self.backbone.eval()
        features = self._features(images)
        batch_size, _, height, width = features.shape
        patches = (
            features[:, self.projection_indices]
            .permute(0, 2, 3, 1)
            .reshape(batch_size, height * width, self.feature_dim)
        )
        distances = torch.empty(
            (batch_size, height * width), device=images.device, dtype=torch.float32
        )
        for start in range(0, height * width, self.patch_batch_size):
            end = min(start + self.patch_batch_size, height * width)
            delta = patches[:, start:end] - self.mean[start:end].unsqueeze(0)
            distances[:, start:end] = torch.einsum(
                "bpi,pij,bpj->bp", delta, self.precision[start:end], delta
            ).clamp_min_(0)
        anomaly_map = distances.reshape(batch_size, 1, height, width)
        anomaly_map = F.interpolate(
            anomaly_map, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        anomaly_map = gaussian_smooth(anomaly_map, self.smoothing_sigma)
        return AnomalyOutput(image_scores(anomaly_map), anomaly_map)

    @property
    def device(self) -> torch.device:
        return (
            self.mean.device
            if self.mean.numel()
            else next(self.parameters(), torch.empty(0)).device
        )
