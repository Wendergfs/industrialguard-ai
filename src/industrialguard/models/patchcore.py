"""PatchCore with bounded candidate memory, greedy coreset, and tiled kNN."""

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


def greedy_coreset(
    features: torch.Tensor, size: int, seed: int = 42, projection_dim: int = 64
) -> torch.Tensor:
    """Select a deterministic farthest-first k-center subset."""
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("features must be a non-empty matrix")
    size = min(max(1, size), len(features))
    if size == len(features):
        return features
    generator = torch.Generator(device=features.device).manual_seed(seed)
    projection = torch.randn(
        (features.shape[1], min(projection_dim, features.shape[1])),
        generator=generator,
        device=features.device,
        dtype=features.dtype,
    )
    projected = F.normalize(features @ projection, dim=1)
    first = int(
        torch.randint(
            len(features), (1,), generator=generator, device=features.device
        ).item()
    )
    selected = torch.empty(size, dtype=torch.long, device=features.device)
    selected[0] = first
    min_distance = torch.full(
        (len(features),), torch.inf, dtype=features.dtype, device=features.device
    )
    for index in range(size - 1):
        center = projected[selected[index]]
        distance = torch.sum((projected - center) ** 2, dim=1)
        min_distance = torch.minimum(min_distance, distance)
        selected[index + 1] = torch.argmax(min_distance)
    return features[selected]


class PatchCore(BaseAnomalyModel):
    """Store representative normal patch embeddings and score nearest distances."""

    model_name = "patchcore"
    memory_bank: torch.Tensor

    def __init__(
        self,
        backbone: nn.Module | None = None,
        *,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        coreset_ratio: float = 0.05,
        max_memory_bank: int = 2048,
        max_candidates: int = 8192,
        patch_stride: int = 4,
        query_chunk_size: int = 2048,
        smoothing_sigma: float = 2.0,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        if (
            not 0 < coreset_ratio <= 1
            or min(max_memory_bank, max_candidates, patch_stride, query_chunk_size) < 1
        ):
            raise ValueError(
                "PatchCore memory, stride, and chunk parameters must be positive"
            )
        self.backbone = backbone or build_backbone(backbone_name, pretrained)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.coreset_ratio = coreset_ratio
        self.max_memory_bank = max_memory_bank
        self.max_candidates = max_candidates
        self.patch_stride = patch_stride
        self.query_chunk_size = query_chunk_size
        self.smoothing_sigma = smoothing_sigma
        self.seed = seed
        self.register_buffer("memory_bank", torch.empty(0))
        self.model_config = {
            "backbone": backbone_name,
            "pretrained": pretrained,
            "coreset_ratio": coreset_ratio,
            "max_memory_bank": max_memory_bank,
            "max_candidates": max_candidates,
            "patch_stride": patch_stride,
            "query_chunk_size": query_chunk_size,
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
    def fit(self, train_loader, validation_loader=None, **kwargs) -> PatchCore:
        del validation_loader
        candidate_limit = int(kwargs.get("max_candidates", self.max_candidates))
        if candidate_limit < 1:
            raise ValueError("max_candidates must be positive")
        candidates = torch.empty((0, 0), device=self.memory_bank.device)
        generator = torch.Generator(device=self.memory_bank.device).manual_seed(
            self.seed
        )
        self.backbone.eval()
        for batch in train_loader:
            if (
                isinstance(batch, dict)
                and "label" in batch
                and torch.any(batch["label"] != 0)
            ):
                raise ValueError(
                    "PatchCore may only be fitted with normal training images"
                )
            features = self._features(_images(batch).to(self.memory_bank.device))
            patches = features[:, :, :: self.patch_stride, :: self.patch_stride]
            patches = (
                patches.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).detach()
            )
            if candidates.numel() == 0:
                candidates = patches[:candidate_limit]
            else:
                combined = torch.cat((candidates, patches), dim=0)
                if len(combined) > candidate_limit:
                    keep = torch.randperm(
                        len(combined), generator=generator, device=combined.device
                    )[:candidate_limit]
                    combined = combined[keep]
                candidates = combined
        if candidates.numel() == 0:
            raise ValueError("PatchCore received no training patches")
        target_size = min(
            self.max_memory_bank,
            max(1, round(len(candidates) * self.coreset_ratio)),
        )
        self.memory_bank = greedy_coreset(candidates, target_size, self.seed)
        self.fitted = True
        return self

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> AnomalyOutput:
        if not self.fitted or self.memory_bank.numel() == 0:
            raise RuntimeError("PatchCore must be fitted before prediction")
        images = images.to(self.memory_bank.device)
        self.backbone.eval()
        features = self._features(images)
        sampled = features[:, :, :: self.patch_stride, :: self.patch_stride]
        batch_size, channels, grid_h, grid_w = sampled.shape
        queries = sampled.permute(0, 2, 3, 1).reshape(-1, channels)
        nearest = []
        for start in range(0, len(queries), self.query_chunk_size):
            distances = torch.cdist(
                queries[start : start + self.query_chunk_size], self.memory_bank
            )
            nearest.append(distances.min(dim=1).values)
        patch_scores = torch.cat(nearest).reshape(batch_size, 1, grid_h, grid_w)
        anomaly_map = F.interpolate(
            patch_scores, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        anomaly_map = gaussian_smooth(anomaly_map, self.smoothing_sigma)
        return AnomalyOutput(image_scores(anomaly_map), anomaly_map)

    @property
    def device(self) -> torch.device:
        return (
            self.memory_bank.device
            if self.memory_bank.numel()
            else next(self.parameters(), torch.empty(0)).device
        )
