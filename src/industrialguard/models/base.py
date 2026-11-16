"""Shared model result and checkpoint interfaces."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class AnomalyOutput:
    """Image-level scores and pixel-level anomaly maps for a batch."""

    image_score: torch.Tensor
    anomaly_map: torch.Tensor


class BaseAnomalyModel(nn.Module, ABC):
    """Common inference and serialization interface; fit remains model-specific."""

    model_name = "base"

    def __init__(self) -> None:
        super().__init__()
        self.fitted = False
        self.model_config: dict[str, Any] = {}

    @abstractmethod
    def fit(self, train_loader, validation_loader=None, **kwargs) -> BaseAnomalyModel:
        """Fit model state using normal training samples only."""

    @abstractmethod
    def predict(self, images: torch.Tensor) -> AnomalyOutput:
        """Return finite image scores and pixel anomaly maps."""

    def metadata(self) -> dict[str, Any]:
        """Describe model type and inference-relevant configuration."""
        return {
            "model": self.model_name,
            "config": self.model_config,
            "fitted": self.fitted,
        }

    def save_checkpoint(self, path: str | Path, **extra: Any) -> None:
        """Save state and metadata without serializing live Python objects."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": self.metadata(),
                "state_dict": self.state_dict(),
                "extra": extra,
            },
            target,
        )

    def load_state(
        self, path: str | Path, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, Any]:
        """Load a checkpoint state dict and return its metadata and extras."""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        stored_metadata = checkpoint.get("metadata", {})
        stored_model = stored_metadata.get("model")
        if stored_model and stored_model != self.model_name:
            raise ValueError(
                f"Checkpoint model {stored_model!r} cannot load into {self.model_name!r}"
            )
        stored_config = stored_metadata.get("config", {})
        for key in (
            "backbone_name",
            "model_size",
            "teacher_out_channels",
            "input_resolution",
            "dataset",
            "category",
        ):
            expected = self.model_config.get(key)
            observed = stored_config.get(key)
            if expected is not None and observed is not None and expected != observed:
                raise ValueError(
                    f"Checkpoint {key}={observed!r} does not match requested {expected!r}"
                )
        state = checkpoint["state_dict"]
        current = self.state_dict()
        for key, value in state.items():
            if key in current and current[key].shape != value.shape:
                parent_name, _, buffer_name = key.rpartition(".")
                parent = self.get_submodule(parent_name) if parent_name else self
                if buffer_name in parent._buffers:
                    parent._buffers[buffer_name] = torch.empty_like(
                        value, device=map_location
                    )
        self.load_state_dict(state)
        self.fitted = bool(stored_metadata.get("fitted", True))
        return {
            "metadata": stored_metadata,
            "extra": checkpoint.get("extra", {}),
        }

    @staticmethod
    def write_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
        """Write portable JSON metadata alongside model weights if desired."""
        Path(path).write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8"
        )
