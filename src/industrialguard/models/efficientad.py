"""EfficientAD adapter using the maintained Anomalib PyTorch implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch

from .base import AnomalyOutput, BaseAnomalyModel
from .postprocessing import gaussian_smooth, image_scores

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _get_images(batch: Any) -> torch.Tensor:
    if isinstance(batch, dict):
        return batch["image"]
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def _as_raw(images: torch.Tensor, *, normalized: bool) -> torch.Tensor:
    images = images.float()
    if not normalized:
        return images
    mean = images.new_tensor(_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(_STD).view(1, 3, 1, 1)
    return images * std + mean


class EfficientAD(BaseAnomalyModel):
    """EfficientAD architecture, loss, and map computation from Anomalib.

    Fitting requires an ImageNet-pretrained teacher state dict and an external
    ImageNet penalty loader. Neither artifact is downloaded automatically.
    """

    model_name = "efficientad"
    network: Any

    def __init__(
        self,
        *,
        teacher_checkpoint: str | Path | None = None,
        model_size: str = "s",
        teacher_out_channels: int = 384,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        smoothing_sigma: float = 2.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        try:
            from anomalib.models.image.efficient_ad.torch_model import (
                EfficientAdModel,
                EfficientAdModelSize,
            )
        except ImportError as exc:
            raise ImportError(
                "EfficientAD requires the optional 'models' dependencies. Install with `pip install -e '.[models]'`."
            ) from exc
        aliases = {"s": "small", "m": "medium", "small": "small", "medium": "medium"}
        if model_size.lower() not in aliases:
            raise ValueError("model_size must be 's'/'small' or 'm'/'medium'")
        if learning_rate <= 0 or weight_decay < 0:
            raise ValueError(
                "learning_rate must be positive and weight_decay non-negative"
            )
        self.network = EfficientAdModel(
            teacher_out_channels=teacher_out_channels,
            model_size=EfficientAdModelSize(aliases[model_size.lower()]),
            padding=False,
            pad_maps=True,
        )
        self.network.to(device)
        self.teacher_checkpoint = (
            Path(teacher_checkpoint) if teacher_checkpoint else None
        )
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.smoothing_sigma = smoothing_sigma
        self.model_config = {
            "implementation": "anomalib.EfficientAdModel",
            "license": "Apache-2.0",
            "model_size": model_size.lower(),
            "teacher_out_channels": teacher_out_channels,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "smoothing_sigma": smoothing_sigma,
            # Never persist a user-specific local resource path in checkpoints.
            "teacher_checkpoint_provided": self.teacher_checkpoint is not None,
        }
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.network.parameters()).device

    def _load_teacher(self) -> None:
        if self.teacher_checkpoint is None or not self.teacher_checkpoint.is_file():
            raise FileNotFoundError(
                "EfficientAD fitting requires a pretrained teacher checkpoint; set teacher_checkpoint."
            )
        state = torch.load(
            self.teacher_checkpoint, map_location=self.device, weights_only=True
        )
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if (
            isinstance(state, dict)
            and state
            and all(key.startswith("teacher.") for key in state)
        ):
            state = {
                key.removeprefix("teacher."): value for key, value in state.items()
            }
        self.network.teacher.load_state_dict(state)
        self.network.teacher.eval()

    @torch.no_grad()
    def _set_teacher_statistics(self, train_loader) -> None:
        total = 0
        sums = squares = None
        for batch in train_loader:
            images = _as_raw(_get_images(batch).to(self.device), normalized=True)
            output = self.network.teacher(images)
            values = output.double()
            batch_count = values.shape[0] * values.shape[2] * values.shape[3]
            batch_sum = values.sum(dim=(0, 2, 3))
            batch_squares = (values * values).sum(dim=(0, 2, 3))
            sums = batch_sum if sums is None else sums + batch_sum
            squares = batch_squares if squares is None else squares + batch_squares
            total += batch_count
        if total == 0 or sums is None or squares is None:
            raise ValueError("EfficientAD received no normal training samples")
        mean = (sums / total).float().view(1, -1, 1, 1)
        variance = (squares / total - (sums / total).square()).clamp_min(1e-12)
        std = variance.sqrt().float().view(1, -1, 1, 1)
        self.network.mean_std["mean"].copy_(mean)
        self.network.mean_std["std"].copy_(std)

    @torch.no_grad()
    def _set_map_quantiles(self, validation_loader) -> None:
        student_maps, ae_maps = [], []
        for batch in validation_loader:
            if (
                isinstance(batch, dict)
                and "label" in batch
                and torch.any(batch["label"] != 0)
            ):
                continue
            images = _as_raw(_get_images(batch).to(self.device), normalized=True)
            student, autoencoder = self.network.get_maps(images, normalize=False)
            student_maps.append(student.flatten())
            ae_maps.append(autoencoder.flatten())
        if not student_maps:
            raise ValueError(
                "EfficientAD threshold calibration requires normal validation samples"
            )
        for name, maps in (("st", student_maps), ("ae", ae_maps)):
            values = torch.cat(maps)
            self.network.quantiles[f"qa_{name}"].copy_(torch.quantile(values, 0.90))
            self.network.quantiles[f"qb_{name}"].copy_(torch.quantile(values, 0.995))

    def fit(
        self,
        train_loader,
        validation_loader=None,
        *,
        penalty_loader=None,
        max_steps: int = 1,
        **kwargs,
    ) -> EfficientAD:
        """Train student and autoencoder with the official three-term loss."""
        if penalty_loader is None:
            raise ValueError(
                "EfficientAD fitting requires ImageNet penalty_loader data"
            )
        if validation_loader is None:
            raise ValueError(
                "EfficientAD requires a separate normal validation loader for map calibration"
            )
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self._load_teacher()
        self._set_teacher_statistics(train_loader)
        self.network.train()
        self.network.teacher.eval()
        optimizer = torch.optim.Adam(
            list(self.network.student.parameters())
            + list(self.network.ae.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        penalty_iterator = iter(penalty_loader)
        self.training_losses: list[float] = []
        completed = 0
        train_iterator = iter(train_loader)
        while completed < max_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_loader)
                try:
                    batch = next(train_iterator)
                except StopIteration as exc:
                    raise ValueError("EfficientAD training loader is empty") from exc
            if completed >= max_steps:
                break
            normal_images = _as_raw(_get_images(batch).to(self.device), normalized=True)
            try:
                penalty_batch = next(penalty_iterator)
            except StopIteration:
                penalty_iterator = iter(penalty_loader)
                try:
                    penalty_batch = next(penalty_iterator)
                except StopIteration as exc:
                    raise ValueError("ImageNet penalty_loader is empty") from exc
            penalty_images = _as_raw(
                _get_images(penalty_batch).to(self.device), normalized=False
            )
            self.network.train()
            self.network.teacher.eval()
            losses = cast(
                tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                self.network(normal_images, penalty_images),
            )
            if not isinstance(losses, tuple) or len(losses) != 3:
                raise RuntimeError("Unexpected Anomalib EfficientAD training output")
            loss = losses[0] + losses[1] + losses[2]
            if not torch.isfinite(loss):
                raise FloatingPointError("EfficientAD produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self.training_losses.append(float(loss.detach().cpu()))
            completed += 1
        if completed == 0:
            raise ValueError("EfficientAD training loader yielded no batches")
        self.network.eval()
        self._set_map_quantiles(validation_loader)
        self.fitted = True
        return self

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> AnomalyOutput:
        if not self.fitted:
            raise RuntimeError("EfficientAD must be fitted before prediction")
        raw = _as_raw(images.to(self.device), normalized=True)
        self.network.eval()
        map_st, map_stae = self.network.get_maps(raw, normalize=True)
        anomaly_map = gaussian_smooth(
            0.5 * map_st + 0.5 * map_stae, self.smoothing_sigma
        )
        return AnomalyOutput(image_scores(anomaly_map), anomaly_map)
