"""PyTorch dataset returning images, binary masks, labels, and provenance."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .discovery import ImageRecord
from .transforms import PairedTransform


class IndustrialAnomalyDataset(Dataset):
    """Map discovered records to normalized image and pixel-mask tensors."""

    def __init__(
        self,
        records: list[ImageRecord],
        cached_paths: Mapping[str, Path],
        transform: PairedTransform,
    ):
        self.records = list(records)
        self.cached_paths = cached_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image_path = self.cached_paths[record.image_path]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        if record.image_type == "anomalous":
            if not record.mask_path:
                raise FileNotFoundError(
                    f"Anomalous image has no discovered mask: {record.image_path}"
                )
            mask_path = self.cached_paths[record.mask_path]
            with Image.open(mask_path) as source_mask:
                mask = source_mask.convert("L")
        else:
            mask_path = None
            mask = Image.new("L", image.size, color=0)
        image_tensor, mask_tensor = self.transform(image, mask)
        if image_tensor.shape[-2:] != mask_tensor.shape[-2:]:
            raise RuntimeError(
                f"Image/mask dimensions differ for sample {record.image_path}"
            )
        sample_id = f"{record.dataset}:{record.category}:{record.image_path}"
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(
                int(record.image_type == "anomalous"), dtype=torch.long
            ),
            "category": record.category,
            "dataset": record.dataset,
            "sample_id": sample_id,
            "image_path": record.image_path,
            "mask_path": record.mask_path,
        }
