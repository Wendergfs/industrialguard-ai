"""Independent labeled validation data loaded from a user-managed manifest."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import PairedTransform, evaluation_transform

_REQUIRED_COLUMNS = {
    "sample_id",
    "dataset",
    "category",
    "label",
    "image_path",
    "mask_path",
}


@dataclass(frozen=True)
class ValidationRecord:
    """One labeled image and optional pixel mask in an independent holdout."""

    sample_id: str
    dataset: str
    category: str
    label: int
    image_path: Path
    mask_path: Path | None


class ValidationManifestDataset(Dataset):
    """Load independent validation images without changing official datasets.

    The CSV manifest must have columns ``sample_id,dataset,category,label,
    image_path,mask_path``. Image and mask paths are relative to ``root``.
    Labels are 0 for normal and 1 for anomalous; every anomalous row requires
    a binary ground-truth mask. The source directory is only read.
    """

    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        *,
        image_size: int | tuple[int, int] = 256,
        categories: set[str] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest).resolve()
        self.root = Path(root).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError("Independent validation manifest is unavailable")
        if not self.root.is_dir():
            raise NotADirectoryError("Independent validation root is unavailable")
        self.transform: PairedTransform = evaluation_transform(image_size)
        self.records = self._read_records(categories)
        if not self.records:
            raise ValueError("Validation manifest contains no selected samples")
        if not any(record.label == 0 for record in self.records):
            raise ValueError("Independent validation requires normal samples")
        if not any(record.label == 1 for record in self.records):
            raise ValueError("Independent validation requires anomalous samples")
        self.data_sha256 = self._fingerprint()

    def _fingerprint(self) -> str:
        """Hash manifest and image/mask content without persisting source paths."""
        digest = hashlib.sha256(self.manifest_path.read_bytes())
        for record in sorted(self.records, key=lambda item: item.sample_id):
            digest.update(record.sample_id.encode("utf-8"))
            digest.update(str(record.label).encode("ascii"))
            for path in (record.image_path, record.mask_path):
                if path is None:
                    digest.update(b"<no-mask>")
                    continue
                file_digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                digest.update(file_digest.digest())
        return digest.hexdigest()

    def _resolve_member(self, value: str, field: str, sample_id: str) -> Path | None:
        value = value.strip()
        if not value:
            return None
        relative = Path(value)
        if relative.is_absolute() or PureWindowsPath(value).is_absolute():
            raise ValueError(f"{field} must be relative for sample {sample_id!r}")
        resolved = (self.root / relative).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError(
                f"{field} escapes validation root for sample {sample_id!r}"
            )
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Missing {field} for validation sample {sample_id!r}"
            )
        return resolved

    def _read_records(self, categories: set[str] | None) -> list[ValidationRecord]:
        records = []
        sample_ids: set[str] = set()
        with self.manifest_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            missing = sorted(_REQUIRED_COLUMNS - fields)
            if missing:
                raise ValueError(f"Validation manifest missing columns: {missing}")
            for row in reader:
                sample_id = (row.get("sample_id") or "").strip()
                dataset = (row.get("dataset") or "").strip()
                category = (row.get("category") or "").strip()
                if not sample_id or not dataset or not category:
                    raise ValueError(
                        "sample_id, dataset, and category must be non-empty"
                    )
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate validation sample_id: {sample_id!r}")
                sample_ids.add(sample_id)
                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label for validation sample {sample_id!r}"
                    ) from exc
                if label not in (0, 1):
                    raise ValueError(f"Label must be 0 or 1 for sample {sample_id!r}")
                if categories is not None and category not in categories:
                    continue
                image_path = self._resolve_member(
                    row["image_path"], "image_path", sample_id
                )
                mask_path = self._resolve_member(
                    row["mask_path"], "mask_path", sample_id
                )
                if image_path is None:
                    raise ValueError(f"image_path is required for sample {sample_id!r}")
                if label == 1 and mask_path is None:
                    raise ValueError(
                        f"Anomalous sample {sample_id!r} requires a mask_path"
                    )
                records.append(
                    ValidationRecord(
                        sample_id=sample_id,
                        dataset=dataset,
                        category=category,
                        label=label,
                        image_path=image_path,
                        mask_path=mask_path,
                    )
                )
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        with Image.open(record.image_path) as source:
            image = source.convert("RGB")
        if record.mask_path is None:
            mask = Image.new("L", image.size, color=0)
        else:
            with Image.open(record.mask_path) as source_mask:
                mask = source_mask.convert("L")
            if mask.size != image.size:
                raise ValueError(
                    f"Image/mask dimensions differ for sample {record.sample_id!r}"
                )
            if record.label == 0 and mask.getbbox() is not None:
                raise ValueError(
                    f"Normal sample {record.sample_id!r} has a non-empty mask"
                )
        image_tensor, mask_tensor = self.transform(image, mask)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": torch.tensor(record.label, dtype=torch.long),
            "category": record.category,
            "dataset": record.dataset,
            "sample_id": record.sample_id,
            "image_path": record.image_path.relative_to(self.root).as_posix(),
            "mask_path": (
                record.mask_path.relative_to(self.root).as_posix()
                if record.mask_path
                else ""
            ),
        }
