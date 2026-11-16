"""Reproducible, leakage-safe train/validation/test split construction."""

from __future__ import annotations

import hashlib
import random

from .discovery import ImageRecord


def create_splits(
    records: list[ImageRecord], validation_fraction: float = 0.1, seed: int = 42
) -> dict[str, list[ImageRecord]]:
    """Keep official test data fixed; derive validation only from train normals."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    train_by_category: dict[str, list[ImageRecord]] = {}
    test_records: list[ImageRecord] = []
    for record in records:
        if record.split == "train":
            if record.image_type != "normal":
                raise ValueError(
                    f"Training split contains non-normal sample: {record.image_path}"
                )
            train_by_category.setdefault(record.category, []).append(record)
        elif record.split == "test":
            test_records.append(record)
        else:
            raise ValueError(
                f"Unknown official split {record.split!r}: {record.image_path}"
            )

    train_records: list[ImageRecord] = []
    validation_records: list[ImageRecord] = []
    for category, category_records in sorted(train_by_category.items()):
        ordered = sorted(category_records, key=lambda record: record.image_path)
        category_seed = int.from_bytes(
            hashlib.sha256(category.encode("utf-8")).digest()[:4], "big"
        )
        random.Random(seed + category_seed).shuffle(ordered)
        val_count = 0
        if validation_fraction > 0 and len(ordered) > 1:
            val_count = min(
                len(ordered) - 1, max(1, round(len(ordered) * validation_fraction))
            )
        validation_records.extend(ordered[:val_count])
        train_records.extend(ordered[val_count:])
    result = {
        "train": train_records,
        "validation": validation_records,
        "test": sorted(test_records, key=lambda record: record.image_path),
    }
    assert_no_leakage(result)
    return result


def assert_no_leakage(splits: dict[str, list[ImageRecord]]) -> None:
    """Raise when a sample path occurs in more than one split."""
    seen: dict[str, str] = {}
    for split_name, records in splits.items():
        for record in records:
            previous = seen.get(record.image_path)
            if previous is not None:
                raise ValueError(
                    f"Data leakage: {record.image_path} appears in {previous} and {split_name}"
                )
            seen[record.image_path] = split_name
