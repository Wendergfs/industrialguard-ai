"""Non-mutating image and annotation checks for archived members."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import PurePosixPath

from PIL import Image, UnidentifiedImageError

from .archive_reader import ArchiveReader
from .discovery import IMAGE_CANDIDATE_EXTENSIONS, IMAGE_EXTENSIONS, ImageRecord


def validate_records(
    reader: ArchiveReader, records: list[ImageRecord], all_names: Iterable[str]
) -> dict:
    """Decode each inventory image and mask incrementally and report issues."""
    errors: list[dict] = []
    warnings: list[dict] = []
    dimensions: Counter[str] = Counter()
    invalid_images = empty_masks = missing_masks = invalid_masks = 0
    supported = unsupported = 0
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in all_names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    for name in duplicates:
        errors.append({"path": name, "issue": "duplicate_archive_path"})
    indexed = {r.image_path: r for r in records}
    mask_to_records: dict[str, list[ImageRecord]] = {}
    for record in records:
        if record.mask_path:
            mask_to_records.setdefault(record.mask_path, []).append(record)
    masks_by_path: set[str] = set()
    invalid_masks = 0
    invalid_images = 0
    for name, payload in reader.iter_member_bytes():
        path = PurePosixPath(name)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            if path.suffix.lower() in IMAGE_CANDIDATE_EXTENSIONS:
                unsupported += 1
                warnings.append({"path": name, "issue": "unsupported_image_extension"})
            continue
        supported += 1
        image_record = indexed.get(name)
        try:
            from io import BytesIO

            is_mask = "/ground_truth/" in f"/{name}/"
            if is_mask:
                masks_by_path.add(name)
            with BytesIO(payload) as stream, Image.open(stream) as image:
                image.load()
                width, height = image.size
                fmt = image.format or path.suffix.lstrip(".").upper()
                if is_mask:
                    if image.getbbox() is None:
                        empty_masks += 1
                        warnings.append({"path": name, "issue": "empty_mask"})
                    for paired in mask_to_records.get(name, []):
                        paired.mask_width, paired.mask_height = width, height
                        paired.mask_empty = image.getbbox() is None
                        histogram = image.convert("L").histogram()
                        total = sum(histogram)
                        paired.defective_pixels_percent = (
                            100.0 * (total - histogram[0]) / total if total else 0.0
                        )
                elif image_record:
                    (
                        image_record.width,
                        image_record.height,
                        image_record.image_format,
                    ) = (
                        width,
                        height,
                        fmt,
                    )
                    dimensions[f"{width}x{height}"] += 1
        except (OSError, ValueError, UnidentifiedImageError, EOFError) as exc:
            if is_mask:
                invalid_masks += 1
            else:
                invalid_images += 1
            issue = "unreadable_mask" if is_mask else "unreadable_image"
            errors.append({"path": name, "issue": issue, "detail": str(exc)})
    matched_masks = {r.mask_path for r in records if r.mask_path}
    for name in masks_by_path - matched_masks:
        warnings.append({"path": name, "issue": "unmatched_mask"})
    for record in records:
        if record.image_type != "anomalous":
            continue
        if not record.mask_path or record.mask_path not in masks_by_path:
            missing_masks += 1
            errors.append({"path": record.image_path, "issue": "missing_mask"})
            continue
        if (record.width, record.height) != (record.mask_width, record.mask_height):
            warnings.append(
                {
                    "path": record.image_path,
                    "issue": "mask_dimension_mismatch",
                    "image_size": [record.width, record.height],
                    "mask_size": [record.mask_width, record.mask_height],
                }
            )
    return {
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "supported_images": supported,
            "unsupported_images": unsupported,
            "invalid_images": invalid_images,
            "invalid_masks": invalid_masks,
            "empty_masks": empty_masks,
            "missing_masks": missing_masks,
            "duplicate_paths": len(duplicates),
            "dimensions": dict(dimensions),
        },
        "mask_member_count": len(masks_by_path),
    }
