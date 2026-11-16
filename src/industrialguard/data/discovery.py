"""Infer image roles and mask pairs from archive paths."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_CANDIDATE_EXTENSIONS = IMAGE_EXTENSIONS | {
    ".webp",
    ".gif",
    ".ppm",
    ".pgm",
    ".pbm",
    ".jp2",
}


@dataclass
class ImageRecord:
    dataset: str
    category: str
    image_path: str
    image_type: str
    split: str
    defect_type: str
    mask_path: str = ""
    width: int | None = None
    height: int | None = None
    image_format: str = ""
    mask_width: int | None = None
    mask_height: int | None = None
    mask_empty: bool | None = None
    defective_pixels_percent: float | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_images(
    dataset: str,
    member_names: Iterable[str],
    categories: Iterable[str],
    expected_root: str | None = None,
) -> tuple[list[ImageRecord], list[str]]:
    """Classify image members using observed train/test/ground_truth paths."""
    category_set = set(categories)
    names = list(member_names)
    records: list[ImageRecord] = []
    warnings: list[str] = []
    masks: dict[tuple[str, str, str], str] = {}
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for raw in names:
        path = PurePosixPath(raw)
        parts = path.parts
        if expected_root and parts and parts[0] == expected_root:
            parts = parts[1:]
        normalized.append((raw, parts))
        if len(parts) >= 4 and parts[0] in category_set and parts[1] == "ground_truth":
            defect = parts[2]
            key = (parts[0], defect, path.stem.removesuffix("_mask"))
            masks[key] = raw
    found = {parts[0] for _, parts in normalized if parts and parts[0] in category_set}
    for category in sorted(category_set - found):
        warnings.append(f"Expected category missing: {category}")
    for raw, parts in normalized:
        path = PurePosixPath(raw)
        if (
            path.suffix.lower() not in IMAGE_EXTENSIONS
            or not parts
            or parts[0] not in category_set
        ):
            continue
        category = parts[0]
        if len(parts) < 4:
            warnings.append(f"Unexpected image path: {raw}")
            continue
        if parts[1] == "ground_truth":
            continue
        split = parts[1]
        defect = parts[2]
        if (
            split == "train"
            and defect == "good"
            or split == "test"
            and defect == "good"
        ):
            image_type, defect_type = "normal", ""
        elif split == "test":
            image_type, defect_type = "anomalous", defect
        else:
            warnings.append(f"Unexpected image directory structure: {raw}")
            image_type, defect_type = "unknown", defect
        mask = ""
        if image_type == "anomalous":
            mask = masks.get((category, defect_type, path.stem), "")
        records.append(
            ImageRecord(dataset, category, raw, image_type, split, defect_type, mask)
        )
    return records, warnings


def match_mask(
    image_path: str, mask_names: Iterable[str], category: str, defect_type: str
) -> str | None:
    """Find a mask whose filename stem matches an image stem."""
    stem = PurePosixPath(image_path).stem
    for name in mask_names:
        path = PurePosixPath(name)
        if (
            category in path.parts
            and defect_type in path.parts
            and path.stem.removesuffix("_mask") == stem
        ):
            return name
    return None
