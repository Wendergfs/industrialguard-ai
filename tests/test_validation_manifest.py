import csv

import pytest
import torch
from PIL import Image

from industrialguard.data.validation_manifest import ValidationManifestDataset


def write_manifest(root, rows):
    path = root / "validation_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "sample_id",
                "dataset",
                "category",
                "label",
                "image_path",
                "mask_path",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_rgb(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color).save(path)


def write_mask(path, *, positive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = Image.new("L", (12, 8), 0)
    if positive:
        mask.putpixel((3, 4), 255)
    mask.save(path)


def valid_rows(root):
    write_rgb(root / "images/normal.png", (20, 30, 40))
    write_rgb(root / "images/defect.png", (50, 60, 70))
    write_mask(root / "masks/normal.png")
    write_mask(root / "masks/defect.png", positive=True)
    return [
        {
            "sample_id": "normal-001",
            "dataset": "mpdd-validation",
            "category": "bracket_black",
            "label": 0,
            "image_path": "images/normal.png",
            "mask_path": "masks/normal.png",
        },
        {
            "sample_id": "defect-001",
            "dataset": "mpdd-validation",
            "category": "bracket_black",
            "label": 1,
            "image_path": "images/defect.png",
            "mask_path": "masks/defect.png",
        },
    ]


def test_independent_manifest_loads_labeled_images_and_masks(tmp_path):
    manifest = write_manifest(tmp_path, valid_rows(tmp_path))
    dataset = ValidationManifestDataset(manifest, tmp_path, image_size=16)

    normal, anomalous = dataset[0], dataset[1]

    assert normal["sample_id"] == "normal-001"
    assert normal["image"].shape == (3, 16, 16)
    assert normal["mask"].shape == (1, 16, 16)
    assert int(normal["label"]) == 0
    assert torch.count_nonzero(normal["mask"]) == 0
    assert anomalous["sample_id"] == "defect-001"
    assert dataset.data_sha256 and len(dataset.data_sha256) == 64
    assert normal["image_path"] == "images/normal.png"
    assert int(anomalous["label"]) == 1
    assert torch.count_nonzero(anomalous["mask"]) > 0


def test_manifest_category_filter_still_requires_both_classes(tmp_path):
    manifest = write_manifest(tmp_path, valid_rows(tmp_path))
    with pytest.raises(ValueError, match="no selected samples"):
        ValidationManifestDataset(manifest, tmp_path, categories={"different_category"})


def test_manifest_rejects_duplicate_ids_and_path_traversal(tmp_path):
    rows = valid_rows(tmp_path)
    rows[1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError, match="Duplicate"):
        ValidationManifestDataset(write_manifest(tmp_path, rows), tmp_path)

    rows = valid_rows(tmp_path)
    rows[0]["image_path"] = "../outside.png"
    write_rgb(tmp_path.parent / "outside.png", (0, 0, 0))
    with pytest.raises(ValueError, match="escapes validation root"):
        ValidationManifestDataset(write_manifest(tmp_path, rows), tmp_path)


def test_manifest_requires_masks_and_consistent_normal_labels(tmp_path):
    rows = valid_rows(tmp_path)
    rows[1]["mask_path"] = ""
    with pytest.raises(ValueError, match="requires a mask_path"):
        ValidationManifestDataset(write_manifest(tmp_path, rows), tmp_path)

    rows = valid_rows(tmp_path)
    write_mask(tmp_path / "masks/normal.png", positive=True)
    dataset = ValidationManifestDataset(write_manifest(tmp_path, rows), tmp_path)
    with pytest.raises(ValueError, match="non-empty mask"):
        dataset[0]
