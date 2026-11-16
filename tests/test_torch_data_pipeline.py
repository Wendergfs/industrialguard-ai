import csv
import io
import json
import tarfile
import zipfile

import pytest
import torch
from PIL import Image, ImageDraw

from industrialguard.data.archive_reader import ArchiveReader
from industrialguard.data.cache_manager import CacheManager
from industrialguard.data.data_module import IndustrialDataModule
from industrialguard.data.dataset_audit import read_config
from industrialguard.data.discovery import discover_images
from industrialguard.data.split_manager import assert_no_leakage, create_splits
from industrialguard.data.torch_dataset import IndustrialAnomalyDataset
from industrialguard.data.transforms import evaluation_transform


def png_bytes(color=(80, 120, 160), *, mask=False):
    image = Image.new("L" if mask else "RGB", (24, 20), 0 if mask else color)
    if mask:
        ImageDraw.Draw(image).rectangle((5, 5, 12, 12), fill=255)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def make_archive(path, dataset):
    category = "bracket_black" if dataset == "mpdd" else "bottle"
    prefix = f"MPDD/{category}" if dataset == "mpdd" else category
    files = {}
    for index in range(8):
        files[f"{prefix}/train/good/train_{index:03}.png"] = png_bytes((index, 80, 120))
    files[f"{prefix}/test/good/good_000.png"] = png_bytes()
    files[f"{prefix}/test/good/good_001.png"] = png_bytes((100, 100, 100))
    for index in range(2):
        files[f"{prefix}/test/dent/defect_{index:03}.png"] = png_bytes((180, 40, 30))
        files[f"{prefix}/ground_truth/dent/defect_{index:03}_mask.png"] = png_bytes(
            mask=True
        )
    if dataset == "mpdd":
        with zipfile.ZipFile(path, "w") as archive:
            for name, payload in files.items():
                archive.writestr(name, payload)
        archive_format, expected_root = "zip", "MPDD"
    else:
        with tarfile.open(path, "w:xz") as archive:
            for name, payload in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
        archive_format, expected_root = "tar.xz", None
    return files, archive_format, expected_root, category


@pytest.mark.parametrize("dataset", ["mpdd", "mvtec"])
def test_discovery_dataset_loading_masks_and_split_leakage(tmp_path, dataset):
    archive_path = tmp_path / ("fixture.zip" if dataset == "mpdd" else "fixture.tar.xz")
    files, archive_format, expected_root, category = make_archive(archive_path, dataset)
    records, warnings = discover_images(dataset, files, [category], expected_root)
    assert not warnings
    splits = create_splits(records, validation_fraction=0.25, seed=37)
    assert_no_leakage(splits)
    assert all(
        item.image_type == "normal" and item.split == "train"
        for item in splits["train"]
    )
    assert all(item.split == "train" for item in splits["validation"])
    assert {item.split for item in splits["test"]} == {"test"}
    assert create_splits(records, 0.25, 37)["validation"] == splits["validation"]

    paths = CacheManager(tmp_path / "cache").materialize(
        ArchiveReader(archive_path, archive_format),
        dataset,
        archive_path,
        [record.image_path for record in records]
        + [record.mask_path for record in records if record.mask_path],
        expected_root,
    )
    evaluation = IndustrialAnomalyDataset(records, paths, evaluation_transform(32))
    normal = evaluation[
        next(i for i, record in enumerate(records) if record.image_type == "normal")
    ]
    anomalous = evaluation[
        next(i for i, record in enumerate(records) if record.image_type == "anomalous")
    ]
    assert (
        normal["label"].item() == 0 and torch.count_nonzero(normal["mask"]).item() == 0
    )
    assert (
        anomalous["label"].item() == 1
        and torch.count_nonzero(anomalous["mask"]).item() > 0
    )
    assert set(torch.unique(anomalous["mask"]).tolist()) <= {0.0, 1.0}
    assert normal["image"].shape[-2:] == normal["mask"].shape[-2:] == (32, 32)
    assert anomalous["image"].shape[-2:] == anomalous["mask"].shape[-2:]
    assert {"sample_id", "category", "dataset"} <= anomalous.keys()


@pytest.mark.parametrize("dataset", ["mpdd", "mvtec"])
def test_data_module_cpu_loader_and_official_test_samples(tmp_path, dataset):
    project = tmp_path / "project"
    (project / "configs").mkdir(parents=True)
    (project / "reports" / "dataset_audit").mkdir(parents=True)
    archive_path = tmp_path / ("fixture.zip" if dataset == "mpdd" else "fixture.tar.xz")
    files, archive_format, expected_root, category = make_archive(archive_path, dataset)
    records, _ = discover_images(dataset, files, [category], expected_root)
    with (project / "reports" / "dataset_audit" / f"{dataset}_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].as_dict()))
        writer.writeheader()
        writer.writerows(record.as_dict() for record in records)
    dataset_config = project / "configs" / "dataset.yaml"
    dataset_config.write_text(
        json.dumps(
            {
                "datasets": {
                    dataset: {
                        "archive": str(archive_path),
                        "format": archive_format,
                        "expected_root": expected_root,
                        "categories": [category],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    experiment_config = project / "configs" / "experiment.yaml"
    experiment_config.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "categories": [category],
                "image_size": 24,
                "batch_size": 4,
                "validation_fraction": 0.25,
                "cache_dir": ".cache/datasets",
                "seed": 15,
            }
        ),
        encoding="utf-8",
    )

    module = IndustrialDataModule(
        dataset,
        dataset_config=dataset_config,
        experiment_config=experiment_config,
        limit_per_split=4,
        num_workers=0,
    )
    module.setup()
    batch = next(iter(module.train_dataloader()))
    assert batch["image"].device.type == "cpu"
    assert batch["image"].shape == (4, 3, 24, 24)
    assert batch["mask"].shape == (4, 1, 24, 24)
    assert set(torch.unique(batch["mask"]).tolist()) <= {0.0, 1.0}
    test_batch = next(iter(module.test_dataloader()))
    assert set(test_batch["label"].tolist()) == {0, 1}
    assert all(
        path.startswith(("MPDD/", "bottle/")) for path in test_batch["image_path"]
    )


def test_cache_repairs_corruption_and_is_idempotent(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    member = "MPDD/bracket_black/train/good/one.png"
    payload = png_bytes()
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, payload)
    manager = CacheManager(tmp_path / "cache")
    reader = ArchiveReader(archive_path, "zip")
    first = manager.materialize(reader, "mpdd", archive_path, [member], "MPDD")
    cached = first[member]
    stamp = cached.stat().st_mtime_ns
    second = manager.materialize(reader, "mpdd", archive_path, [member], "MPDD")
    assert second[member] == cached and cached.stat().st_mtime_ns == stamp
    cached.write_bytes(b"corrupt")
    repaired = manager.materialize(reader, "mpdd", archive_path, [member], "MPDD")
    assert repaired[member].read_bytes() == payload


@pytest.mark.parametrize(
    "member", ["../outside.png", "C:/outside.png", "\\\\server\\outside.png"]
)
def test_cache_rejects_unsafe_member_paths(member):
    with pytest.raises(ValueError, match="Unsafe archive member path"):
        CacheManager._safe_relative(member, None)


def test_environment_archive_paths_are_json_escaped(tmp_path, monkeypatch):
    config = tmp_path / "dataset.yaml"
    config.write_text(
        '{"archive": "${INDUSTRIALGUARD_TEST_ARCHIVE}"}', encoding="utf-8"
    )
    archive = r"D:\datasets\MPDD (1).zip"
    monkeypatch.setenv("INDUSTRIALGUARD_TEST_ARCHIVE", archive)
    assert read_config(config)["archive"] == archive
