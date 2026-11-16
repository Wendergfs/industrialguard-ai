"""Configuration-driven PyTorch datasets and DataLoaders for anomaly data."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .archive_reader import ArchiveReader
from .cache_manager import CacheManager
from .dataset_audit import read_config
from .discovery import ImageRecord
from .split_manager import create_splits
from .torch_dataset import IndustrialAnomalyDataset
from .transforms import evaluation_transform, training_transform


def _seed_worker(worker_id: int) -> None:
    """Seed Python random in each worker from DataLoader's torch seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


class IndustrialDataModule:
    """Build train-normal, validation-normal, and official-test DataLoaders."""

    def __init__(
        self,
        dataset: str | None = None,
        categories: list[str] | None = None,
        *,
        dataset_config: str | Path = "configs/dataset.yaml",
        experiment_config: str | Path = "configs/experiment.yaml",
        cache_dir: str | Path | None = None,
        image_size: int | None = None,
        batch_size: int | None = None,
        num_workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        seed: int | None = None,
        validation_fraction: float | None = None,
        limit_per_split: int | None = None,
    ):
        self.dataset_config_path = Path(dataset_config).resolve()
        self.project_root = self.dataset_config_path.parent.parent
        self.dataset_config = read_config(self.dataset_config_path)
        self.experiment = (
            read_config(experiment_config) if Path(experiment_config).is_file() else {}
        )
        self.dataset_name = dataset or self.experiment.get("dataset", "mpdd")
        if self.dataset_name not in self.dataset_config["datasets"]:
            raise ValueError(f"Unknown dataset {self.dataset_name!r}")
        spec = self.dataset_config["datasets"][self.dataset_name]
        self.categories = list(
            categories
            if categories is not None
            else self.experiment.get("categories", spec["categories"])
        )
        unknown = sorted(set(self.categories) - set(spec["categories"]))
        if unknown:
            raise ValueError(f"Unknown {self.dataset_name} categories: {unknown}")
        self.image_size = (
            self.experiment.get("image_size", 256) if image_size is None else image_size
        )
        self.batch_size = (
            self.experiment.get("batch_size", 16) if batch_size is None else batch_size
        )
        self.num_workers = (
            self.experiment.get("num_workers", 0)
            if num_workers is None
            else num_workers
        )
        self.pin_memory = (
            self.experiment.get("pin_memory", False)
            if pin_memory is None
            else pin_memory
        )
        self.persistent_workers = (
            self.experiment.get("persistent_workers", False)
            if persistent_workers is None
            else persistent_workers
        )
        self.seed = self.experiment.get("seed", 42) if seed is None else seed
        self.validation_fraction = (
            self.experiment.get("validation_fraction", 0.1)
            if validation_fraction is None
            else validation_fraction
        )
        self.limit_per_split = limit_per_split
        cache_value = cache_dir or self.experiment.get("cache_dir", ".cache/datasets")
        cache_path = Path(cache_value)
        self.cache_dir = (
            cache_path if cache_path.is_absolute() else self.project_root / cache_path
        )
        if self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("batch_size must be positive and num_workers non-negative")
        if self.limit_per_split is not None and self.limit_per_split < 1:
            raise ValueError("limit_per_split must be positive when provided")
        self._datasets: dict[str, IndustrialAnomalyDataset] = {}
        self.split_records: dict[str, list[ImageRecord]] = {}

    def _load_inventory(self) -> list[ImageRecord]:
        inventory = (
            self.project_root
            / "reports"
            / "dataset_audit"
            / f"{self.dataset_name}_inventory.csv"
        )
        if not inventory.is_file():
            raise FileNotFoundError(
                f"Audit inventory not found: {inventory}. Run the dataset audit first."
            )
        records = []
        with inventory.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row["category"] not in self.categories:
                    continue
                records.append(
                    ImageRecord(
                        dataset=row["dataset"],
                        category=row["category"],
                        image_path=row["image_path"],
                        image_type=row["image_type"],
                        split=row["split"],
                        defect_type=row["defect_type"],
                        mask_path=row["mask_path"],
                    )
                )
        if not records:
            raise ValueError(
                f"No inventory samples found for {self.dataset_name}: {self.categories}"
            )
        return records

    def _limit(self, records: list[ImageRecord], split_name: str) -> list[ImageRecord]:
        limit = self.limit_per_split
        if limit is None or len(records) <= limit:
            return records
        if limit < 1:
            raise ValueError("limit_per_split must be positive when provided")
        ordered = sorted(records, key=lambda record: record.image_path)
        rng = random.Random(self.seed + sum(map(ord, split_name)))
        rng.shuffle(ordered)
        if split_name == "test" and limit >= 2:
            anomaly = next(
                (item for item in ordered if item.image_type == "anomalous"), None
            )
            normal = next(
                (item for item in ordered if item.image_type == "normal"), None
            )
            chosen = [item for item in (normal, anomaly) if item is not None]
            chosen_ids = {item.image_path for item in chosen}
            chosen.extend(item for item in ordered if item.image_path not in chosen_ids)
            return chosen[:limit]
        return ordered[:limit]

    def setup(self) -> None:
        """Discover configured inventory records and cache only selected files."""
        spec = self.dataset_config["datasets"][self.dataset_name]
        records = self._load_inventory()
        splits = create_splits(records, self.validation_fraction, self.seed)
        self.split_records = {
            name: self._limit(split_records, name)
            for name, split_records in splits.items()
        }
        selected = [
            record
            for split_records in self.split_records.values()
            for record in split_records
        ]
        member_names = [record.image_path for record in selected]
        member_names.extend(record.mask_path for record in selected if record.mask_path)
        reader = ArchiveReader(spec["archive"], spec["format"])
        paths = CacheManager(self.cache_dir).materialize(
            reader,
            self.dataset_name,
            spec["archive"],
            member_names,
            spec.get("expected_root"),
        )
        augmentation = self.experiment.get("augmentation", {})
        train_transform = training_transform(
            self.image_size,
            horizontal_flip=bool(augmentation.get("horizontal_flip", False)),
            vertical_flip=bool(augmentation.get("vertical_flip", False)),
        )
        eval_transform = evaluation_transform(self.image_size)
        self._datasets = {
            "train": IndustrialAnomalyDataset(
                self.split_records["train"], paths, train_transform
            ),
            "validation": IndustrialAnomalyDataset(
                self.split_records["validation"], paths, eval_transform
            ),
            "test": IndustrialAnomalyDataset(
                self.split_records["test"], paths, eval_transform
            ),
        }

    def _loader(self, split: str, *, shuffle: bool) -> DataLoader:
        if not self._datasets:
            self.setup()
        dataset = self._datasets[split]
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader("validation", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader("test", shuffle=False)
