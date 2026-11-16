"""Explicitly train or fit one configured anomaly model/category."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.data_module import IndustrialDataModule
from industrialguard.data.dataset_audit import read_config
from industrialguard.evaluation.metrics import select_threshold
from industrialguard.evaluation.runner import _collect
from industrialguard.mlops.reproducibility import seed_everything
from industrialguard.models.registry import create_model


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _imagenet_penalty_loader(
    directory: str | Path,
    image_size: int,
    batch_size: int,
    workers: int,
    pin_memory: bool,
):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("EfficientAD penalty loading requires torchvision") from exc
    transform = transforms.Compose(
        [
            transforms.Resize((image_size * 2, image_size * 2)),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(str(directory), transform=transform)
    if not len(dataset):
        raise ValueError(f"ImageNet penalty directory contains no images: {directory}")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="One configs/models/*.yaml file"
    )
    parser.add_argument("--dataset", choices=("mpdd", "mvtec"))
    parser.add_argument("--category", action="append")
    parser.add_argument("--device")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--imagenet-penalty-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--allow-pretrained-download",
        action="store_true",
        help="Permit torchvision to download missing ImageNet backbone weights",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing checkpoint"
    )
    parser.add_argument(
        "--random-backbone",
        action="store_true",
        help="Disable pretrained weights for isolated architecture smoke checks only",
    )
    args = parser.parse_args()
    config = read_config(args.config)
    model_name = config["model"]
    dataset_name = args.dataset or config["dataset"]
    categories = args.category or config["categories"]
    device = _device(args.device or config.get("device", "auto"))
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    project_root = Path(__file__).resolve().parents[1]
    checkpoint_root = Path(config.get("checkpoint_dir", "checkpoints"))
    if not checkpoint_root.is_absolute():
        checkpoint_root = project_root / checkpoint_root

    for category in categories:
        target = checkpoint_root / model_name / dataset_name / category / "model.pt"
        if target.exists() and not args.overwrite:
            raise FileExistsError(
                f"Checkpoint already exists: {target}. Choose a new checkpoint_dir or pass --overwrite."
            )
        pretrained = False
        if model_name in {"padim", "patchcore"}:
            pretrained = (
                bool(config.get("pretrained", True)) and not args.random_backbone
            )
            weight_file = {
                "resnet18": "resnet18-f37072fd.pth",
                "resnet34": "resnet34-b627a593.pth",
                "resnet50": "resnet50-11ad3fa6.pth",
                "wide_resnet50_2": "wide_resnet50_2-9ba9bcbe.pth",
            }.get(config["backbone"])
            torch_home = Path(
                os.environ.get("TORCH_HOME", str(Path.home() / ".cache" / "torch"))
            )
            weight_path = torch_home / "hub" / "checkpoints" / str(weight_file)
            if (
                pretrained
                and not args.allow_pretrained_download
                and (not weight_file or not weight_path.is_file())
            ):
                raise FileNotFoundError(
                    "Pretrained torchvision weights are not cached. Pass "
                    "--allow-pretrained-download to explicitly permit a download."
                )
        data = IndustrialDataModule(
            dataset_name,
            [category],
            image_size=args.image_size or int(config["image_size"]),
            batch_size=args.batch_size or int(config["batch_size"]),
            num_workers=int(config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            seed=seed,
            validation_fraction=float(config.get("validation_fraction", 0.1)),
            cache_dir=config.get("cache_dir", ".cache/datasets"),
            limit_per_split=args.limit_per_split,
        )
        data.setup()
        model_kwargs = dict(config.get("hyperparameters", {}))
        if model_name in {"padim", "patchcore"}:
            model_kwargs.update(
                backbone_name=config["backbone"],
                pretrained=pretrained,
                seed=seed,
            )
        if model_name == "efficientad":
            model_kwargs["teacher_checkpoint"] = args.teacher_checkpoint or config.get(
                "teacher_checkpoint"
            )
        model_kwargs["device"] = device
        model = create_model(model_name, **model_kwargs)
        model.model_config.update(
            {
                "dataset": dataset_name,
                "category": category,
                "input_resolution": args.image_size or int(config["image_size"]),
                "seed": seed,
                "batch_size": args.batch_size or int(config["batch_size"]),
                "model_version": "1",
            }
        )
        training_started = time.perf_counter()
        if model_name == "efficientad":
            penalty_dir = args.imagenet_penalty_dir or config.get(
                "imagenet_penalty_dir"
            )
            if not penalty_dir:
                raise ValueError(
                    "EfficientAD requires --imagenet-penalty-dir or config imagenet_penalty_dir"
                )
            penalty_loader = _imagenet_penalty_loader(
                penalty_dir,
                args.image_size or int(config["image_size"]),
                int(config.get("penalty_batch_size", 1)),
                int(config.get("num_workers", 0)),
                device.type == "cuda",
            )
            model.fit(
                data.train_dataloader(),
                data.val_dataloader(),
                penalty_loader=penalty_loader,
                max_steps=args.max_steps or int(config["max_steps"]),
            )
        elif model_name == "padim":
            model.fit(
                data.train_dataloader(),
                data.val_dataloader(),
                max_train_images=args.max_train_images,
            )
        else:
            model.fit(data.train_dataloader(), data.val_dataloader())

        validation = _collect(model, data.val_dataloader(), device)
        val = validation[category]
        image_threshold = select_threshold(
            val["labels"],
            val["scores"],
            quantile=float(config.get("threshold_quantile", 0.995)),
        )
        pixel_values = np.concatenate(
            [np.asarray(item).reshape(-1) for item in val["maps"]]
        )
        pixel_threshold = select_threshold(
            [0] * len(pixel_values),
            pixel_values,
            quantile=float(config.get("threshold_quantile", 0.995)),
        )
        model.save_checkpoint(
            target,
            category=category,
            dataset=dataset_name,
            image_threshold=image_threshold,
            pixel_threshold=pixel_threshold,
            threshold_source="normal validation quantile",
        )
        summary_path = target.with_name("training_metadata.json")
        summary_path.write_text(
            json.dumps(
                {
                    "model": model.metadata(),
                    "dataset": dataset_name,
                    "category": category,
                    "training_samples": len(data.split_records["train"]),
                    "validation_samples": len(data.split_records["validation"]),
                    "test_samples_not_used_during_fit": len(data.split_records["test"]),
                    "image_threshold": image_threshold,
                    "pixel_threshold": pixel_threshold,
                    "threshold_source": "normal validation quantile",
                    "checkpoint": str(target.relative_to(project_root))
                    if target.is_relative_to(project_root)
                    else target.name,
                    "training_duration_seconds": time.perf_counter() - training_started,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"Saved {model_name} {dataset_name}/{category} checkpoint: {target}")


if __name__ == "__main__":
    main()
