"""Build a small cache and print one CPU batch from a configured dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.data_module import IndustrialDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("mpdd", "mvtec"), default=None)
    parser.add_argument("--category", action="append", dest="categories")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-per-split", type=int, default=4)
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()
    module = IndustrialDataModule(
        args.dataset,
        args.categories,
        dataset_config=args.dataset_config,
        experiment_config=args.experiment_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_per_split=args.limit_per_split,
        cache_dir=args.cache_dir,
    )
    module.setup()
    batch = next(iter(module.train_dataloader()))
    print(f"dataset={module.dataset_name} categories={module.categories}")
    print(
        f"split_sizes={ {name: len(items) for name, items in module.split_records.items()} }"
    )
    print(
        f"image_shape={tuple(batch['image'].shape)} mask_shape={tuple(batch['mask'].shape)}"
    )
    print(f"labels={batch['label'].tolist()} device={batch['image'].device.type}")
    print(f"sample_ids={batch['sample_id']}")


if __name__ == "__main__":
    main()
