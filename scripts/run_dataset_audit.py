"""Run the archive-only dataset audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrialguard.data.dataset_audit import run_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset.yaml")
    parser.add_argument("--output", default="reports/dataset_audit")
    args = parser.parse_args()
    summaries, validations, _ = run_audit(args.config, args.output)
    print(
        json.dumps(
            {
                name: {
                    "images": s["image_count"],
                    "normal": s["normal_images"],
                    "anomalous": s["anomalous_images"],
                    "masks": s["ground_truth_mask_members"],
                    "errors": len(validations[name]["errors"]),
                    "warnings": len(validations[name]["warnings"]),
                }
                for name, s in summaries.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
