"""Write portable baseline environment and configuration fingerprints."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_benchmark import ROOT, environment_metadata


def main() -> None:
    configs = [
        ROOT / "configs" / "dataset.yaml",
        ROOT / "configs" / "experiment.yaml",
        ROOT / "configs" / "benchmark.yaml",
    ]
    configs.extend(sorted((ROOT / "configs" / "models").glob("*.yaml")))
    output = ROOT / "reports" / "reproducibility" / "environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    import json

    output.write_text(
        json.dumps(environment_metadata(configs, 42), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote reproducibility metadata: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
