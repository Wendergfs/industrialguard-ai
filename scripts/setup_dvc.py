"""Initialize local DVC metadata and version selected small protocol artifacts."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("DVC_NO_ANALYTICS", "true")
os.environ.setdefault("DVC_SITE_CACHE_DIR", str(ROOT / ".cache" / "dvc" / "site"))
TARGETS = (
    "configs",
    "reports/dataset_audit",
    "reports/model_development/experimental_protocol.md",
    "reports/reproducibility/environment.json",
)


def main() -> int:
    try:
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        print(
            "A Git repository is required. Run `git init` in the project first.",
            file=sys.stderr,
        )
        return 2
    config = ROOT / ".dvc" / "config"
    dvc_command = [sys.executable, "-m", "dvc"]
    if not config.exists():
        subprocess.run([*dvc_command, "init"], cwd=ROOT, check=True)
    if config.exists() and "[remote" in config.read_text(encoding="utf-8"):
        raise RuntimeError("A DVC remote already exists; setup refuses to modify it")
    for target in TARGETS:
        path = ROOT / target
        if not path.exists():
            raise FileNotFoundError(f"DVC target does not exist: {target}")
        subprocess.run([*dvc_command, "add", target], cwd=ROOT, check=True)
    print(
        "DVC local metadata created. No remote was configured and no dataset archives were added."
    )
    print(
        "Commit the generated .dvc files, .dvc/config, and DVC-created .gitignore files when ready."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
