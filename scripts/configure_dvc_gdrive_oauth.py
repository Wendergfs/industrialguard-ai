"""Store a custom Google OAuth client in ignored, local DVC configuration."""

from __future__ import annotations

import getpass
import subprocess
from pathlib import Path

from configobj import ConfigObj


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / ".dvc" / "config.local"
    relative_config_path = config_path.relative_to(project_root).as_posix()

    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", relative_config_path],
        cwd=project_root,
        check=False,
    )
    if ignored.returncode != 0:
        raise SystemExit(
            "Refusing to write credentials: .dvc/config.local is not Git-ignored."
        )

    client_id = input("Google OAuth Desktop client ID: ").strip()
    client_secret = getpass.getpass(
        "Google OAuth Desktop client secret (input hidden): "
    ).strip()
    if not client_id or not client_secret:
        raise SystemExit("Both values are required; no configuration was changed.")

    config = ConfigObj(str(config_path), encoding="utf-8")
    remote = config.setdefault('remote "gdrive"', {})
    if not isinstance(remote, dict):
        raise SystemExit(
            "Unexpected local DVC configuration structure; no configuration was changed."
        )

    remote["gdrive_client_id"] = client_id
    remote["gdrive_client_secret"] = client_secret
    config.write()

    print(
        "Custom OAuth client saved to ignored .dvc/config.local; values were not displayed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
