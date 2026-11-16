"""Lazy, integrity-checked cache for selected archive members."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from .archive_reader import ArchiveReader


def sha256_file(path: Path) -> str:
    """Hash a file incrementally without loading it fully into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class CacheManager:
    """Materialize only requested image and mask members under a cache root."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)

    @staticmethod
    def _safe_relative(member: str, expected_root: str | None) -> Path:
        posix = PurePosixPath(member)
        if (
            posix.is_absolute()
            or ".." in posix.parts
            or "\\" in member
            or ":" in member
            or "\x00" in member
        ):
            raise ValueError(f"Unsafe archive member path: {member}")
        parts = posix.parts
        if expected_root and parts and parts[0] == expected_root:
            parts = parts[1:]
        if not parts:
            raise ValueError(f"Archive member has no cache-relative path: {member}")
        return Path(*parts)

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=".cache-tmp-", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def materialize(
        self,
        reader: ArchiveReader,
        dataset: str,
        source_archive: str | Path,
        members: Iterable[str],
        expected_root: str | None = None,
    ) -> dict[str, Path]:
        """Cache and verify requested members; archives remain read-only.

        TAR.XZ is streamed once when any requested item is missing. ZIP members
        are read individually. A manifest is bound to archive path, size, and mtime.
        """
        source = Path(source_archive).resolve()
        stat = source.stat()
        signature = hashlib.sha256(
            f"{source}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        ).hexdigest()[:20]
        root = self.cache_dir / dataset / signature
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {
                "source": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "files": {},
            }
        if (manifest.get("source"), manifest.get("size"), manifest.get("mtime_ns")) != (
            str(source),
            stat.st_size,
            stat.st_mtime_ns,
        ):
            manifest = {
                "source": str(source),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "files": {},
            }
        entries: dict[str, str] = manifest.setdefault("files", {})
        unique_members = list(dict.fromkeys(members))
        member_paths = {
            member: root / self._safe_relative(member, expected_root)
            for member in unique_members
        }
        root_resolved = root.resolve()
        for member, target in member_paths.items():
            if not target.resolve().is_relative_to(root_resolved):
                raise ValueError(f"Cache path escapes configured root: {member}")
        missing = []
        for member, target in member_paths.items():
            expected_hash = entries.get(member)
            if (
                not target.is_file()
                or not expected_hash
                or sha256_file(target) != expected_hash
            ):
                missing.append(member)
        if missing:
            if reader.archive_format == "zip":
                for member in missing:
                    payload = reader.read_bytes(member)
                    target = member_paths[member]
                    self._atomic_write(target, payload)
                    entries[member] = hashlib.sha256(payload).hexdigest()
            else:
                pending = set(missing)
                for member, payload in reader.iter_member_bytes():
                    if member in pending:
                        target = member_paths[member]
                        self._atomic_write(target, payload)
                        entries[member] = hashlib.sha256(payload).hexdigest()
                        pending.remove(member)
                if pending:
                    raise FileNotFoundError(
                        f"Archive members not found: {sorted(pending)[:5]}"
                    )
            encoded = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            self._atomic_write(manifest_path, encoded)
        # Verify on return, including files that were freshly written.
        for member, target in member_paths.items():
            expected_hash = entries.get(member)
            if (
                not expected_hash
                or not target.is_file()
                or sha256_file(target) != expected_hash
            ):
                raise OSError(f"Cached member failed SHA-256 verification: {member}")
        return member_paths
