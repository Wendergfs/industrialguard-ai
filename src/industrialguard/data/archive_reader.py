"""Safe, read-only access to ZIP and TAR.XZ members."""

from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO


class ArchiveReader:
    """Read archive members without extracting or modifying the archive."""

    def __init__(self, path: str | Path, archive_format: str):
        self.path = Path(path)
        self.archive_format = archive_format.lower()
        self.archive_member_count = 0
        self.directory_member_count = 0

    def names(self) -> list[str]:
        if self.archive_format == "zip":
            with zipfile.ZipFile(self.path, "r") as archive:
                items = archive.infolist()
                self.archive_member_count = len(items)
                self.directory_member_count = sum(item.is_dir() for item in items)
                return [item.filename for item in items if not item.is_dir()]
        if self.archive_format in {"tar.xz", "tar_xz"}:
            with tarfile.open(self.path, "r:xz") as archive:
                names = []
                for item in archive:
                    self.archive_member_count += 1
                    if item.isdir():
                        self.directory_member_count += 1
                    elif item.isfile():
                        names.append(item.name)
                return names
        raise ValueError(f"Unsupported archive format: {self.archive_format}")

    def read_bytes(self, member: str) -> bytes:
        if self.archive_format == "zip":
            with zipfile.ZipFile(self.path, "r") as archive:
                return archive.read(member)
        if self.archive_format in {"tar.xz", "tar_xz"}:
            with tarfile.open(self.path, "r:xz") as archive:
                stream = archive.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(f"No regular file member named {member!r}")
                with stream:
                    return stream.read()
        raise ValueError(f"Unsupported archive format: {self.archive_format}")

    def iter_member_bytes(self) -> Iterator[tuple[str, bytes]]:
        """Yield file contents one at a time in archive order."""
        if self.archive_format == "zip":
            with zipfile.ZipFile(self.path, "r") as archive:
                for zip_item in archive.infolist():
                    if not zip_item.is_dir():
                        yield zip_item.filename, archive.read(zip_item)
            return
        if self.archive_format in {"tar.xz", "tar_xz"}:
            with tarfile.open(self.path, "r|xz") as archive:
                for tar_item in archive:
                    if not tar_item.isfile():
                        continue
                    stream = archive.extractfile(tar_item)
                    if stream is not None:
                        with stream:
                            yield tar_item.name, stream.read()
            return
        raise ValueError(f"Unsupported archive format: {self.archive_format}")

    def open_member(self, member: str) -> BinaryIO:
        """Return an in-memory stream for a member, suitable for Pillow."""
        return io.BytesIO(self.read_bytes(member))

    def test_integrity(self) -> str | None:
        if self.archive_format == "zip":
            with zipfile.ZipFile(self.path, "r") as archive:
                return archive.testzip()
        if self.archive_format in {"tar.xz", "tar_xz"}:
            # Iterating forces the XZ stream and tar headers to be read.
            with tarfile.open(self.path, "r|xz") as archive:
                for _ in archive:
                    pass
            return None
        raise ValueError(f"Unsupported archive format: {self.archive_format}")
