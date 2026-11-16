"""File storage behind an interface, so S3 / Cloudflare R2 can replace local disk later.

Keys are '/'-separated relative paths like "projects/<id>/clips/<cid>/clip.mp4".
The engine works on local files (FFmpeg needs a path), so every backend must be able to
give a local path for a key (`local_path`): the local backend returns the file itself; a
cloud backend would download to a cache first and upload on `put`.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Iterator

from mog.errors import MogError


class Storage(ABC):
    @abstractmethod
    def put_file(self, key: str, src: Path) -> None: ...

    @abstractmethod
    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> int:
        """Write a stream; raise MogError (and keep nothing) if it exceeds max_bytes. Returns size."""

    @abstractmethod
    def put_bytes(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get_bytes(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def local_path(self, key: str) -> Path: ...

    @abstractmethod
    def list(self, prefix: str) -> Iterator[str]:
        """Immediate children names under prefix (like `ls`)."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...

    @abstractmethod
    def size(self, key: str) -> int: ...


class LocalStorage(Storage):
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, key: str) -> Path:
        p = (self.root / key).resolve()
        if p != self.root and self.root not in p.parents:
            raise MogError("Invalid file key.")       # blocks "../" escapes
        return p

    def put_file(self, key: str, src: Path) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def put_stream(self, key: str, stream: BinaryIO, max_bytes: int) -> int:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        total = 0
        try:
            with open(tmp, "wb") as f:
                while chunk := stream.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise MogError(f"File is too large (limit {max_bytes / 1024**2:.0f} MB).")
                    f.write(chunk)
            os.replace(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
        return total

    def put_bytes(self, key: str, data: bytes) -> None:
        dst = self._p(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)

    def get_bytes(self, key: str) -> bytes:
        p = self._p(key)
        if not p.exists():
            raise MogError(f"Not found: {key}")
        return p.read_bytes()

    def exists(self, key: str) -> bool:
        return self._p(key).exists()

    def local_path(self, key: str) -> Path:
        return self._p(key)

    def list(self, prefix: str) -> Iterator[str]:
        p = self._p(prefix)
        if p.is_dir():
            yield from sorted(c.name for c in p.iterdir() if not c.name.endswith(".part"))

    def delete_prefix(self, prefix: str) -> None:
        p = self._p(prefix)
        if p == self.root:
            raise MogError("Refusing to delete the storage root.")
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink()

    def size(self, key: str) -> int:
        return self._p(key).stat().st_size


def default_storage() -> Storage:
    from mog.config import settings

    return LocalStorage(settings().data_dir / "files")
