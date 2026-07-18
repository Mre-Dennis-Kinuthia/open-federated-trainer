"""Artifact store abstraction (content-addressed bytes)."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import BinaryIO, Optional, Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        """Store bytes; return content hash (sha256 hex)."""

    def put_file(self, path: Path) -> str:
        """Store file contents; return content hash."""

    def open_read(self, content_hash: str) -> BinaryIO:
        """Open stored object for reading."""

    def exists(self, content_hash: str) -> bool: ...

    def local_path(self, content_hash: str) -> Path:
        """Filesystem path for local adapters (dev)."""

    def uri_for(self, content_hash: str) -> str:
        """Storage URI (file:// or s3://)."""


class LocalFilesystemArtifactStore:
    """Content-addressed store under coordinator/artifacts/sha256/ab/cd/<hash>."""

    def __init__(self, root: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "artifacts"
        self.root = Path(root or os.getenv("ARTIFACT_STORE_ROOT", str(default)))
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, content_hash: str) -> Path:
        h = content_hash.lower()
        if len(h) < 4 or any(c not in "0123456789abcdef" for c in h):
            raise ValueError(f"Invalid content hash: {content_hash}")
        return self.root / "sha256" / h[:2] / h[2:4] / h

    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        content_hash = hashlib.sha256(data).hexdigest()
        path = self._path_for(content_hash)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
        return content_hash

    def put_file(self, path: Path) -> str:
        return self.put_bytes(Path(path).read_bytes())

    def open_read(self, content_hash: str) -> BinaryIO:
        return self._path_for(content_hash).open("rb")

    def exists(self, content_hash: str) -> bool:
        return self._path_for(content_hash).exists()

    def local_path(self, content_hash: str) -> Path:
        return self._path_for(content_hash)

    def uri_for(self, content_hash: str) -> str:
        return self._path_for(content_hash).resolve().as_uri()


def get_artifact_store() -> ArtifactStore:
    kind = os.getenv("ARTIFACT_STORE", "local").strip().lower() or "local"
    if kind in ("s3", "minio"):
        from .s3_store import S3ArtifactStore

        return S3ArtifactStore()
    if kind != "local":
        raise RuntimeError(
            f"Unknown ARTIFACT_STORE={kind!r}; use local, s3, or minio."
        )
    return LocalFilesystemArtifactStore()
