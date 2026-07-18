"""S3-compatible content-addressed artifact store (MinIO / AWS S3)."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Optional


def s3_object_key(content_hash: str, *, prefix: str = "") -> str:
    """Layout: [{prefix}/]sha256/ab/cd/<hash> — mirrors local FS store."""
    h = content_hash.lower()
    if len(h) < 4 or any(c not in "0123456789abcdef" for c in h):
        raise ValueError(f"Invalid content hash: {content_hash}")
    base = f"sha256/{h[:2]}/{h[2:4]}/{h}"
    prefix = prefix.strip("/")
    return f"{prefix}/{base}" if prefix else base


class S3ArtifactStore:
    """Content-addressed store backed by an S3-compatible bucket."""

    def __init__(
        self,
        *,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region: Optional[str] = None,
        client: Any = None,
        cache_dir: Optional[str] = None,
    ):
        self.bucket = bucket or os.getenv("S3_BUCKET", "fedcompute-artifacts")
        self.prefix = (
            prefix
            if prefix is not None
            else os.getenv("S3_PREFIX", "artifacts")
        )
        self.endpoint_url = endpoint_url or os.getenv("S3_ENDPOINT_URL") or None
        self.region = region or os.getenv("S3_REGION", "us-east-1")
        self._client = client
        default_cache = Path(tempfile.gettempdir()) / "fedcompute-artifact-cache"
        self.cache_dir = Path(
            cache_dir or os.getenv("ARTIFACT_CACHE_DIR", str(default_cache))
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "ARTIFACT_STORE=s3 requires boto3. "
                    "Install coordinator requirements or pip install boto3."
                ) from exc
            kwargs: dict[str, Any] = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url
            # Prefer explicit keys when set (MinIO / CI); else default chain.
            access = os.getenv("S3_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_ID")
            secret = os.getenv("S3_SECRET_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
            if access and secret:
                kwargs["aws_access_key_id"] = access
                kwargs["aws_secret_access_key"] = secret
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _key(self, content_hash: str) -> str:
        return s3_object_key(content_hash, prefix=self.prefix or "")

    def put_bytes(self, data: bytes, *, suffix: str = "") -> str:
        del suffix  # reserved for future content-type hints
        content_hash = hashlib.sha256(data).hexdigest()
        if self.exists(content_hash):
            return content_hash
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(content_hash),
            Body=data,
            ContentType="application/octet-stream",
            Metadata={"sha256": content_hash},
        )
        return content_hash

    def put_file(self, path: Path) -> str:
        return self.put_bytes(Path(path).read_bytes())

    def open_read(self, content_hash: str) -> BinaryIO:
        obj = self.client.get_object(Bucket=self.bucket, Key=self._key(content_hash))
        body = obj["Body"].read()
        return io.BytesIO(body)

    def exists(self, content_hash: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(content_hash))
            return True
        except Exception as exc:  # noqa: BLE001 — botocore ClientError + fakes
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            # botocore raises ClientError; missing key often has HTTPStatusCode 404
            status = getattr(exc, "response", {}).get("ResponseMetadata", {}).get(
                "HTTPStatusCode"
            )
            if status == 404:
                return False
            # In-memory fakes may raise KeyError / FileNotFoundError
            if isinstance(exc, (KeyError, FileNotFoundError)):
                return False
            name = type(exc).__name__
            if "ClientError" in name or "NoSuchKey" in name:
                return False
            raise

    def local_path(self, content_hash: str) -> Path:
        """Materialize object into a local cache for adapters that need a Path."""
        dest = self.cache_dir / content_hash
        if not dest.exists():
            data = self.open_read(content_hash).read()
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        return dest

    def uri_for(self, content_hash: str) -> str:
        key = self._key(content_hash)
        if self.endpoint_url:
            return f"s3://{self.bucket}/{key}?endpoint={self.endpoint_url}"
        return f"s3://{self.bucket}/{key}"
