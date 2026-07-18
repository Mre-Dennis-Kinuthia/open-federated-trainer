"""Milestone 8: S3/MinIO artifact store + production deploy helpers."""

from __future__ import annotations

import io
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


def _ensure_path() -> None:
    if str(COORD_SRC) not in sys.path:
        sys.path.insert(0, str(COORD_SRC))


class _FakeBody:
    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """Minimal S3 client for unit tests (no network)."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body, **_kwargs) -> dict:
        if hasattr(Body, "read"):
            Body = Body.read()
        self.objects[(Bucket, Key)] = bytes(Body)
        return {"ETag": "fake"}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"Body": _FakeBody(self.objects[(Bucket, Key)])}

    def head_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[(Bucket, Key)])}


def test_s3_object_key_layout():
    _ensure_path()
    from artifacts.s3_store import s3_object_key

    h = "abcdef" + "0" * 58
    assert s3_object_key(h) == f"sha256/ab/cd/{h}"
    assert s3_object_key(h, prefix="artifacts") == f"artifacts/sha256/ab/cd/{h}"


def test_s3_artifact_store_round_trip(tmp_path, monkeypatch):
    _ensure_path()
    monkeypatch.setenv("ARTIFACT_STORE", "s3")
    from artifacts.s3_store import S3ArtifactStore

    fake = FakeS3Client()
    store = S3ArtifactStore(
        bucket="test-bucket",
        prefix="artifacts",
        endpoint_url="http://minio:9000",
        client=fake,
        cache_dir=str(tmp_path / "cache"),
    )
    h1 = store.put_bytes(b"prod-model-bytes")
    h2 = store.put_bytes(b"prod-model-bytes")
    assert h1 == h2
    assert store.exists(h1)
    assert not store.exists("f" * 64)
    assert store.open_read(h1).read() == b"prod-model-bytes"
    assert store.uri_for(h1).startswith("s3://test-bucket/artifacts/sha256/")
    local = store.local_path(h1)
    assert local.read_bytes() == b"prod-model-bytes"


def test_get_artifact_store_s3(monkeypatch):
    _ensure_path()
    monkeypatch.setenv("ARTIFACT_STORE", "s3")
    monkeypatch.setenv("S3_BUCKET", "b")
    # Avoid constructing real boto3 client — patch after import
    from artifacts import get_artifact_store
    from artifacts.s3_store import S3ArtifactStore

    store = get_artifact_store()
    assert isinstance(store, S3ArtifactStore)


def test_get_artifact_store_unknown(monkeypatch):
    _ensure_path()
    monkeypatch.setenv("ARTIFACT_STORE", "gcs")
    from artifacts import get_artifact_store

    try:
        get_artifact_store()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Unknown ARTIFACT_STORE" in str(exc)


def test_deploy_assets_exist():
    assert (ROOT / "docker-compose.prod.yml").is_file()
    assert (ROOT / "deploy" / "nginx" / "nginx.conf").is_file()
    assert (ROOT / "deploy" / "helm" / "fed-compute" / "Chart.yaml").is_file()
    assert (ROOT / "scripts" / "backup.sh").is_file()
    assert (ROOT / "scripts" / "restore.sh").is_file()
    assert (ROOT / "docs" / "deploy" / "LOAD_AND_FAULT_REPORT.md").is_file()
    assert (ROOT / "docs" / "deploy" / "PRODUCTION_DEPLOY.md").is_file()
