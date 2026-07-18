"""Milestone 1: persistence interfaces and local artifact store."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


def _load(name: str, relative: str):
    path = COORD_SRC / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Ensure package parents exist for relative imports inside modules
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_local_artifact_store_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    sys.path.insert(0, str(COORD_SRC))
    from artifacts import LocalFilesystemArtifactStore

    store = LocalFilesystemArtifactStore(root=str(tmp_path / "artifacts"))
    h1 = store.put_bytes(b"hello-model")
    h2 = store.put_bytes(b"hello-model")
    assert h1 == h2
    assert store.exists(h1)
    assert store.open_read(h1).read() == b"hello-model"
    assert store.uri_for(h1).startswith("file:")


def test_json_artifact_repository_and_round_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_INDEX_PATH", str(tmp_path / "artifacts_index.json"))
    monkeypatch.setenv("CLASSIC_ROUNDS_PATH", str(tmp_path / "classic_rounds.json"))
    monkeypatch.setenv("METADATA_BACKEND", "json")
    sys.path.insert(0, str(COORD_SRC))
    from persistence import ArtifactRecord, RoundRecord
    from persistence.json_repos import (
        JsonArtifactRepository,
        JsonRoundRepository,
        metadata_backend,
    )

    assert metadata_backend() == "json"
    repo = JsonArtifactRepository(index_path=str(tmp_path / "artifacts_index.json"))
    rec = ArtifactRecord(
        artifact_id="global_model:v1:abc",
        artifact_type="global_model",
        content_hash="a" * 64,
        byte_size=12,
        storage_uri="file:///tmp/x",
        created_at=1.0,
    )
    repo.put_manifest(rec)
    assert repo.get_manifest(rec.artifact_id).content_hash == rec.content_hash
    assert repo.get_by_hash(rec.content_hash).artifact_id == rec.artifact_id

    rounds = JsonRoundRepository(rounds_path=str(tmp_path / "classic_rounds.json"))
    rounds.save_round(
        RoundRecord(round_id=7, state="COLLECTING", model_version="v1", assigned_clients=["c1"])
    )
    got = rounds.get_round(7)
    assert got is not None
    assert got.state == "COLLECTING"
    assert rounds.list_rounds(limit=10)[0].round_id == 7


def test_metadata_backend_sqlite_uses_sql_repos(tmp_path, monkeypatch):
    monkeypatch.setenv("METADATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'backend.db'}")
    sys.path.insert(0, str(COORD_SRC))
    from persistence.db import reset_engine
    from persistence.json_repos import get_artifact_repository
    from persistence.sql_repos import SqlArtifactRepository

    reset_engine()
    repo = get_artifact_repository()
    assert isinstance(repo, SqlArtifactRepository)
    monkeypatch.setenv("METADATA_BACKEND", "json")
    reset_engine()
