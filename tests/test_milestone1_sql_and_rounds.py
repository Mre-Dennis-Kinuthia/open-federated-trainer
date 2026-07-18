"""Milestone 1: SQL metadata + durable classic rounds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


@pytest.fixture()
def sql_env(tmp_path, monkeypatch):
    db = tmp_path / "meta.sqlite"
    monkeypatch.setenv("METADATA_BACKEND", "sqlite")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db}")
    monkeypatch.setenv("CLASSIC_ROUNDS_PATH", str(tmp_path / "classic_rounds.json"))
    sys.path.insert(0, str(COORD_SRC))
    from persistence.db import reset_engine, create_all_tables

    reset_engine()
    create_all_tables()
    yield
    reset_engine()


def test_sql_round_and_artifact_repositories(sql_env):
    from persistence import ArtifactRecord, RoundRecord
    from persistence.json_repos import get_artifact_repository, get_round_repository

    rounds = get_round_repository()
    arts = get_artifact_repository()
    rounds.save_round(
        RoundRecord(
            round_id=3,
            state="COLLECTING",
            model_version="v2",
            assigned_clients=["a", "b"],
            updates_received=["a"],
        )
    )
    got = rounds.get_round(3)
    assert got is not None
    assert got.assigned_clients == ["a", "b"]
    assert rounds.list_rounds(limit=5)[0].round_id == 3

    arts.put_manifest(
        ArtifactRecord(
            artifact_id="m:v1:deadbeef",
            artifact_type="global_model",
            content_hash="b" * 64,
            byte_size=10,
            storage_uri="file:///tmp/x",
            created_at=1.0,
        )
    )
    assert arts.get_by_hash("b" * 64).artifact_id == "m:v1:deadbeef"


def test_round_manager_survives_restart_via_json_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("METADATA_BACKEND", "json")
    monkeypatch.setenv("CLASSIC_ROUNDS_PATH", str(tmp_path / "classic_rounds.json"))
    sys.path.insert(0, str(COORD_SRC))
    from core.round_manager import RoundManager, RoundState
    from persistence.json_repos import JsonRoundRepository

    repo = JsonRoundRepository(rounds_path=str(tmp_path / "classic_rounds.json"))
    mgr = RoundManager(round_repo=repo)
    mgr.register_client("c1")
    mgr.register_client("c2")
    rid = mgr.assign_client_to_round("c1", "v1")
    assert rid is not None
    mgr.assign_client_to_round("c2", "v1")
    assert mgr.add_update("c1", rid, "{}")
    mgr.set_round_state(rid, RoundState.AGGREGATING)

    # Simulate process restart — AGGREGATING without published_version
    # rolls back to COLLECTING for reconcile (Milestone 3).
    mgr2 = RoundManager(round_repo=JsonRoundRepository(rounds_path=str(tmp_path / "classic_rounds.json")))
    assert rid in mgr2.rounds
    restored = mgr2.rounds[rid]
    assert restored.state == RoundState.COLLECTING
    assert restored.metadata.get("resume_after_crash") is True
    assert "c1" in restored.updates_received
    assert "c2" in restored.assigned_clients
