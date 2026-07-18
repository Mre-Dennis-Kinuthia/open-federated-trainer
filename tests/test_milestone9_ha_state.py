"""Milestone 9: shared HA state across logical replicas (SQLite)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


def _ensure_path() -> None:
    if str(COORD_SRC) not in sys.path:
        sys.path.insert(0, str(COORD_SRC))


def _sql_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("METADATA_BACKEND", "sqlite")
    monkeypatch.setenv("SHARED_STATE", "auto")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'ha.db'}")
    _ensure_path()
    from persistence.db import create_all_tables, reset_engine

    reset_engine()
    create_all_tables()


def test_reputation_shared_across_instances(tmp_path, monkeypatch):
    _sql_backend(tmp_path, monkeypatch)
    from core.reputation import ReputationManager
    from persistence.ha_repos import SqlReputationRepository

    repo = SqlReputationRepository()
    a = ReputationManager(repo=repo)
    a.register_client("c1")
    a.record_round_participation("c1", 1)
    a.record_update_accepted("c1", 1)

    b = ReputationManager(repo=repo)
    assert b.get_reputation("c1") is not None
    assert b.get_reputation("c1").updates_accepted >= 1
    assert "c1" in b.get_all_reputations()


def test_incentives_and_geo_shared(tmp_path, monkeypatch):
    _sql_backend(tmp_path, monkeypatch)
    from core.geo_presence import GeoPresence
    from core.incentives import IncentiveManager
    from persistence.ha_repos import SqlGeoPresenceRepository, SqlIncentiveRepository

    irepo = SqlIncentiveRepository()
    grepo = SqlGeoPresenceRepository()
    a = IncentiveManager(repo=irepo)
    tokens = a.award_update_reward("worker-1", round_id=3, latency_seconds=1.0)
    assert tokens > 0

    b = IncentiveManager(repo=irepo)
    assert b.get_client_balance("worker-1") == tokens

    geo_a = GeoPresence(state_path=str(tmp_path / "geo.json"), repo=grepo)
    geo_a.record("n1", ip=None)
    with geo_a._lock:
        geo_a._clients["n1"]["lat"] = 1.0
        geo_a._clients["n1"]["lng"] = 2.0
        geo_a._clients["n1"]["city"] = "Test"
        geo_a._clients["n1"]["country"] = "TZ"
    geo_a._persist(force=True)

    geo_b = GeoPresence(state_path=str(tmp_path / "geo2.json"), repo=grepo)
    snap = geo_b.snapshot()
    assert any(n.get("city") == "Test" for n in snap)


def test_round_assignment_visible_across_managers(tmp_path, monkeypatch):
    _sql_backend(tmp_path, monkeypatch)
    from core.round_manager import RoundManager, RoundState
    from persistence.json_repos import get_round_repository

    repo = get_round_repository()
    rm_a = RoundManager(round_repo=repo)
    rm_a.register_client("alice")
    rid = rm_a.assign_client_to_round("alice", "v1")
    assert rid is not None

    rm_b = RoundManager(round_repo=repo)
    status = rm_b.get_round_status(rid)
    assert status is not None
    assert "alice" in status["assigned_clients"]
    assert status["state"] in (RoundState.COLLECTING.value, RoundState.OPEN.value)


def test_aggregate_lock_single_winner(tmp_path, monkeypatch):
    _sql_backend(tmp_path, monkeypatch)
    from core.round_manager import RoundManager, RoundState
    from persistence.json_repos import get_round_repository

    repo = get_round_repository()
    rm = RoundManager(round_repo=repo)
    rm.register_client("a")
    rm.register_client("b")
    rid = rm.assign_client_to_round("a", "v1")
    rm.assign_client_to_round("b", "v1")
    assert rid is not None
    rm.add_update("a", rid, "{}")
    rm.add_update("b", rid, "{}")

    first = rm.try_begin_aggregating(rid)
    second = rm.try_begin_aggregating(rid)
    assert first is True
    assert second is False
    assert rm.refresh_round(rid).state == RoundState.AGGREGATING


def test_replica_kill_simulation_continues_round(tmp_path, monkeypatch):
    _sql_backend(tmp_path, monkeypatch)
    from core.round_manager import RoundManager
    from persistence.json_repos import get_round_repository

    repo = get_round_repository()
    rm_a = RoundManager(round_repo=repo)
    rm_a.register_client("c1")
    rid = rm_a.assign_client_to_round("c1", "v1")
    # "kill" replica A
    del rm_a

    rm_b = RoundManager(round_repo=repo)
    rm_b.register_client("c1")  # may already exist in memory empty set
    # clients set is process-local; re-register for assignment APIs
    status = rm_b.get_round_status(rid)
    assert status is not None
    assert "c1" in status["assigned_clients"]
    rm_b.clients.add("c1")
    assert rm_b.add_update("c1", rid, "{}") is True
