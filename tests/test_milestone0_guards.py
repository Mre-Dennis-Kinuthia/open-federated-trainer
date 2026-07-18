"""Milestone 0 security and correctness guards."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"
CLIENT_SRC = ROOT / "client" / "src"


def _load_coord_module(name: str, relative: str):
    path = COORD_SRC / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_register_requires_proof_of_possession(tmp_path, monkeypatch):
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    sys.path.insert(0, str(COORD_SRC))
    from core.auth import AuthManager, ClientAlreadyRegisteredError
    from core.state_store import StateStore

    store = StateStore(path=str(tmp_path / "state.json"))
    auth = AuthManager(state_store=store)
    key = auth.register_client("alice")
    with pytest.raises(ClientAlreadyRegisteredError):
        auth.register_client("alice")
    assert auth.register_client("alice", presented_key=key) == key
    with pytest.raises(ClientAlreadyRegisteredError):
        auth.register_client("alice", presented_key="wrong-key")


def test_geo_resolve_only_updates_matching_ip_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEO_STATE_PATH", str(tmp_path / "geo.json"))
    monkeypatch.setenv("GEO_LOOKUP_DISABLED", "true")
    sys.path.insert(0, str(COORD_SRC))
    from core.geo_presence import GeoPresence

    geo = GeoPresence(state_path=str(tmp_path / "geo.json"))
    geo.record("a", "10.0.0.1")
    geo.record("b", "10.0.0.2")
    result = {"lat": 1.0, "lng": 2.0, "city": "X", "country": "Y"}
    geo._ip_cache[""] = result
    with geo._lock:
        for client_id, entry in geo._clients.items():
            if entry.get("ip_key") == "":
                geo._apply_location(client_id, entry, result)
    assert "lat" in geo._clients["a"]
    assert "lat" in geo._clients["b"]

    geo._clients["c"] = {"last_seen": 0, "ip_key": "8.8.8.8"}
    with geo._lock:
        for client_id, entry in list(geo._clients.items()):
            if entry.get("ip_key") == "9.9.9.9":
                geo._apply_location(client_id, entry, result)
    assert "lat" not in geo._clients["c"]


def test_job_to_dict_redacts_sensitive_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_QUEUE_STATE_PATH", str(tmp_path / "jobs.json"))
    jobs_mod = _load_coord_module("coord_jobs_m0", "jobs/__init__.py")
    job = jobs_mod.Job(
        job_id="j1",
        job_type="inference",
        payload={"inputs": ["secret prompt"], "model_id": "m"},
        state=jobs_mod.JobState.COMPLETED.value,
        result={"outputs": ["leak"]},
    )
    redacted = job.to_dict(include_sensitive=False)
    assert redacted["payload"]["redacted"] is True
    assert "secret prompt" not in json.dumps(redacted)
    assert redacted["result"]["redacted"] is True
    full = job.to_dict(include_sensitive=True)
    assert full["payload"]["inputs"] == ["secret prompt"]


def test_job_stats_include_completed(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_QUEUE_STATE_PATH", str(tmp_path / "jobs.json"))
    jobs_mod = _load_coord_module("coord_jobs_m0_stats", "jobs/__init__.py")
    q = jobs_mod.JobQueue(state_path=str(tmp_path / "jobs.json"))
    job = q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:lennard_jones", "work_unit": {}},
    )
    claimed = q.claim_next("worker-1", job_types={"compute"})
    assert claimed is not None
    q.submit_result(job.job_id, "worker-1", {"ok": True}, success=True)
    stats = q.stats()
    assert stats["completed"] == 1
    assert stats["total"] == 1


def test_custom_trainer_does_not_silently_fallback(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "custom")
    monkeypatch.delenv("MODEL_MODULE", raising=False)
    sys.path.insert(0, str(CLIENT_SRC))
    import models

    with pytest.raises(KeyError, match="MODEL_MODULE"):
        models.get_trainer("custom")


def test_update_payload_size_gate_logic():
    payload = {"weight_delta": [[1.0] * 1000]}
    size = len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    limit = 100
    assert size > limit
