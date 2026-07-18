"""Milestone 6: compute runtime, entrypoint hardening, canaries, N-of-M."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_SRC = ROOT / "client" / "src"
COORD_SRC = ROOT / "coordinator" / "src"


def _load_client_runtime():
    path = CLIENT_SRC / "runtime" / "__init__.py"
    spec = importlib.util.spec_from_file_location("client_runtime_m6", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_client_jobs():
    # Ensure runtime package is importable as "runtime"
    sys.path.insert(0, str(CLIENT_SRC))
    path = CLIENT_SRC / "jobs" / "__init__.py"
    spec = importlib.util.spec_from_file_location("client_jobs_m6", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def jobs_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_QUEUE_STATE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("DATASET_ALIAS_PATH", str(tmp_path / "aliases.json"))
    sys.path.insert(0, str(COORD_SRC))
    from jobs import reset_job_queue_for_tests

    reset_job_queue_for_tests()
    yield tmp_path
    reset_job_queue_for_tests()


def test_entrypoint_rejects_paths_and_urls():
    rt = _load_client_runtime()
    with pytest.raises(rt.EntrypointRejected):
        rt.parse_entrypoint("../os:system")
    with pytest.raises(rt.EntrypointRejected):
        rt.parse_entrypoint("/tmp/evil:run")
    with pytest.raises(rt.EntrypointRejected):
        rt.parse_entrypoint("https://evil.example/mod:run")
    # Valid identifier form still must pass allowlist separately
    assert rt.parse_entrypoint("os:system") == ("os", "system")
    # Valid form
    assert rt.parse_entrypoint("examples.science_plugin:lennard_jones") == (
        "examples.science_plugin",
        "lennard_jones",
    )


def test_empty_allowlist_refuses_all(monkeypatch):
    rt = _load_client_runtime()
    monkeypatch.delenv("COMPUTE_PLUGIN_ALLOWLIST", raising=False)
    with pytest.raises(rt.EntrypointRejected, match="empty"):
        rt.assert_allowlisted("examples.science_plugin")


def test_non_allowlisted_module_refused(monkeypatch):
    rt = _load_client_runtime()
    monkeypatch.setenv("COMPUTE_PLUGIN_ALLOWLIST", "examples.science_plugin")
    with pytest.raises(rt.EntrypointRejected, match="not allowlisted"):
        rt.assert_allowlisted("evil.pkg")


def test_local_import_runtime_runs_lj(monkeypatch):
    monkeypatch.setenv("COMPUTE_PLUGIN_ALLOWLIST", "examples.science_plugin")
    sys.path.insert(0, str(ROOT / "client"))
    rt = _load_client_runtime()
    runtime = rt.LocalImportRuntime()
    out = runtime.execute(
        "examples.science_plugin:lennard_jones",
        {"positions": [[0, 0, 0], [1.2, 0, 0]], "steps": 2},
    )
    assert out["runtime"] == "local_import"
    assert out["result"]["particle_count"] == 2


def test_container_runtime_stub(monkeypatch):
    rt = _load_client_runtime()
    monkeypatch.setenv("COMPUTE_RUNTIME", "container")
    runtime = rt.get_compute_runtime()
    with pytest.raises(rt.RuntimeError_, match="not wired"):
        runtime.execute("examples.science_plugin:lennard_jones", {})


def test_canary_pass_and_fail(jobs_env):
    from jobs import JobQueue, JobState
    from jobs.verification import result_fingerprint

    expected = {"particle_count": 2, "ok": True}
    fp = result_fingerprint(expected)
    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=30)
    job = q.create_job(
        "compute",
        {
            "entrypoint": "examples.science_plugin:lennard_jones",
            "work_unit": {},
            "verification": {
                "mode": "canary",
                "expected_fingerprint": fp,
            },
        },
    )
    q.claim_next("w1", {"compute"})
    ok = q.submit_result(
        job.job_id,
        "w1",
        {"result": expected},
        success=True,
    )
    assert ok.state == JobState.COMPLETED.value
    assert ok.validation["status"] == "passed"

    job2 = q.create_job(
        "compute",
        {
            "entrypoint": "examples.science_plugin:lennard_jones",
            "work_unit": {},
            "canary": True,
            "expected_fingerprint": fp,
        },
    )
    q.claim_next("w2", {"compute"})
    bad = q.submit_result(
        job2.job_id,
        "w2",
        {"result": {"particle_count": 999}},
        success=True,
    )
    assert bad.state == JobState.FAILED.value
    assert bad.validation["status"] == "failed"


def test_n_of_m_quorum(jobs_env):
    from jobs import JobQueue, JobState

    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=30)
    job = q.create_job(
        "compute",
        {
            "entrypoint": "examples.science_plugin:lennard_jones",
            "work_unit": {},
            "verification": {"mode": "n_of_m", "n": 2, "m": 3},
        },
        max_attempts=5,
    )
    q.claim_next("w1", {"compute"})
    mid = q.submit_result(
        job.job_id, "w1", {"result": {"answer": 42}}, success=True
    )
    assert mid.state == JobState.QUEUED.value
    assert mid.validation["status"] == "pending"

    q.claim_next("w2", {"compute"})
    done = q.submit_result(
        job.job_id, "w2", {"result": {"answer": 42}}, success=True
    )
    assert done.state == JobState.COMPLETED.value
    assert done.validation["status"] == "agreed"
    assert done.validation["agreed_fingerprint"]


def test_coordinator_rejects_path_entrypoint(jobs_env):
    from jobs import validate_job_spec

    with pytest.raises(ValueError, match="path"):
        validate_job_spec(
            "compute",
            {"entrypoint": "../evil:run", "work_unit": {}},
        )
