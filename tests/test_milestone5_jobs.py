"""Milestone 5: JobSpec, leases, attempts, dataset aliases."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


@pytest.fixture()
def jobs_env(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_QUEUE_STATE_PATH", str(tmp_path / "jobs.json"))
    monkeypatch.setenv("DATASET_ALIAS_PATH", str(tmp_path / "aliases.json"))
    monkeypatch.setenv("JOB_LEASE_SECONDS", "2")
    sys.path.insert(0, str(COORD_SRC))
    from jobs import reset_job_queue_for_tests

    reset_job_queue_for_tests()
    yield tmp_path
    reset_job_queue_for_tests()


def test_jobspec_rejects_incomplete_inference(jobs_env):
    from jobs import JobQueue, validate_job_spec

    with pytest.raises(ValueError, match="dataset_alias"):
        validate_job_spec("inference", {})
    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=2)
    with pytest.raises(ValueError):
        q.create_job("inference", {})


def test_lease_expiry_requeues_then_dead_letters(jobs_env):
    from jobs import JobQueue, JobState

    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=1)
    job = q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:lennard_jones", "work_unit": {}},
        max_attempts=2,
        lease_seconds=1,
    )
    claimed = q.claim_next("w1", {"compute"})
    assert claimed and claimed.state == JobState.ASSIGNED.value
    assert claimed.lease_expires_at is not None
    assert claimed.attempts == 1

    # Force expiry
    claimed.lease_expires_at = time.time() - 10
    claimed.assigned_at = time.time() - 10
    q._persist()

    again = q.claim_next("w2", {"compute"})
    assert again is not None
    assert again.job_id == job.job_id
    assert again.assigned_client == "w2"
    assert again.attempts == 2
    assert again.attempt_history[-2]["outcome"] == "lease_expired"

    again.lease_expires_at = time.time() - 10
    again.assigned_at = time.time() - 10
    q._persist()
    # Max attempts reached → FAILED, not requeued
    none = q.claim_next("w3", {"compute"})
    assert none is None
    failed = q.get(job.job_id)
    assert failed.state == JobState.FAILED.value
    assert "maximum lease attempts" in (failed.error or "")


def test_extend_lease_prevents_reclaim(jobs_env):
    from jobs import JobQueue, JobState

    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=1)
    q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:x", "work_unit": {}},
        lease_seconds=1,
    )
    job = q.claim_next("w1", {"compute"})
    assert job is not None
    original_expiry = job.lease_expires_at
    time.sleep(0.2)
    extended = q.extend_lease(job.job_id, "w1", extend_seconds=5)
    assert extended is not None
    assert extended.lease_expires_at > original_expiry

    # Other worker cannot steal while lease valid
    assert q.claim_next("w2", {"compute"}) is None
    assert q.get(job.job_id).state == JobState.ASSIGNED.value
    assert q.get(job.job_id).assigned_client == "w1"


def test_cancel_queued_and_assigned(jobs_env):
    from jobs import JobQueue, JobState

    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=30)
    queued = q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:x", "work_unit": {}},
    )
    assert q.cancel(queued.job_id).state == JobState.CANCELLED.value
    assert q.claim_next("w1", {"compute"}) is None

    active = q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:y", "work_unit": {}},
    )
    claimed = q.claim_next("w1", {"compute"})
    assert claimed is not None
    cancelled = q.cancel(active.job_id)
    assert cancelled.state == JobState.CANCELLED.value
    assert q.submit_result(active.job_id, "w1", {"x": 1}) is None


def test_dataset_alias_registry_and_job(jobs_env):
    from jobs import DatasetAliasRegistry, JobQueue

    aliases = DatasetAliasRegistry(state_path=str(jobs_env / "aliases.json"))
    aliases.upsert("reviews", description="local review corpus", format_hint="jsonl")
    q = JobQueue(
        state_path=str(jobs_env / "jobs2.json"),
        lease_seconds=30,
        alias_registry=aliases,
    )
    with pytest.raises(ValueError, match="Unknown dataset_alias"):
        q.create_job("inference", {"dataset_alias": "missing", "model_id": "m"})
    job = q.create_job(
        "inference",
        {"dataset_alias": "reviews", "model_id": "sshleifer/tiny-gpt2"},
    )
    assert job.payload["dataset_alias"] == "reviews"
    assert aliases.list_aliases()[0]["alias"] == "reviews"


def test_failed_result_requeues_until_max(jobs_env):
    from jobs import JobQueue, JobState

    q = JobQueue(state_path=str(jobs_env / "jobs.json"), lease_seconds=30)
    job = q.create_job(
        "compute",
        {"entrypoint": "examples.science_plugin:x", "work_unit": {}},
        max_attempts=2,
    )
    q.claim_next("w1", {"compute"})
    requeued = q.submit_result(job.job_id, "w1", {"error": "boom"}, success=False)
    assert requeued.state == JobState.QUEUED.value
    q.claim_next("w2", {"compute"})
    failed = q.submit_result(job.job_id, "w2", {"error": "boom"}, success=False)
    assert failed.state == JobState.FAILED.value
