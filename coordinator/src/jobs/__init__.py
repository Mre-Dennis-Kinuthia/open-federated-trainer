"""
General job queue for non-training volunteer/edge work.

Job types:
  - train     (legacy FL — still via rounds)
  - inference — run model on private inputs, return predictions
  - label     — human/auto labeling task chunk
  - compute   — Folding@home-style science chunk (arbitrary payload)
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class JobType(str, Enum):
    INFERENCE = "inference"
    LABEL = "label"
    COMPUTE = "compute"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class Job:
    job_id: str
    job_type: str
    payload: Dict[str, Any]
    state: str = JobState.QUEUED.value
    created_at: float = field(default_factory=time.time)
    assigned_client: Optional[str] = None
    assigned_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobQueue:
    """In-memory job queue with lease-style assignment."""

    def __init__(self, lease_seconds: float = 300.0):
        self.jobs: Dict[str, Job] = {}
        self.lease_seconds = lease_seconds
        self._lock = threading.RLock()

    def create_job(
        self,
        job_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        tags: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> Job:
        with self._lock:
            jid = job_id or str(uuid.uuid4())
            job = Job(
                job_id=jid,
                job_type=job_type,
                payload=payload or {},
                priority=priority,
                tags=tags or [],
            )
            self.jobs[jid] = job
            return job

    def _reclaim_expired(self) -> None:
        now = time.time()
        for job in self.jobs.values():
            if (
                job.state == JobState.ASSIGNED.value
                and job.assigned_at
                and now - job.assigned_at > self.lease_seconds
            ):
                job.state = JobState.QUEUED.value
                job.assigned_client = None
                job.assigned_at = None

    def claim_next(
        self,
        client_id: str,
        job_types: Optional[Set[str]] = None,
    ) -> Optional[Job]:
        with self._lock:
            self._reclaim_expired()
            candidates = [
                j
                for j in self.jobs.values()
                if j.state == JobState.QUEUED.value
                and (not job_types or j.job_type in job_types)
            ]
            if not candidates:
                return None
            candidates.sort(key=lambda j: (-j.priority, j.created_at))
            job = candidates[0]
            job.state = JobState.ASSIGNED.value
            job.assigned_client = client_id
            job.assigned_at = time.time()
            return job

    def submit_result(
        self,
        job_id: str,
        client_id: str,
        result: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            if job.assigned_client and job.assigned_client != client_id:
                return None
            if success:
                job.state = JobState.COMPLETED.value
                job.result = result
                job.error = None
            else:
                job.state = JobState.FAILED.value
                job.error = error or "failed"
                job.result = result
            job.completed_at = time.time()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        state: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self.jobs.values())
            if state:
                items = [j for j in items if j.state == state]
            if job_type:
                items = [j for j in items if j.job_type == job_type]
            items.sort(key=lambda j: j.created_at, reverse=True)
            return [j.to_dict() for j in items[:limit]]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for j in self.jobs.values():
                counts[j.state] = counts.get(j.state, 0) + 1
                counts[f"type:{j.job_type}"] = counts.get(f"type:{j.job_type}", 0) + 1
            return {"total": len(self.jobs), "counts": counts}


_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue()
    return _queue
