"""
General job queue for non-training volunteer/edge work.

Milestone 5: JobSpec validation, lease extend/expiry, JobAttempt history,
and coordinator-side dataset alias registry (paths stay on workers).
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


class JobType(str, Enum):
    INFERENCE = "inference"
    LABEL = "label"
    COMPUTE = "compute"


class JobState(str, Enum):
    QUEUED = "QUEUED"
    ASSIGNED = "ASSIGNED"  # leased to a worker (domain: LEASED)
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass
class JobAttempt:
    attempt: int
    client_id: str
    started_at: float
    ended_at: Optional[float] = None
    outcome: Optional[str] = None
    lease_expires_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "JobAttempt":
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in allowed})


@dataclass
class JobSpec:
    """Validated create-time job specification."""

    job_type: str
    payload: Dict[str, Any]
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    max_attempts: int = 3
    lease_seconds: Optional[float] = None
    dataset_alias: Optional[str] = None


def validate_job_spec(
    job_type: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    priority: int = 0,
    tags: Optional[List[str]] = None,
    max_attempts: int = 3,
    lease_seconds: Optional[float] = None,
) -> JobSpec:
    """Validate and normalize a job create request into a JobSpec."""
    if job_type not in {item.value for item in JobType}:
        raise ValueError(f"Unsupported job type: {job_type}")
    body = dict(payload or {})
    if not isinstance(body, dict):
        raise ValueError("payload must be an object")

    dataset_alias = body.get("dataset_alias")
    if dataset_alias is not None:
        if not isinstance(dataset_alias, str) or not dataset_alias.strip():
            raise ValueError("payload.dataset_alias must be a non-empty string")
        body["dataset_alias"] = dataset_alias.strip()

    if job_type == JobType.INFERENCE.value:
        has_inputs = isinstance(body.get("inputs"), list) and len(body["inputs"]) > 0
        has_local = bool(body.get("dataset_path") or body.get("dataset_alias"))
        if not has_inputs and not has_local:
            raise ValueError(
                "inference JobSpec requires payload.inputs or "
                "payload.dataset_alias / payload.dataset_path"
            )
        if not (body.get("model_id") or os.getenv("INFERENCE_MODEL_ID")):
            # Soft: workers may supply INFERENCE_MODEL_ID; warn via tag
            pass
    elif job_type == JobType.LABEL.value:
        has_inputs = isinstance(body.get("inputs"), list) and len(body["inputs"]) > 0
        has_local = bool(body.get("dataset_path") or body.get("dataset_alias"))
        if not has_inputs and not has_local:
            raise ValueError(
                "label JobSpec requires payload.inputs or "
                "payload.dataset_alias / payload.dataset_path"
            )
    elif job_type == JobType.COMPUTE.value:
        entrypoint = body.get("entrypoint") or body.get("plugin") or body.get("plugin_module")
        if not entrypoint:
            raise ValueError(
                "compute JobSpec requires payload.entrypoint (module:function)"
            )
        # Sanitize: no path/URL entrypoints (worker also enforces allowlist).
        from .verification import sanitize_entrypoint_string

        body["entrypoint"] = sanitize_entrypoint_string(str(entrypoint))
        if "work_unit" in body and not isinstance(body.get("work_unit"), dict):
            raise ValueError("payload.work_unit must be a JSON object when provided")
        if "work_unit" not in body:
            body["work_unit"] = {}
        verification = body.get("verification")
        if verification is not None and not isinstance(verification, dict):
            raise ValueError("payload.verification must be an object")
        if isinstance(verification, dict):
            mode = str(verification.get("mode") or "").lower()
            if mode and mode not in {"none", "canary", "n_of_m"}:
                raise ValueError("verification.mode must be none|canary|n_of_m")
            if mode == "n_of_m":
                n = int(verification.get("n") or 2)
                m = int(verification.get("m") or 3)
                if n < 1 or m < n:
                    raise ValueError("verification requires 1 <= n <= m")
                verification["n"] = n
                verification["m"] = m
                body["verification"] = verification

    if lease_seconds is not None and float(lease_seconds) <= 0:
        raise ValueError("lease_seconds must be positive")

    return JobSpec(
        job_type=job_type,
        payload=body,
        priority=int(priority),
        tags=list(tags or []),
        max_attempts=max(1, int(max_attempts)),
        lease_seconds=float(lease_seconds) if lease_seconds is not None else None,
        dataset_alias=body.get("dataset_alias"),
    )


@dataclass
class Job:
    job_id: str
    job_type: str
    payload: Dict[str, Any]
    state: str = JobState.QUEUED.value
    created_at: float = field(default_factory=time.time)
    assigned_client: Optional[str] = None
    assigned_at: Optional[float] = None
    lease_expires_at: Optional[float] = None
    lease_seconds: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    completed_at: Optional[float] = None
    priority: int = 0
    tags: List[str] = field(default_factory=list)
    attempts: int = 0
    max_attempts: int = 3
    attempt_history: List[Dict[str, Any]] = field(default_factory=list)
    candidate_results: List[Dict[str, Any]] = field(default_factory=list)
    validation: Optional[Dict[str, Any]] = None

    def to_dict(self, *, include_sensitive: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_sensitive:
            data["payload"] = {
                "redacted": True,
                "keys": sorted(self.payload.keys()) if isinstance(self.payload, dict) else [],
                "dataset_alias": self.payload.get("dataset_alias")
                if isinstance(self.payload, dict)
                else None,
            }
            if self.result is not None:
                data["result"] = {"redacted": True}
        return data

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Job":
        allowed = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in raw.items() if k in allowed}
        cleaned.setdefault("attempt_history", [])
        cleaned.setdefault("candidate_results", [])
        return cls(**cleaned)


@dataclass
class DatasetAlias:
    """Coordinator-visible alias metadata (no absolute worker paths)."""

    alias: str
    description: str = ""
    format_hint: Optional[str] = None
    required_env: Optional[str] = None  # e.g. DATASET_ALIAS_private_reviews
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetAliasRegistry:
    """Durable registry of dataset aliases (paths resolve only on workers)."""

    def __init__(self, state_path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "dataset_aliases.json"
        self.state_path = Path(
            state_path or os.getenv("DATASET_ALIAS_PATH", str(default))
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.aliases: Dict[str, DatasetAlias] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        for item in raw.get("aliases", []):
            alias = DatasetAlias(**{
                k: v for k, v in item.items()
                if k in {f.name for f in fields(DatasetAlias)}
            })
            self.aliases[alias.alias] = alias

    def _persist(self) -> None:
        payload = {
            "version": 1,
            "aliases": [a.to_dict() for a in self.aliases.values()],
        }
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def upsert(
        self,
        alias: str,
        *,
        description: str = "",
        format_hint: Optional[str] = None,
        required_env: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DatasetAlias:
        with self._lock:
            name = alias.strip()
            if not name or "/" in name or "\\" in name or ".." in name:
                raise ValueError("alias must be a simple name without path separators")
            record = DatasetAlias(
                alias=name,
                description=description,
                format_hint=format_hint,
                required_env=required_env or f"DATASET_ALIAS_{name}",
                metadata=dict(metadata or {}),
            )
            self.aliases[name] = record
            self._persist()
            return record

    def get(self, alias: str) -> Optional[DatasetAlias]:
        return self.aliases.get(alias)

    def list_aliases(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in sorted(self.aliases.values(), key=lambda x: x.alias)]

    def require_known(self, alias: str) -> DatasetAlias:
        record = self.get(alias)
        if record is None:
            raise ValueError(f"Unknown dataset_alias: {alias}")
        return record


_alias_registry: Optional[DatasetAliasRegistry] = None


def get_dataset_alias_registry() -> DatasetAliasRegistry:
    global _alias_registry
    if _alias_registry is None:
        _alias_registry = DatasetAliasRegistry()
    return _alias_registry


class JobQueue:
    """Durable local job queue with lease-style assignment."""

    def __init__(
        self,
        lease_seconds: float = 300.0,
        state_path: Optional[str] = None,
        alias_registry: Optional[DatasetAliasRegistry] = None,
    ):
        self.jobs: Dict[str, Job] = {}
        env_lease = os.getenv("JOB_LEASE_SECONDS")
        self.lease_seconds = float(env_lease) if env_lease else float(lease_seconds)
        self._lock = threading.RLock()
        self.alias_registry = alias_registry
        default_path = Path(__file__).resolve().parents[2] / "data" / "jobs.json"
        self.state_path = Path(
            state_path or os.getenv("JOB_QUEUE_STATE_PATH", str(default_path))
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            for value in raw.get("jobs", []):
                job = Job.from_dict(value)
                self.jobs[job.job_id] = job
            self._reclaim_expired()
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load job queue state {self.state_path}: {exc}"
            ) from exc

    def _persist(self) -> None:
        payload = {
            "version": 1,
            "jobs": [job.to_dict() for job in self.jobs.values()],
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.state_path)

    def _job_lease_seconds(self, job: Job) -> float:
        if job.lease_seconds is not None:
            return float(job.lease_seconds)
        return float(self.lease_seconds)

    def create_job(
        self,
        job_type: str,
        payload: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        tags: Optional[List[str]] = None,
        job_id: Optional[str] = None,
        max_attempts: int = 3,
        lease_seconds: Optional[float] = None,
        validate: bool = True,
    ) -> Job:
        with self._lock:
            if validate:
                spec = validate_job_spec(
                    job_type,
                    payload,
                    priority=priority,
                    tags=tags,
                    max_attempts=max_attempts,
                    lease_seconds=lease_seconds,
                )
                if spec.dataset_alias and self.alias_registry is not None:
                    self.alias_registry.require_known(spec.dataset_alias)
                job_type = spec.job_type
                payload = spec.payload
                priority = spec.priority
                tags = spec.tags
                max_attempts = spec.max_attempts
                lease_seconds = spec.lease_seconds
            elif job_type not in {item.value for item in JobType}:
                raise ValueError(f"Unsupported job type: {job_type}")

            jid = job_id or str(uuid.uuid4())
            if jid in self.jobs:
                raise ValueError(f"Job already exists: {jid}")
            job = Job(
                job_id=jid,
                job_type=job_type,
                payload=payload or {},
                priority=priority,
                tags=tags or [],
                max_attempts=max(1, int(max_attempts)),
                lease_seconds=lease_seconds,
            )
            self.jobs[jid] = job
            self._persist()
            return job

    def _close_open_attempt(self, job: Job, outcome: str, now: float) -> None:
        if not job.attempt_history:
            return
        last = job.attempt_history[-1]
        if last.get("ended_at") is None:
            last["ended_at"] = now
            last["outcome"] = outcome

    def _reclaim_expired(self) -> bool:
        now = time.time()
        changed = False
        for job in self.jobs.values():
            if job.state != JobState.ASSIGNED.value:
                continue
            expires = job.lease_expires_at
            if expires is None and job.assigned_at:
                expires = job.assigned_at + self._job_lease_seconds(job)
            if expires is None or now <= expires:
                continue
            self._close_open_attempt(job, "lease_expired", now)
            job.state = (
                JobState.QUEUED.value
                if job.attempts < job.max_attempts
                else JobState.FAILED.value
            )
            job.assigned_client = None
            job.assigned_at = None
            job.lease_expires_at = None
            if job.state == JobState.FAILED.value:
                job.error = "maximum lease attempts exceeded"
                job.completed_at = now
            changed = True
        return changed

    def claim_next(
        self,
        client_id: str,
        job_types: Optional[Set[str]] = None,
    ) -> Optional[Job]:
        try:
            from persistence.shared_state import shared_state_enabled
            from persistence.ha_repos import acquire_named_lock

            if shared_state_enabled():
                with acquire_named_lock(f"job-claim:{client_id}"):
                    return self._claim_next_unlocked(client_id, job_types)
        except Exception:
            pass
        with self._lock:
            return self._claim_next_unlocked(client_id, job_types)

    def _claim_next_unlocked(
        self,
        client_id: str,
        job_types: Optional[Set[str]] = None,
    ) -> Optional[Job]:
        if self._reclaim_expired():
            self._persist()
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
        now = time.time()
        lease = self._job_lease_seconds(job)
        job.state = JobState.ASSIGNED.value
        job.assigned_client = client_id
        job.assigned_at = now
        job.lease_expires_at = now + lease
        job.attempts += 1
        job.attempt_history.append(
            JobAttempt(
                attempt=job.attempts,
                client_id=client_id,
                started_at=now,
                lease_expires_at=job.lease_expires_at,
                outcome="claimed",
            ).to_dict()
        )
        self._persist()
        return job

    def extend_lease(
        self,
        job_id: str,
        client_id: str,
        *,
        extend_seconds: Optional[float] = None,
    ) -> Optional[Job]:
        """Heartbeat: push lease expiry forward for the claiming client."""
        with self._lock:
            self._reclaim_expired()
            job = self.jobs.get(job_id)
            if not job:
                return None
            if (
                job.state != JobState.ASSIGNED.value
                or job.assigned_client != client_id
            ):
                return None
            now = time.time()
            delta = float(extend_seconds) if extend_seconds is not None else self._job_lease_seconds(job)
            if delta <= 0:
                raise ValueError("extend_seconds must be positive")
            base = max(now, job.lease_expires_at or now)
            job.lease_expires_at = base + delta
            job.assigned_at = now  # keep reclaim math consistent for legacy readers
            if job.attempt_history:
                job.attempt_history[-1]["lease_expires_at"] = job.lease_expires_at
            self._persist()
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
            self._reclaim_expired()
            job = self.jobs.get(job_id)
            if not job:
                return None
            if (
                job.state != JobState.ASSIGNED.value
                or job.assigned_client != client_id
            ):
                return None
            now = time.time()
            if success:
                from .verification import apply_verification_on_success

                outcome, validation, candidates, stored = apply_verification_on_success(
                    job.payload if isinstance(job.payload, dict) else {},
                    result,
                    client_id=client_id,
                    candidates=list(job.candidate_results or []),
                )
                job.candidate_results = candidates
                job.validation = validation
                if outcome == "completed":
                    job.state = JobState.COMPLETED.value
                    job.result = stored
                    job.error = None
                    self._close_open_attempt(job, "completed", now)
                    job.completed_at = now
                    job.assigned_client = None
                    job.assigned_at = None
                    job.lease_expires_at = None
                elif outcome == "requeue":
                    job.state = JobState.QUEUED.value
                    job.result = None
                    job.error = None
                    self._close_open_attempt(job, "awaiting_quorum", now)
                    job.assigned_client = None
                    job.assigned_at = None
                    job.lease_expires_at = None
                else:
                    job.state = JobState.FAILED.value
                    job.result = stored
                    job.error = (validation or {}).get("reason") or "verification failed"
                    self._close_open_attempt(job, "verification_failed", now)
                    job.completed_at = now
                    job.assigned_client = None
                    job.assigned_at = None
                    job.lease_expires_at = None
                self._persist()
                return job
            else:
                # Retryable worker failure → requeue if attempts remain
                if job.attempts < job.max_attempts:
                    job.state = JobState.QUEUED.value
                    job.error = error or "failed"
                    job.result = result
                    job.assigned_client = None
                    job.assigned_at = None
                    job.lease_expires_at = None
                    self._close_open_attempt(job, "failed_requeued", now)
                else:
                    job.state = JobState.FAILED.value
                    job.error = error or "failed"
                    job.result = result
                    job.completed_at = now
                    self._close_open_attempt(job, "failed", now)
                    self._persist()
                    return job
            if job.state in {JobState.COMPLETED.value, JobState.FAILED.value}:
                job.completed_at = now
                job.assigned_client = None
                job.assigned_at = None
                job.lease_expires_at = None
            self._persist()
            return job

    def cancel(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job or job.state in {
                JobState.COMPLETED.value,
                JobState.CANCELLED.value,
            }:
                return None
            now = time.time()
            self._close_open_attempt(job, "cancelled", now)
            job.state = JobState.CANCELLED.value
            job.completed_at = now
            job.error = "cancelled by operator"
            job.assigned_client = None
            job.assigned_at = None
            job.lease_expires_at = None
            self._persist()
            return job

    def get(self, job_id: str) -> Optional[Job]:
        return self.jobs.get(job_id)

    def list_jobs(
        self,
        state: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 50,
        include_sensitive: bool = True,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            items = list(self.jobs.values())
            if state:
                items = [j for j in items if j.state == state]
            if job_type:
                items = [j for j in items if j.job_type == job_type]
            items.sort(key=lambda j: j.created_at, reverse=True)
            return [
                j.to_dict(include_sensitive=include_sensitive)
                for j in items[:limit]
            ]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            counts: Dict[str, int] = {}
            for j in self.jobs.values():
                counts[j.state] = counts.get(j.state, 0) + 1
                counts[f"type:{j.job_type}"] = counts.get(f"type:{j.job_type}", 0) + 1
            return {
                "total": len(self.jobs),
                "completed": counts.get(JobState.COMPLETED.value, 0),
                "lease_seconds": self.lease_seconds,
                "counts": counts,
            }


_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    global _queue
    if _queue is None:
        _queue = JobQueue(alias_registry=get_dataset_alias_registry())
    return _queue


def reset_job_queue_for_tests() -> None:
    """Test helper to clear the process singleton."""
    global _queue, _alias_registry
    _queue = None
    _alias_registry = None
