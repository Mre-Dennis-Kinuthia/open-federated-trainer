"""JSON-backed repository adapters wrapping existing coordinator stores."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import (
    ArtifactRecord,
    ArtifactRepository,
    JobRecord,
    JobRepository,
    NodeRecord,
    NodeRepository,
    RoundRecord,
    RoundRepository,
)


class JsonNodeRepository:
    """Reads client ids / keys from StateStore-compatible state.json."""

    def __init__(self, state_path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "state.json"
        self.state_path = Path(state_path or os.getenv("STATE_PATH", str(default)))

    def _load(self) -> Dict[str, Any]:
        if not self.state_path.exists():
            return {"clients": [], "auth": {"client_keys": {}}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def upsert_node(self, node: NodeRecord) -> None:
        # Authoritative writes still go through AuthManager/StateStore.
        # This adapter is read-oriented for Milestone 1 contracts.
        raise NotImplementedError(
            "Use AuthManager/StateStore for node writes in METADATA_BACKEND=json"
        )

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        raw = self._load()
        keys = (raw.get("auth") or {}).get("client_keys") or {}
        clients = set(raw.get("clients") or [])
        if node_id not in clients and node_id not in keys:
            return None
        key = keys.get(node_id)
        prefix = (key[:8] + "…") if key else None
        return NodeRecord(node_id=node_id, api_key_hash_prefix=prefix)

    def list_nodes(self) -> List[NodeRecord]:
        raw = self._load()
        keys = (raw.get("auth") or {}).get("client_keys") or {}
        clients = set(raw.get("clients") or []) | set(keys.keys())
        return [self.get_node(cid) for cid in sorted(clients) if self.get_node(cid)]


class JsonRoundRepository:
    """Persists classic FedAvg rounds to classic_rounds.json for restart recovery."""

    def __init__(self, rounds_path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "classic_rounds.json"
        self.rounds_path = Path(rounds_path or os.getenv("CLASSIC_ROUNDS_PATH", str(default)))

    def save_round(self, round_rec: RoundRecord) -> None:
        data: Dict[str, Any] = {"rounds": {}}
        if self.rounds_path.exists():
            data = json.loads(self.rounds_path.read_text(encoding="utf-8"))
        data.setdefault("rounds", {})[str(round_rec.round_id)] = {
            "round_id": round_rec.round_id,
            "state": round_rec.state,
            "model_version": round_rec.model_version,
            "assigned_clients": round_rec.assigned_clients,
            "updates_received": round_rec.updates_received,
            "metadata": round_rec.metadata,
        }
        self.rounds_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.rounds_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.rounds_path)

    def get_round(self, round_id: int) -> Optional[RoundRecord]:
        if not self.rounds_path.exists():
            return None
        data = json.loads(self.rounds_path.read_text(encoding="utf-8"))
        raw = (data.get("rounds") or {}).get(str(round_id))
        if not raw:
            return None
        return RoundRecord(
            round_id=int(raw["round_id"]),
            state=raw["state"],
            model_version=raw.get("model_version"),
            assigned_clients=list(raw.get("assigned_clients") or []),
            updates_received=list(raw.get("updates_received") or []),
            metadata=dict(raw.get("metadata") or {}),
        )

    def list_rounds(self, limit: int = 50) -> List[RoundRecord]:
        if not self.rounds_path.exists():
            return []
        data = json.loads(self.rounds_path.read_text(encoding="utf-8"))
        rows = []
        for raw in (data.get("rounds") or {}).values():
            rows.append(
                RoundRecord(
                    round_id=int(raw["round_id"]),
                    state=raw["state"],
                    model_version=raw.get("model_version"),
                    assigned_clients=list(raw.get("assigned_clients") or []),
                    updates_received=list(raw.get("updates_received") or []),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
        rows.sort(key=lambda r: r.round_id, reverse=True)
        return rows[:limit]


class JsonJobRepository:
    def __init__(self, jobs_path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "jobs.json"
        self.jobs_path = Path(
            jobs_path or os.getenv("JOB_QUEUE_STATE_PATH", str(default))
        )

    def _load_jobs(self) -> List[Dict[str, Any]]:
        if not self.jobs_path.exists():
            return []
        raw = json.loads(self.jobs_path.read_text(encoding="utf-8"))
        return list(raw.get("jobs") or [])

    def save_job(self, job: JobRecord) -> None:
        raise NotImplementedError("Use JobQueue for job writes in METADATA_BACKEND=json")

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        for raw in self._load_jobs():
            if raw.get("job_id") == job_id:
                return JobRecord(
                    job_id=raw["job_id"],
                    job_type=raw["job_type"],
                    state=raw["state"],
                    payload=dict(raw.get("payload") or {}),
                    result=raw.get("result"),
                    assigned_client=raw.get("assigned_client"),
                )
        return None

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        rows = []
        for raw in self._load_jobs():
            rows.append(
                JobRecord(
                    job_id=raw["job_id"],
                    job_type=raw["job_type"],
                    state=raw["state"],
                    payload=dict(raw.get("payload") or {}),
                    result=raw.get("result"),
                    assigned_client=raw.get("assigned_client"),
                )
            )
        rows.sort(key=lambda j: j.job_id)
        return rows[:limit]


class JsonArtifactRepository:
    """Manifest index stored as coordinator/data/artifacts_index.json."""

    def __init__(self, index_path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "artifacts_index.json"
        self.index_path = Path(
            index_path or os.getenv("ARTIFACT_INDEX_PATH", str(default))
        )

    def _load(self) -> Dict[str, Any]:
        if not self.index_path.exists():
            return {"artifacts": {}}
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save(self, data: Dict[str, Any]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.index_path)

    def put_manifest(self, record: ArtifactRecord) -> None:
        data = self._load()
        data.setdefault("artifacts", {})[record.artifact_id] = {
            "artifact_id": record.artifact_id,
            "artifact_type": record.artifact_type,
            "content_hash": record.content_hash,
            "byte_size": record.byte_size,
            "storage_uri": record.storage_uri,
            "media_type": record.media_type,
            "created_at": record.created_at or time.time(),
            "parent_id": record.parent_id,
            "metadata": record.metadata,
        }
        self._save(data)

    def get_manifest(self, artifact_id: str) -> Optional[ArtifactRecord]:
        raw = (self._load().get("artifacts") or {}).get(artifact_id)
        if not raw:
            return None
        return ArtifactRecord(**{k: raw[k] for k in ArtifactRecord.__dataclass_fields__})

    def get_by_hash(self, content_hash: str) -> Optional[ArtifactRecord]:
        for raw in (self._load().get("artifacts") or {}).values():
            if raw.get("content_hash") == content_hash:
                return ArtifactRecord(
                    **{k: raw[k] for k in ArtifactRecord.__dataclass_fields__}
                )
        return None

    def list_manifests(self, limit: int = 100) -> List[ArtifactRecord]:
        rows = [
            ArtifactRecord(**{k: raw[k] for k in ArtifactRecord.__dataclass_fields__})
            for raw in (self._load().get("artifacts") or {}).values()
        ]
        rows.sort(key=lambda a: a.created_at, reverse=True)
        return rows[:limit]


def metadata_backend() -> str:
    return os.getenv("METADATA_BACKEND", "json").strip().lower() or "json"


def _use_sql() -> bool:
    return metadata_backend() in {"postgres", "sql", "sqlite"}


def get_artifact_repository() -> ArtifactRepository:
    if _use_sql():
        from .db import create_all_tables
        from .sql_repos import SqlArtifactRepository

        create_all_tables()
        return SqlArtifactRepository()
    return JsonArtifactRepository()


def get_node_repository() -> NodeRepository:
    if _use_sql():
        from .db import create_all_tables
        from .sql_repos import SqlNodeRepository

        create_all_tables()
        return SqlNodeRepository()
    return JsonNodeRepository()


def get_job_repository() -> JobRepository:
    if _use_sql():
        from .db import create_all_tables
        from .sql_repos import SqlJobRepository

        create_all_tables()
        return SqlJobRepository()
    return JsonJobRepository()


def get_round_repository() -> RoundRepository:
    if _use_sql():
        from .db import create_all_tables
        from .sql_repos import SqlRoundRepository

        create_all_tables()
        return SqlRoundRepository()
    return JsonRoundRepository()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
