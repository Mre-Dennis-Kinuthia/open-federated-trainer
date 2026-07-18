"""Persistence repository interfaces (Milestone 1 foundation).

Default backend remains JSON via thin adapters around existing stores.
PostgreSQL adapters are opt-in behind METADATA_BACKEND=postgres (stub until wired).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    content_hash: str
    byte_size: int
    storage_uri: str
    media_type: str = "application/octet-stream"
    created_at: float = 0.0
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeRecord:
    node_id: str
    api_key_hash_prefix: Optional[str] = None
    registered_at: Optional[float] = None
    last_seen: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundRecord:
    round_id: int
    state: str
    model_version: Optional[str] = None
    assigned_clients: List[str] = field(default_factory=list)
    updates_received: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobRecord:
    job_id: str
    job_type: str
    state: str
    payload: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    assigned_client: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class NodeRepository(Protocol):
    def upsert_node(self, node: NodeRecord) -> None: ...
    def get_node(self, node_id: str) -> Optional[NodeRecord]: ...
    def list_nodes(self) -> List[NodeRecord]: ...


@runtime_checkable
class RoundRepository(Protocol):
    def save_round(self, round_rec: RoundRecord) -> None: ...
    def get_round(self, round_id: int) -> Optional[RoundRecord]: ...
    def list_rounds(self, limit: int = 50) -> List[RoundRecord]: ...


@runtime_checkable
class JobRepository(Protocol):
    def save_job(self, job: JobRecord) -> None: ...
    def get_job(self, job_id: str) -> Optional[JobRecord]: ...
    def list_jobs(self, limit: int = 50) -> List[JobRecord]: ...


@runtime_checkable
class ArtifactRepository(Protocol):
    def put_manifest(self, record: ArtifactRecord) -> None: ...
    def get_manifest(self, artifact_id: str) -> Optional[ArtifactRecord]: ...
    def get_by_hash(self, content_hash: str) -> Optional[ArtifactRecord]: ...
    def list_manifests(self, limit: int = 100) -> List[ArtifactRecord]: ...
