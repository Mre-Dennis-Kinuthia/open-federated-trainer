"""SQL repository adapters for METADATA_BACKEND=postgres (or SQLite via DATABASE_URL)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select

from . import (
    ArtifactRecord,
    JobRecord,
    NodeRecord,
    RoundRecord,
)
from .db import get_session
from .models import ArtifactRow, JobRow, NodeRow, RoundRow


def _dt(ts: Optional[float]) -> Optional[datetime]:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _ts(dt: Optional[datetime]) -> Optional[float]:
    if dt is None:
        return None
    return dt.timestamp()


class SqlNodeRepository:
    def upsert_node(self, node: NodeRecord) -> None:
        with get_session() as session:
            row = session.scalar(select(NodeRow).where(NodeRow.node_id == node.node_id))
            if row is None:
                row = NodeRow(node_id=node.node_id)
                session.add(row)
            row.api_key_hash_prefix = node.api_key_hash_prefix
            row.registered_at = _dt(node.registered_at) or row.registered_at
            row.last_seen = _dt(node.last_seen)
            row.metadata_json = dict(node.metadata or {})
            session.commit()

    def get_node(self, node_id: str) -> Optional[NodeRecord]:
        with get_session() as session:
            row = session.scalar(select(NodeRow).where(NodeRow.node_id == node_id))
            if row is None:
                return None
            return NodeRecord(
                node_id=row.node_id,
                api_key_hash_prefix=row.api_key_hash_prefix,
                registered_at=_ts(row.registered_at),
                last_seen=_ts(row.last_seen),
                metadata=dict(row.metadata_json or {}),
            )

    def list_nodes(self) -> List[NodeRecord]:
        with get_session() as session:
            rows = session.scalars(select(NodeRow).order_by(NodeRow.node_id)).all()
            return [
                NodeRecord(
                    node_id=r.node_id,
                    api_key_hash_prefix=r.api_key_hash_prefix,
                    registered_at=_ts(r.registered_at),
                    last_seen=_ts(r.last_seen),
                    metadata=dict(r.metadata_json or {}),
                )
                for r in rows
            ]


class SqlRoundRepository:
    def save_round(self, round_rec: RoundRecord) -> None:
        with get_session() as session:
            row = session.scalar(
                select(RoundRow).where(RoundRow.round_id == round_rec.round_id)
            )
            if row is None:
                row = RoundRow(round_id=round_rec.round_id)
                session.add(row)
            row.state = round_rec.state
            row.model_version = round_rec.model_version
            row.assigned_clients = list(round_rec.assigned_clients)
            row.updates_received = list(round_rec.updates_received)
            row.metadata_json = dict(round_rec.metadata or {})
            session.commit()

    def get_round(self, round_id: int) -> Optional[RoundRecord]:
        with get_session() as session:
            row = session.scalar(select(RoundRow).where(RoundRow.round_id == round_id))
            if row is None:
                return None
            return RoundRecord(
                round_id=row.round_id,
                state=row.state,
                model_version=row.model_version,
                assigned_clients=list(row.assigned_clients or []),
                updates_received=list(row.updates_received or []),
                metadata=dict(row.metadata_json or {}),
            )

    def list_rounds(self, limit: int = 50) -> List[RoundRecord]:
        with get_session() as session:
            rows = session.scalars(
                select(RoundRow).order_by(RoundRow.round_id.desc()).limit(limit)
            ).all()
            return [
                RoundRecord(
                    round_id=r.round_id,
                    state=r.state,
                    model_version=r.model_version,
                    assigned_clients=list(r.assigned_clients or []),
                    updates_received=list(r.updates_received or []),
                    metadata=dict(r.metadata_json or {}),
                )
                for r in rows
            ]


class SqlJobRepository:
    def save_job(self, job: JobRecord) -> None:
        with get_session() as session:
            row = session.scalar(select(JobRow).where(JobRow.job_id == job.job_id))
            if row is None:
                row = JobRow(job_id=job.job_id)
                session.add(row)
            row.job_type = job.job_type
            row.state = job.state
            row.payload = dict(job.payload or {})
            row.result = job.result
            row.assigned_client = job.assigned_client
            row.metadata_json = dict(job.metadata or {})
            session.commit()

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with get_session() as session:
            row = session.scalar(select(JobRow).where(JobRow.job_id == job_id))
            if row is None:
                return None
            return JobRecord(
                job_id=row.job_id,
                job_type=row.job_type,
                state=row.state,
                payload=dict(row.payload or {}),
                result=row.result,
                assigned_client=row.assigned_client,
                metadata=dict(row.metadata_json or {}),
            )

    def list_jobs(self, limit: int = 50) -> List[JobRecord]:
        with get_session() as session:
            rows = session.scalars(
                select(JobRow).order_by(JobRow.job_id).limit(limit)
            ).all()
            return [
                JobRecord(
                    job_id=r.job_id,
                    job_type=r.job_type,
                    state=r.state,
                    payload=dict(r.payload or {}),
                    result=r.result,
                    assigned_client=r.assigned_client,
                    metadata=dict(r.metadata_json or {}),
                )
                for r in rows
            ]


class SqlArtifactRepository:
    def put_manifest(self, record: ArtifactRecord) -> None:
        with get_session() as session:
            row = session.scalar(
                select(ArtifactRow).where(ArtifactRow.artifact_id == record.artifact_id)
            )
            if row is None:
                row = ArtifactRow(artifact_id=record.artifact_id)
                session.add(row)
            row.artifact_type = record.artifact_type
            row.content_hash = record.content_hash
            row.byte_size = record.byte_size
            row.storage_uri = record.storage_uri
            row.media_type = record.media_type
            row.parent_id = record.parent_id
            row.metadata_json = dict(record.metadata or {})
            if record.created_at:
                row.created_at = datetime.fromtimestamp(
                    record.created_at, tz=timezone.utc
                )
            session.commit()

    def get_manifest(self, artifact_id: str) -> Optional[ArtifactRecord]:
        with get_session() as session:
            row = session.scalar(
                select(ArtifactRow).where(ArtifactRow.artifact_id == artifact_id)
            )
            if row is None:
                return None
            return ArtifactRecord(
                artifact_id=row.artifact_id,
                artifact_type=row.artifact_type,
                content_hash=row.content_hash,
                byte_size=int(row.byte_size or 0),
                storage_uri=row.storage_uri,
                media_type=row.media_type,
                created_at=_ts(row.created_at) or 0.0,
                parent_id=row.parent_id,
                metadata=dict(row.metadata_json or {}),
            )

    def get_by_hash(self, content_hash: str) -> Optional[ArtifactRecord]:
        with get_session() as session:
            row = session.scalar(
                select(ArtifactRow).where(ArtifactRow.content_hash == content_hash)
            )
            if row is None:
                return None
            return ArtifactRecord(
                artifact_id=row.artifact_id,
                artifact_type=row.artifact_type,
                content_hash=row.content_hash,
                byte_size=int(row.byte_size or 0),
                storage_uri=row.storage_uri,
                media_type=row.media_type,
                created_at=_ts(row.created_at) or 0.0,
                parent_id=row.parent_id,
                metadata=dict(row.metadata_json or {}),
            )

    def list_manifests(self, limit: int = 100) -> List[ArtifactRecord]:
        with get_session() as session:
            rows = session.scalars(
                select(ArtifactRow)
                .order_by(ArtifactRow.created_at.desc())
                .limit(limit)
            ).all()
            return [
                ArtifactRecord(
                    artifact_id=r.artifact_id,
                    artifact_type=r.artifact_type,
                    content_hash=r.content_hash,
                    byte_size=int(r.byte_size or 0),
                    storage_uri=r.storage_uri,
                    media_type=r.media_type,
                    created_at=_ts(r.created_at) or 0.0,
                    parent_id=r.parent_id,
                    metadata=dict(r.metadata_json or {}),
                )
                for r in rows
            ]
