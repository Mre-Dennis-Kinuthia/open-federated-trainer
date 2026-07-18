"""SQLAlchemy ORM models for METADATA_BACKEND=postgres (Milestone 1)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, BigInteger, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# Use portable JSON; Postgres can still store JSONB-compatible values.
JsonType = JSON().with_variant(JSONB(), "postgresql")


class NodeRow(Base):
    __tablename__ = "nodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    api_key_hash_prefix: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    registered_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)


class RoundRow(Base):
    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    round_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    state: Mapped[str] = mapped_column(String(64), index=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    assigned_clients: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    updates_received: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    job_type: Mapped[str] = mapped_column(String(64), index=True)
    state: Mapped[str] = mapped_column(String(64), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    result: Mapped[Optional[dict[str, Any]]] = mapped_column(JsonType, nullable=True)
    assigned_client: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ArtifactRow(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_artifacts_content_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    artifact_id: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    artifact_type: Mapped[str] = mapped_column(String(64), index=True)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    byte_size: Mapped[int] = mapped_column(BigInteger, default=0)
    storage_uri: Mapped[str] = mapped_column(Text)
    media_type: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    parent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column("metadata", JsonType, default=dict)


class ClientReputationRow(Base):
    __tablename__ = "client_reputation"

    client_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    rounds_participated: Mapped[int] = mapped_column(Integer, default=0)
    rounds_completed: Mapped[int] = mapped_column(Integer, default=0)
    rounds_dropped: Mapped[int] = mapped_column(Integer, default=0)
    updates_submitted: Mapped[int] = mapped_column(Integer, default=0)
    updates_accepted: Mapped[int] = mapped_column(Integer, default=0)
    updates_rejected: Mapped[int] = mapped_column(Integer, default=0)
    total_latency_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    latency_samples: Mapped[int] = mapped_column(Integer, default=0)
    first_seen: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    last_seen: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    client_rounds: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class ClientIncentiveRow(Base):
    __tablename__ = "client_incentives"

    client_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    total_tokens_earned: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_spent: Mapped[float] = mapped_column(Float, default=0.0)
    speed_bonuses: Mapped[int] = mapped_column(Integer, default=0)
    consistency_bonuses: Mapped[int] = mapped_column(Integer, default=0)
    rewards_received: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    consecutive_completions: Mapped[int] = mapped_column(Integer, default=0)
    last_completion_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class GeoPresenceRow(Base):
    """Coarse presence only — never store raw client IPs."""

    __tablename__ = "geo_presence"

    client_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    lat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lng: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    city: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    last_seen: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class RateLimitBucketRow(Base):
    __tablename__ = "rate_limit_buckets"

    bucket_key: Mapped[str] = mapped_column(String(512), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)
    timestamps: Mapped[list[Any]] = mapped_column(JsonType, default=list)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class HaLockRow(Base):
    """Named locks for multi-replica critical sections (SQLite + Postgres)."""

    __tablename__ = "ha_locks"

    lock_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    holder: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
