"""SQL repositories for HA shared state (reputation, incentives, geo, locks)."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import select, text

from .db import get_engine, get_session
from .models import (
    ClientIncentiveRow,
    ClientReputationRow,
    GeoPresenceRow,
    HaLockRow,
    RateLimitBucketRow,
    RoundRow,
)


class SqlReputationRepository:
    def save(self, data: Dict[str, Any], client_rounds: Optional[List[int]] = None) -> None:
        client_id = data["client_id"]
        with get_session() as session:
            row = session.get(ClientReputationRow, client_id)
            if row is None:
                row = ClientReputationRow(client_id=client_id)
                session.add(row)
            row.rounds_participated = int(data.get("rounds_participated", 0))
            row.rounds_completed = int(data.get("rounds_completed", 0))
            row.rounds_dropped = int(data.get("rounds_dropped", 0))
            row.updates_submitted = int(data.get("updates_submitted", 0))
            row.updates_accepted = int(data.get("updates_accepted", 0))
            row.updates_rejected = int(data.get("updates_rejected", 0))
            row.total_latency_seconds = float(data.get("total_latency_seconds", 0.0))
            row.latency_samples = int(data.get("latency_samples", 0))
            row.first_seen = data.get("first_seen")
            row.last_seen = data.get("last_seen")
            if client_rounds is not None:
                row.client_rounds = list(client_rounds)
            session.commit()

    def get(self, client_id: str) -> Optional[Dict[str, Any]]:
        with get_session() as session:
            row = session.get(ClientReputationRow, client_id)
            if row is None:
                return None
            return self._to_dict(row)

    def list_all(self) -> List[Dict[str, Any]]:
        with get_session() as session:
            rows = session.scalars(select(ClientReputationRow)).all()
            return [self._to_dict(r) for r in rows]

    @staticmethod
    def _to_dict(row: ClientReputationRow) -> Dict[str, Any]:
        return {
            "client_id": row.client_id,
            "rounds_participated": row.rounds_participated,
            "rounds_completed": row.rounds_completed,
            "rounds_dropped": row.rounds_dropped,
            "updates_submitted": row.updates_submitted,
            "updates_accepted": row.updates_accepted,
            "updates_rejected": row.updates_rejected,
            "total_latency_seconds": row.total_latency_seconds,
            "latency_samples": row.latency_samples,
            "first_seen": row.first_seen,
            "last_seen": row.last_seen,
            "client_rounds": list(row.client_rounds or []),
        }


class SqlIncentiveRepository:
    def save(self, data: Dict[str, Any]) -> None:
        client_id = data["client_id"]
        with get_session() as session:
            row = session.get(ClientIncentiveRow, client_id)
            if row is None:
                row = ClientIncentiveRow(client_id=client_id)
                session.add(row)
            row.total_tokens_earned = float(data.get("total_tokens_earned", 0.0))
            row.tokens_spent = float(data.get("tokens_spent", 0.0))
            row.speed_bonuses = int(data.get("speed_bonuses", 0))
            row.consistency_bonuses = int(data.get("consistency_bonuses", 0))
            row.rewards_received = list(data.get("rewards_received") or [])
            row.consecutive_completions = int(data.get("consecutive_completions", 0))
            row.last_completion_time = data.get("last_completion_time")
            session.commit()

    def get(self, client_id: str) -> Optional[Dict[str, Any]]:
        with get_session() as session:
            row = session.get(ClientIncentiveRow, client_id)
            if row is None:
                return None
            return self._to_dict(row)

    def list_all(self) -> List[Dict[str, Any]]:
        with get_session() as session:
            rows = session.scalars(select(ClientIncentiveRow)).all()
            return [self._to_dict(r) for r in rows]

    @staticmethod
    def _to_dict(row: ClientIncentiveRow) -> Dict[str, Any]:
        return {
            "client_id": row.client_id,
            "total_tokens_earned": row.total_tokens_earned,
            "tokens_spent": row.tokens_spent,
            "speed_bonuses": row.speed_bonuses,
            "consistency_bonuses": row.consistency_bonuses,
            "rewards_received": list(row.rewards_received or []),
            "consecutive_completions": row.consecutive_completions,
            "last_completion_time": row.last_completion_time,
        }


class SqlGeoPresenceRepository:
    def upsert(self, client_id: str, entry: Dict[str, Any]) -> None:
        # Never persist ip_key / raw IPs
        with get_session() as session:
            row = session.get(GeoPresenceRow, client_id)
            if row is None:
                row = GeoPresenceRow(client_id=client_id)
                session.add(row)
            row.lat = entry.get("lat")
            row.lng = entry.get("lng")
            row.city = entry.get("city")
            row.country = entry.get("country")
            row.last_seen = float(entry.get("last_seen") or time.time())
            session.commit()

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        with get_session() as session:
            rows = session.scalars(select(GeoPresenceRow)).all()
            out: Dict[str, Dict[str, Any]] = {}
            for row in rows:
                entry: Dict[str, Any] = {"last_seen": row.last_seen}
                if row.lat is not None:
                    entry["lat"] = row.lat
                if row.lng is not None:
                    entry["lng"] = row.lng
                if row.city is not None:
                    entry["city"] = row.city
                if row.country is not None:
                    entry["country"] = row.country
                out[row.client_id] = entry
            return out


class SqlRateLimitRepository:
    def get_timestamps(self, bucket_key: str) -> List[float]:
        with get_session() as session:
            row = session.get(RateLimitBucketRow, bucket_key)
            if row is None:
                return []
            return [float(t) for t in (row.timestamps or [])]

    def set_timestamps(self, bucket_key: str, timestamps: List[float]) -> None:
        with get_session() as session:
            row = session.get(RateLimitBucketRow, bucket_key)
            if row is None:
                row = RateLimitBucketRow(bucket_key=bucket_key)
                session.add(row)
            row.timestamps = list(timestamps)
            row.count = len(timestamps)
            session.commit()

    def get_update_count(self, client_id: str, round_id: int) -> int:
        key = f"upd:{client_id}:{round_id}"
        with get_session() as session:
            row = session.get(RateLimitBucketRow, key)
            return int(row.count) if row else 0

    def incr_update_count(self, client_id: str, round_id: int) -> int:
        key = f"upd:{client_id}:{round_id}"
        with get_session() as session:
            row = session.get(RateLimitBucketRow, key)
            if row is None:
                row = RateLimitBucketRow(bucket_key=key, count=0, timestamps=[])
                session.add(row)
            row.count = int(row.count or 0) + 1
            session.commit()
            return int(row.count)


@contextmanager
def acquire_named_lock(lock_name: str, holder: str = "coordinator") -> Iterator[None]:
    """
    Cross-replica critical section.

    Postgres: pg_advisory_xact_lock on hash of name.
    SQLite / others: row lock on ha_locks via SELECT FOR UPDATE.
    """
    engine = get_engine()
    dialect = engine.dialect.name
    with get_session() as session:
        if dialect == "postgresql":
            session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext(:name))"),
                {"name": lock_name},
            )
        else:
            row = session.get(HaLockRow, lock_name)
            if row is None:
                session.add(HaLockRow(lock_name=lock_name, holder=holder))
                session.flush()
            # Re-select with FOR UPDATE
            session.execute(
                select(HaLockRow)
                .where(HaLockRow.lock_name == lock_name)
                .with_for_update()
            )
            row = session.get(HaLockRow, lock_name)
            if row is not None:
                row.holder = holder
        try:
            yield
            session.commit()
        except Exception:
            session.rollback()
            raise


def try_transition_round_aggregating(round_id: int) -> tuple[bool, Optional[str]]:
    """
    Atomically move COLLECTING/OPEN → AGGREGATING under a row lock.

    Returns (acquired, current_state). If already AGGREGATING/CLOSED, acquired=False.
    """
    with get_session() as session:
        row = session.execute(
            select(RoundRow).where(RoundRow.round_id == round_id).with_for_update()
        ).scalar_one_or_none()
        if row is None:
            session.rollback()
            return False, None
        state = row.state
        if state in ("AGGREGATING", "CLOSED"):
            session.commit()
            return False, state
        if state not in ("OPEN", "COLLECTING"):
            session.commit()
            return False, state
        row.state = "AGGREGATING"
        meta = dict(row.metadata_json or {})
        meta["aggregate_lock_at"] = time.time()
        row.metadata_json = meta
        session.commit()
        return True, "AGGREGATING"
