"""Database engine/session helpers for METADATA_BACKEND=postgres."""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def database_url() -> str:
    """
    Connection URL for SQL metadata.

    Defaults to a local SQLite file so unit tests and offline smoke work
    without Postgres. Production should set DATABASE_URL to PostgreSQL.
    """
    return os.getenv(
        "DATABASE_URL",
        f"sqlite:///{os.getenv('SQLITE_METADATA_PATH', '/tmp/fedcompute-metadata.sqlite')}",
    )


def get_engine(*, echo: bool = False) -> Engine:
    global _engine, _SessionLocal
    if _engine is None:
        url = database_url()
        connect_args = {}
        kwargs: dict = {"echo": echo, "future": True, "connect_args": connect_args}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
            # Avoid cross-connection lock storms under multi-replica SQLite tests
            from sqlalchemy.pool import StaticPool

            if ":memory:" in url or "mode=memory" in url:
                kwargs["poolclass"] = StaticPool
            else:
                kwargs["pool_pre_ping"] = True
                connect_args["timeout"] = 30
        _engine = create_engine(url, **kwargs)

        if url.startswith("sqlite"):

            @event.listens_for(_engine, "connect")
            def _sqlite_fk(dbapi_conn, _):  # type: ignore[no-untyped-def]
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.close()

        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)
    return _engine


def get_session() -> Session:
    get_engine()
    assert _SessionLocal is not None
    return _SessionLocal()


def reset_engine() -> None:
    """Test helper to clear the process-global engine."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None


def create_all_tables() -> None:
    """Create ORM tables if missing.

    Under multi-replica boot, concurrent ``CREATE TABLE`` / sequence DDL can
    race (Postgres UniqueViolation on ``pg_class``). Serialize with an
    advisory lock on PostgreSQL; ignore duplicate-object errors as a fallback.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

    from .models import Base

    engine = get_engine()
    url = database_url()
    if url.startswith("postgresql"):
        # Fixed key shared by all coordinator replicas for schema bootstrap.
        lock_key = 87231401
        with engine.connect() as conn:
            conn.execute(text("SELECT pg_advisory_lock(:k)"), {"k": lock_key})
            try:
                Base.metadata.create_all(bind=conn)
                conn.commit()
            except (IntegrityError, OperationalError, ProgrammingError) as exc:
                conn.rollback()
                msg = str(getattr(exc, "orig", exc)).lower()
                if "already exists" not in msg and "duplicate" not in msg:
                    raise
            finally:
                conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": lock_key})
                conn.commit()
        return

    try:
        Base.metadata.create_all(bind=engine)
    except (IntegrityError, OperationalError, ProgrammingError) as exc:
        msg = str(getattr(exc, "orig", exc)).lower()
        if "already exists" not in msg and "duplicate" not in msg:
            raise
