"""Shared-state helpers for multi-replica HA (Postgres/SQLite)."""

from __future__ import annotations

import os


def metadata_backend() -> str:
    return os.getenv("METADATA_BACKEND", "json").strip().lower() or "json"


def shared_state_enabled() -> bool:
    """
    When true, reputation / incentives / geo / rate limits / round SoT use SQL.

    SHARED_STATE=auto (default) follows METADATA_BACKEND in {postgres,sql,sqlite}.
    SHARED_STATE=true|false overrides.
    """
    flag = os.getenv("SHARED_STATE", "auto").strip().lower() or "auto"
    if flag in ("1", "true", "yes", "on"):
        return True
    if flag in ("0", "false", "no", "off"):
        return False
    return metadata_backend() in {"postgres", "sql", "sqlite"}
