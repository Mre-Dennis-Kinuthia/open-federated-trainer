"""Resolve client credentials from headers (preferred) or legacy query/body."""

from __future__ import annotations

from typing import Optional


def extract_api_key(
    *,
    x_api_key: Optional[str] = None,
    authorization: Optional[str] = None,
    query_api_key: Optional[str] = None,
    body_api_key: Optional[str] = None,
) -> Optional[str]:
    """
    Prefer ``X-Api-Key`` / ``Authorization: Bearer``, then body, then query.

    Query-param keys remain supported as a compatibility adapter (soft deprecation).
    """
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    if authorization:
        auth = authorization.strip()
        lower = auth.lower()
        if lower.startswith("bearer "):
            token = auth[7:].strip()
            if token:
                return token
    if body_api_key and str(body_api_key).strip():
        return str(body_api_key).strip()
    if query_api_key and str(query_api_key).strip():
        return str(query_api_key).strip()
    return None
