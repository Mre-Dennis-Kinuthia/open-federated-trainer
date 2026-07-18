"""Protocol version negotiation."""

from __future__ import annotations

import os
from typing import Optional, Tuple


PROTOCOL_MAJOR = 2
PROTOCOL_MINOR = 0
PROTOCOL_VERSION = f"{PROTOCOL_MAJOR}.{PROTOCOL_MINOR}"

# Legacy clients omit the header; major 1 remains accepted via adapters.
SUPPORTED_MAJORS = {1, 2}


class ProtocolIncompatibleError(Exception):
    def __init__(self, requested: str, message: str):
        self.requested = requested
        super().__init__(message)


def protocol_v2_enabled() -> bool:
    return os.getenv("PROTOCOL_V2", "").strip().lower() in {"1", "true", "yes", "on"}


def parse_protocol_version(value: str) -> Tuple[int, int]:
    text = (value or "").strip()
    if not text:
        raise ProtocolIncompatibleError(value, "empty protocol_version")
    parts = text.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError as exc:
        raise ProtocolIncompatibleError(value, f"invalid protocol_version: {value}") from exc
    return major, minor


def negotiate_protocol_version(requested: Optional[str]) -> str:
    """
    Accept additive minors within a supported major; reject unknown majors.

    Omitting the version is allowed (legacy v1 clients).
    """
    if requested is None or str(requested).strip() == "":
        return "1.0"
    major, minor = parse_protocol_version(str(requested))
    if major not in SUPPORTED_MAJORS:
        raise ProtocolIncompatibleError(
            requested,
            f"unsupported protocol major {major}; supported={sorted(SUPPORTED_MAJORS)}",
        )
    if major == PROTOCOL_MAJOR and minor > PROTOCOL_MINOR:
        # Forward-compatible: unknown minor features may be ignored, but we still accept.
        return f"{major}.{minor}"
    return f"{major}.{minor}"
