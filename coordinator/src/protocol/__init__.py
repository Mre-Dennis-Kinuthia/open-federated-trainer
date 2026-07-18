"""Protocol V2 control-plane helpers (schemas, negotiation, credentials)."""

from .version import (
    PROTOCOL_MAJOR,
    PROTOCOL_MINOR,
    PROTOCOL_VERSION,
    ProtocolIncompatibleError,
    negotiate_protocol_version,
    protocol_v2_enabled,
)
from .credentials import extract_api_key

__all__ = [
    "PROTOCOL_MAJOR",
    "PROTOCOL_MINOR",
    "PROTOCOL_VERSION",
    "ProtocolIncompatibleError",
    "negotiate_protocol_version",
    "protocol_v2_enabled",
    "extract_api_key",
]
