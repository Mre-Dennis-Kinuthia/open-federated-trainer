"""Typed Protocol V2 message schemas (control plane)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ProtocolInfoResponse(BaseModel):
    protocol_version: str
    supported_majors: List[int]
    v2_enabled: bool
    legacy_adapters: Dict[str, str] = Field(
        default_factory=lambda: {
            "api_key_query": "accepted",
            "api_key_body": "accepted",
            "json_weight_delta": "accepted",
        }
    )


class ArtifactManifestV2(BaseModel):
    """Minimum production manifest for content-addressed artifacts."""

    schema_version: str = "2.0"
    artifact_type: str
    artifact_id: str
    content_hash: str
    byte_size: int
    storage_uri: str
    media_type: str = "application/octet-stream"
    serialization_format: str = "raw"
    compression: Optional[str] = None
    created_at: float
    producer_node: Optional[str] = None
    round_id: Optional[int] = None
    job_id: Optional[str] = None
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NodeRegisterV2Request(BaseModel):
    client_name: str
    api_key: Optional[str] = None
    public_key: Optional[str] = None  # raw Ed25519 public key, base64url or hex
    protocol_version: Optional[str] = None
    capabilities: Dict[str, Any] = Field(default_factory=dict)


class NodeRegisterV2Response(BaseModel):
    success: bool
    message: str
    client_id: str
    api_key: str
    protocol_version: str
    public_key_registered: bool = False


class ArtifactUploadMeta(BaseModel):
    """JSON metadata companion for multipart/binary upload headers."""

    artifact_type: str = "weight_delta"
    media_type: str = "application/octet-stream"
    serialization_format: str = "json"
    round_id: Optional[int] = None
    parent_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ArtifactUploadResponse(BaseModel):
    success: bool
    manifest: ArtifactManifestV2
    deduplicated: bool = False


class UpdateByArtifactRequest(BaseModel):
    client_id: str
    round_id: int
    content_hash: str
    idempotency_key: str
    api_key: Optional[str] = None  # legacy body adapter


class UpdateByArtifactResponse(BaseModel):
    success: bool
    message: str
    replayed: bool = False
