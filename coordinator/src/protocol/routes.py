"""Protocol V2 HTTP routes (mounted from main)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import Response

from artifacts import get_artifact_store
from persistence import ArtifactRecord
from persistence.json_repos import get_artifact_repository
from protocol.credentials import extract_api_key
from protocol.idempotency import IdempotencyStore
from protocol.schemas import (
    ArtifactManifestV2,
    ArtifactUploadResponse,
    NodeRegisterV2Request,
    NodeRegisterV2Response,
    ProtocolInfoResponse,
    UpdateByArtifactRequest,
    UpdateByArtifactResponse,
)
from protocol.version import (
    PROTOCOL_VERSION,
    SUPPORTED_MAJORS,
    ProtocolIncompatibleError,
    negotiate_protocol_version,
    protocol_v2_enabled,
)

router = APIRouter(tags=["protocol-v2"])

_idempotency = IdempotencyStore()


def bind_dependencies(
    *,
    auth_manager,
    round_manager,
    update_validator,
    metrics_collector,
    geo_presence,
    client_ip_fn,
    register_legacy_fn,
) -> None:
    """Attach coordinator singletons (called once from main)."""
    router.auth_manager = auth_manager  # type: ignore[attr-defined]
    router.round_manager = round_manager  # type: ignore[attr-defined]
    router.update_validator = update_validator  # type: ignore[attr-defined]
    router.metrics_collector = metrics_collector  # type: ignore[attr-defined]
    router.geo_presence = geo_presence  # type: ignore[attr-defined]
    router.client_ip_fn = client_ip_fn  # type: ignore[attr-defined]
    router.register_legacy_fn = register_legacy_fn  # type: ignore[attr-defined]


def _optional_header(value: Optional[str]) -> Optional[str]:
    """FastAPI Header defaults are objects when routes are invoked directly in tests."""
    return value if isinstance(value, str) else None


def _require_protocol(x_protocol_version: Optional[str]) -> str:
    try:
        return negotiate_protocol_version(_optional_header(x_protocol_version))
    except ProtocolIncompatibleError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _auth_or_401(client_id: str, api_key: Optional[str]) -> None:
    auth = router.auth_manager  # type: ignore[attr-defined]
    if auth and not auth.validate_api_key(api_key, client_id):
        raise HTTPException(status_code=401, detail="Authentication failed. Valid API key required.")


@router.get("/protocol", response_model=ProtocolInfoResponse)
async def protocol_info() -> ProtocolInfoResponse:
    return ProtocolInfoResponse(
        protocol_version=PROTOCOL_VERSION,
        supported_majors=sorted(SUPPORTED_MAJORS),
        v2_enabled=protocol_v2_enabled(),
    )


@router.post("/v2/node/register", response_model=NodeRegisterV2Response)
async def register_node_v2(
    request: NodeRegisterV2Request,
    http_request: Request,
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> NodeRegisterV2Response:
    negotiated = _require_protocol(request.protocol_version or x_protocol_version or PROTOCOL_VERSION)
    # Reuse legacy registration (PoP + key issue) then attach public key.
    from core.auth import ClientAlreadyRegisteredError

    auth = router.auth_manager  # type: ignore[attr-defined]
    geo = router.geo_presence  # type: ignore[attr-defined]
    geo.record(request.client_name, router.client_ip_fn(http_request))  # type: ignore[attr-defined]

    already = auth.is_registered(request.client_name)
    try:
        api_key = auth.register_client(
            request.client_name,
            presented_key=request.api_key,
        )
    except ClientAlreadyRegisteredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    rm = router.round_manager  # type: ignore[attr-defined]
    if request.client_name not in rm.clients:
        rm.register_client(request.client_name)

    public_registered = False
    if request.public_key:
        try:
            auth.set_public_key(request.client_name, request.public_key)
            public_registered = True
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    message = (
        f"Client {request.client_name} resumed."
        if already
        else f"Client {request.client_name} registered (protocol {negotiated})."
    )
    return NodeRegisterV2Response(
        success=True,
        message=message,
        client_id=request.client_name,
        api_key=api_key,
        protocol_version=negotiated,
        public_key_registered=public_registered,
    )


@router.post("/v2/artifacts", response_model=ArtifactUploadResponse)
async def upload_artifact(
    http_request: Request,
    client_id: str = Query(...),
    artifact_type: str = Query("weight_delta"),
    media_type: str = Query("application/octet-stream"),
    serialization_format: str = Query("raw"),
    round_id: Optional[int] = Query(None),
    parent_id: Optional[str] = Query(None),
    idempotency_key: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
    x_content_sha256: Optional[str] = Header(None, alias="X-Content-SHA256"),
) -> ArtifactUploadResponse:
    _require_protocol(x_protocol_version or PROTOCOL_VERSION)
    resolved = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        query_api_key=api_key,
    )
    _auth_or_401(client_id, resolved)

    max_bytes = int(os.getenv("MAX_ARTIFACT_UPLOAD_BYTES", str(50_000_000)))
    body = await http_request.body()
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Artifact is {len(body)} bytes; maximum is {max_bytes}",
        )
    if not body:
        raise HTTPException(status_code=400, detail="Empty artifact body")

    digest = hashlib.sha256(body).hexdigest()
    if x_content_sha256 and x_content_sha256.lower() != digest:
        raise HTTPException(status_code=400, detail="X-Content-SHA256 does not match body")

    if idempotency_key:
        prior = _idempotency.get("artifact_upload", f"{client_id}:{idempotency_key}")
        if prior:
            return ArtifactUploadResponse(**prior["outcome"])

    store = get_artifact_store()
    content_hash = hashlib.sha256(body).hexdigest()
    deduplicated = store.exists(content_hash)
    stored_hash = store.put_bytes(body)
    if stored_hash != content_hash:
        raise HTTPException(status_code=500, detail="Artifact hash mismatch after store")
    uri = store.uri_for(content_hash)
    artifact_id = f"{artifact_type}:{content_hash[:16]}"
    created_at = time.time()
    manifest = ArtifactManifestV2(
        artifact_type=artifact_type,
        artifact_id=artifact_id,
        content_hash=content_hash,
        byte_size=len(body),
        storage_uri=uri,
        media_type=media_type,
        serialization_format=serialization_format,
        created_at=created_at,
        producer_node=client_id,
        round_id=round_id,
        parent_id=parent_id,
    )
    repo = get_artifact_repository()
    repo.put_manifest(
        ArtifactRecord(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            content_hash=content_hash,
            byte_size=len(body),
            storage_uri=uri,
            media_type=media_type,
            created_at=created_at,
            parent_id=parent_id,
            metadata={"producer_node": client_id, "round_id": round_id},
        )
    )
    response = ArtifactUploadResponse(success=True, manifest=manifest, deduplicated=deduplicated)
    if idempotency_key:
        _idempotency.put(
            "artifact_upload",
            f"{client_id}:{idempotency_key}",
            response.model_dump(),
        )
    return response


@router.get("/v2/artifacts/{content_hash}")
async def download_artifact(
    content_hash: str,
    client_id: str = Query(...),
    api_key: Optional[str] = Query(None),
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> Response:
    _require_protocol(x_protocol_version)
    resolved = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        query_api_key=api_key,
    )
    _auth_or_401(client_id, resolved)
    store = get_artifact_store()
    if not store.exists(content_hash):
        raise HTTPException(status_code=404, detail="Artifact not found")
    data = store.open_read(content_hash).read()
    return Response(content=data, media_type="application/octet-stream")


@router.post("/v2/updates", response_model=UpdateByArtifactResponse)
async def submit_update_by_artifact(
    request: UpdateByArtifactRequest,
    x_api_key: Optional[str] = Header(None, alias="X-Api-Key"),
    authorization: Optional[str] = Header(None),
    x_protocol_version: Optional[str] = Header(None, alias="X-Protocol-Version"),
) -> UpdateByArtifactResponse:
    _require_protocol(x_protocol_version or PROTOCOL_VERSION)
    resolved = extract_api_key(
        x_api_key=x_api_key,
        authorization=authorization,
        body_api_key=request.api_key,
    )
    _auth_or_401(request.client_id, resolved)

    prior = _idempotency.get("update", f"{request.client_id}:{request.idempotency_key}")
    if prior:
        data = dict(prior["outcome"])
        data["replayed"] = True
        return UpdateByArtifactResponse(**data)

    store = get_artifact_store()
    if not store.exists(request.content_hash):
        raise HTTPException(
            status_code=400,
            detail="Unknown content_hash; upload artifact via POST /v2/artifacts first",
        )

    # Compatibility: load bytes and feed classic validator if JSON text, else wrap hash ref.
    raw = store.open_read(request.content_hash).read()
    try:
        weight_delta = raw.decode("utf-8")
        json.loads(weight_delta)  # validate JSON text deltas
    except (UnicodeDecodeError, json.JSONDecodeError):
        weight_delta = json.dumps(
            {
                "artifact_content_hash": request.content_hash,
                "byte_size": len(raw),
            }
        )

    validator = router.update_validator  # type: ignore[attr-defined]
    is_valid, reason = validator.validate(
        request.client_id,
        request.round_id,
        weight_delta,
        api_key=resolved,
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Update rejected: {reason}")

    rm = router.round_manager  # type: ignore[attr-defined]
    if not rm.add_update(request.client_id, request.round_id, weight_delta):
        raise HTTPException(status_code=400, detail="Failed to add update to round")

    outcome = UpdateByArtifactResponse(
        success=True,
        message=f"Update accepted for round {request.round_id}",
        replayed=False,
    )
    _idempotency.put(
        "update",
        f"{request.client_id}:{request.idempotency_key}",
        {"success": outcome.success, "message": outcome.message, "replayed": False},
    )
    return outcome
