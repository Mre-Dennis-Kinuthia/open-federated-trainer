"""Adapter manifests and isolated merge helpers (Milestone 4)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AdapterManifest:
    schema_version: str = "2.0"
    artifact_type: str = "lora_adapter"
    artifact_id: str = ""
    content_hash: str = ""
    byte_size: int = 0
    storage_uri: str = ""
    media_type: str = "application/json"
    serialization_format: str = "json_nested_lists"
    base_model_id: str = ""
    adapter_version: str = ""
    round_id: Optional[int] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    target_modules: List[str] = field(default_factory=list)
    aggregation_strategy: str = "delta_svd"
    task_type: str = "causal_lm"
    tensor_names: List[str] = field(default_factory=list)
    created_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def hash_adapter_state(adapter_state_dict: Dict[str, Any]) -> str:
    payload = json.dumps(adapter_state_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_adapter_manifest(
    *,
    adapter_version: str,
    adapter_state_dict: Dict[str, Any],
    base_model_id: str,
    round_id: int,
    lora_r: int,
    lora_alpha: int,
    target_modules: List[str],
    aggregation_strategy: str,
    task_type: str = "causal_lm",
    storage_uri: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> AdapterManifest:
    content_hash = hash_adapter_state(adapter_state_dict)
    payload = json.dumps(adapter_state_dict, sort_keys=True, separators=(",", ":"))
    return AdapterManifest(
        artifact_id=f"lora:{adapter_version}:{content_hash[:16]}",
        content_hash=content_hash,
        byte_size=len(payload.encode("utf-8")),
        storage_uri=storage_uri or f"file://adapters/model_{adapter_version}.json",
        base_model_id=base_model_id,
        adapter_version=adapter_version,
        round_id=round_id,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        aggregation_strategy=aggregation_strategy,
        task_type=task_type,
        tensor_names=sorted(adapter_state_dict.keys()),
        created_at=time.time(),
        metadata=dict(extra_metadata or {}),
    )


def register_adapter_manifest(manifest: AdapterManifest) -> None:
    """Best-effort register with ArtifactRepository (JSON or SQL)."""
    try:
        from persistence import ArtifactRecord
        from persistence.json_repos import get_artifact_repository

        repo = get_artifact_repository()
        # Manifest metadata only; adapter bytes live in ModelStore / adapters/.
        repo.put_manifest(
            ArtifactRecord(
                artifact_id=manifest.artifact_id,
                artifact_type=manifest.artifact_type,
                content_hash=manifest.content_hash,
                byte_size=manifest.byte_size,
                storage_uri=manifest.storage_uri,
                media_type=manifest.media_type,
                created_at=manifest.created_at,
                parent_id=None,
                metadata=manifest.to_dict(),
            )
        )
    except Exception:
        # Demo path must not fail publish if artifact registry is unavailable.
        pass
