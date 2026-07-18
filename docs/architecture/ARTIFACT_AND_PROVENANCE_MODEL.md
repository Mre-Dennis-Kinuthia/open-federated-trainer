# Artifact and Provenance Model

## Goal

Immutable, content-addressed artifacts with manifests and lineage:

```text
base model → assignment → node contribution → validation → aggregation → evaluation → published model
```

## Artifact types

Global model weights, weight deltas, LoRA adapters, evaluation reports, job inputs/outputs, checkpoints, process logs (operator-scoped).

## Storage

| Environment | Backend |
|-------------|---------|
| Local demo | `LocalFilesystemArtifactStore` under `coordinator/artifacts/` |
| Production | S3-compatible (MinIO / cloud) |

Bytes never mutated after publish. Aliases (`latest`, `production`, `candidate`) are pointers, not rewritten blobs.

## Manifest

See PROTOCOL_V2.md. Minimum for Milestone 1: type, id, sha256, size, path/uri, created_at, optional parent id, media_type=`application/json` for legacy model files.

## Formats

- Prefer `safetensors` for tensors (Milestone 2+)
- Avoid pickle for untrusted artifacts
- Verify hash before load
- Legacy `models/model_vN.json` remains readable via importer that computes hash and registers manifest

## Provenance record

Links artifact id → round/job/aggregation run → contributing node ids (policy-scoped) → validation outcome → evaluation metrics → publisher actor.

## Audit

Append-oriented `AuditEvent` for auth, aggregate, publish, job cancel, launcher use, credential revoke — never log secrets or full tensors.
