# Protocol V2 (Design)

Versioned, typed control plane. Large tensors **never** as nested JSON in the production path.

## Negotiation

- Clients send `protocol_version` major.minor
- Reject incompatible major; accept additive minor
- Legacy v1 JSON routes remain behind compatibility adapters until deprecated

## Message families

Node: registration, capability advertisement, heartbeat, assignment request/response, upload init/complete, evaluation result, job claim, lease extend, job result, error.

Operator: experiment/round/job CRUD, aggregate triggers, policy, audit export.

## Artifact manifest (required fields)

Schema version, artifact type/id, content hash, byte size, storage reference, media type, serialization format (prefer `safetensors`), compression, architecture id, model config hash, base artifact id/hash, framework versions, dtype, tensor names/shapes, created_at, producer node, run/round/job ids, provenance/signature ref.

## Idempotency

All retryable mutations carry `idempotency_key`. Duplicate submissions return the original outcome.

## Transport

- Control: HTTPS JSON/Protobuf
- Data: short-lived authorized URLs to object store; checksum verify before accept
- Payload size limits **before** full body parse where practical
- Request ID + correlation ID on every call

## Migration from v1

| v1 | v2 |
|----|-----|
| `weight_delta` JSON lists | Artifact upload + update manifest |
| API key query param | Header / signed requests |
| Open register returns key | Proof-of-possession / invite |
| In-memory round | Persisted Round + Assignment |

Implementation starts in Milestone 2; Milestone 1 adds manifests wrapping existing files.

## Implemented (Milestone 2)

- Version negotiation via `X-Protocol-Version` / `GET /protocol`
- Header credentials: `X-Api-Key` or `Authorization: Bearer` (query/body adapters remain)
- Optional Ed25519 `public_key` on register; helpers in `protocol/identity.py`
- Binary path: `POST /v2/artifacts` → content-addressed store; `POST /v2/updates` by hash + `idempotency_key`
- Legacy nested JSON `/update` still accepted for demo clients
