# Competitive Capability Matrix

Comparison axes vs a production federated-compute platform. Status: **Have** / **Partial** / **Missing** / **Planned**.

| Capability | Status | Notes |
|------------|--------|-------|
| Classic FedAvg | Have | Real PyTorch; shared base verification |
| Pluggable trainers | Partial | `MODEL_MODULE`; no allowlist; silent fallback (fixing) |
| Federated LoRA | Partial | Durable rounds; ΔW+SVD agg default; manifests; task-aware eval hooks; JSON adapters remain |
| Job queue (inference/label/compute) | Partial | Durable; JobSpec; leases+heartbeat; attempts; dataset aliases; weak isolation |
| Private dataset locality | Partial | Local loaders; exception via job `inputs` |
| Operator console | Have | Live API; `#/status` / `#/privacy`; a11y tests |
| Public network viz | Partial | Real geo presence; decorative arcs; truthful landing copy |
| Durable metadata (SQL) | Partial | JSON default; Postgres/SQLite via `METADATA_BACKEND` |
| Binary artifacts / object store | Partial | Local store + `/v2/artifacts`; S3/MinIO via `ARTIFACT_STORE=s3` |
| Protocol versioning | Partial | V2 negotiation + header auth; legacy JSON adapters remain |
| Node cryptographic identity | Partial | Optional Ed25519 public key; API keys still primary |
| RBAC / orgs / federations | Missing | Single operator key |
| Secure aggregation | Missing | Planned |
| Differential privacy | Missing | Planned |
| Strategy engine (FedProx, robust agg) | Partial | `fedavg` / `adaptive` / `robust` via `AGGREGATION_STRATEGY` |
| Federated evaluation workload | Missing | LoRA eval only |
| Sandboxed compute (containers) | Partial | Runtime iface + allowlist harden; container stub |
| Verification / canaries / N-of-M | Partial | Canary fingerprint + N-of-M quorum stub |
| Reputation ledger durable | Partial | SQL when SHARED_STATE/SQL backend; JSON demo still RAM |
| Observability (OTel/Prom) | Missing | Struct logs partial |
| HA / multi-replica coordinator | Partial | SQL shared reputation/geo/rounds + aggregate locks; metrics still local |
| CI/CD | Partial | GitHub Actions pytest/ui + k6 load-fault |
| CLI / SDK | Missing | Planned `fedcompute` |
| Simulator / benchmarks | Partial | Shell simulate; not non-IID suite |

## Positioning vs “another FL library”

This repo already includes **jobs, LoRA, UI, launcher, reputation** — keep that breadth. Gap vs mature platforms (Flower, OpenFL, Fate, etc.) is durability, protocol, identity, privacy primitives, and ops — not the idea of multi-workload coordination.
