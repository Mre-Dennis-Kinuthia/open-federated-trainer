# Load & fault report (Milestone 8–9)

Exit evidence for production deployment scaffolding and HA shared state.

## Topology under test

- 2× coordinator replicas (Compose production profile or `scripts/ci-load-fault.sh`)
- Postgres 16 (Compose) or shared SQLite (CI harness) for SQL metadata + HA tables
- MinIO for `ARTIFACT_STORE=s3` (Compose); local FS in CI harness
- Operator key required in Compose (`REQUIRE_OPERATOR_KEY=true`)
- TLS edge on Compose `:8443` (self-signed lab certs via `scripts/gen-dev-certs.sh`)

## What survives a replica kill

| Workload | Shared store | Expected after killing one replica |
|----------|--------------|-------------------------------------|
| Artifact upload/download (`/v2/artifacts`) | MinIO / shared FS | Continues via remaining replica |
| SQL-backed job/round metadata | Postgres / SQLite | Continues when `METADATA_BACKEND=postgres\|sqlite` |
| Reputation / incentives / geo presence | SQL HA tables | Shared across replicas (`SHARED_STATE=auto`) |
| Classic round live maps | SQL rounds + refresh | Visible cross-replica; aggregate lock single-winner |
| JSON demo path (`METADATA_BACKEND=json`) | Local volume | **Not** multi-replica safe |
| Rate limits | SQL buckets when shared | Shared counters |
| Metrics dashboard counters | Process RAM (+ disk) | Still per-replica approximate |

## Automated load / fault (CI)

GitHub Actions: [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)

- `pytest` — full backend suite
- `ui` — Vitest + production build
- `load-fault` — `scripts/ci-load-fault.sh` (2 uvicorn replicas + RR proxy + k6)

k6 scripts:

| Script | Purpose |
|--------|---------|
| [`deploy/load/health_ready.js`](../load/health_ready.js) | `/health` + `/ready` p95 &lt; 500ms |
| [`deploy/load/artifact_roundtrip.js`](../load/artifact_roundtrip.js) | Protocol V2 artifact upload/download |

Manual Compose fault drills remain valid:

1. `docker kill federated-coordinator-a` → `curl -sk https://localhost:8443/health`
2. Artifact continuity across replicas
3. Postgres stop → `/ready` 503; `/health` may still 200
4. `./scripts/backup.sh` / `restore.sh`

## Load notes

- Nginx `client_max_body_size 64m` aligns with artifact upload caps.
- Prefer Protocol V2 binary artifacts in production (nested JSON `/update` remains a DoS risk).
- Torch/transformers images are heavy; CI load harness runs the API from a Python venv without rebuilding Docker when possible.

## Remaining gaps

- Dashboard **metrics** maps are still process-local (approximate under multi-replica).
- Compose TLS is **self-signed lab**; real CA via Ingress/cert-manager (Helm values document issuer annotations).
- Full capacity benchmarking (sustained RPS / GPU jobs) is out of scope for this report.
