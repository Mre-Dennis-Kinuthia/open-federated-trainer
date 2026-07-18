# Migration Plan

## Goals

- Never silently discard rounds, jobs, models, adapters, keys, or balances
- Keep `METADATA_BACKEND=json` as default for local demo
- Provide import path into Postgres when enabled
- Deprecate, don’t delete, legacy APIs until Protocol V2 clients exist

## JSON → SQL

1. Freeze write path behind repository interfaces (M1)
2. `python -m tools.import_json_state` (or `fedcompute migrate json-state`) reads:
   - `coordinator/data/state.json`
   - `coordinator/data/jobs.json`
   - `coordinator/data/lora_rounds.json`
   - `coordinator/data/geo_presence.json` (optional)
   - `coordinator/models/*.json`, `adapters/*.json`
3. Registers artifact manifests with sha256 of file bytes
4. Idempotent re-run (upsert by natural keys)

## Feature flags

| Flag | Default | Meaning |
|------|---------|---------|
| `METADATA_BACKEND` | `json` | `json` \| `sqlite` \| `sql` \| `postgres` |
| `ARTIFACT_STORE` | `local` | `local` \| `s3` \| `minio` |
| `REQUIRE_OPERATOR_KEY` | false (dev) | Refuse start if no operator key |
| `AGGREGATION_STRATEGY` | `fedavg` | `fedavg` \| `adaptive` \| `robust` |
| `LORA_AGG_STRATEGY` | `delta_svd` | `delta_svd` \| `param_fedavg` |
| `JOB_LEASE_SECONDS` | `300` | Default job lease TTL |
| `JOB_LEASE_HEARTBEAT_SECONDS` | `60` | Worker lease extend interval |
| `COMPUTE_RUNTIME` | `local_import` | `local_import` \| `container` (stub) |

## Deprecations

| Legacy | Replacement | Window |
|--------|-------------|--------|
| API key query param | Header | Soft warn → reject |
| Nested JSON updates | Artifact upload | After M2 clients |
| Open register key return | PoP register | M0 hard break for attackers; clients with saved keys OK |
| Public job results | Operator or node-scoped | M0 redact / auth |

## Backup / restore

- Local demo: copy `coordinator/data/`, `models/`, `adapters/`, `artifacts/`
- Production: `./scripts/backup.sh` (`pg_dump` + optional `BACKUP_MINIO=1`) and `./scripts/restore.sh`
- Docs: `docs/deploy/PRODUCTION_DEPLOY.md`

## Rollback

- Keep JSON files after import until validated
- Flip `METADATA_BACKEND` back to `json` if Postgres unhealthy
- Do not delete source JSON in import tool
