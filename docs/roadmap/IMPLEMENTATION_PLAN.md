# Implementation Plan

Vertical milestones. After each: tests green, docs updated, quick start works.

| Milestone | Deliver | Exit criteria |
|-----------|---------|---------------|
| **M0** Audit & guardrails | Audit docs, AGENTS/rules, safe P0 fixes | Docs present; registration/size/geo/trainer/UI fixes; pytest+build green |
| **M1** Durable metadata & artifacts | Repository interfaces, optional Postgres, ArtifactStore local FS, JSON importer, default JSON | Dual backend flag; import tool; demo unchanged |
| **M2** Protocol V2 & node identity | Typed schemas, binary transfer, keypair identity, legacy adapters | AuthZ tests; version negotiation |
| **M3** Reliable rounds & strategies | Persisted state machine, idempotent aggregate, FedAvg behind strategy interface + one adaptive + one robust | Restart mid-round tests |
| **M4** Correct federated LoRA | Adapter manifests, corrected agg strategy, task-aware eval, isolated merge | Numerical tests |
| **M5** General job runtime | JobSpec, leases, attempts, dataset aliases | Lease expiry / cancel tests |
| **M6** Sandboxing & verification | Runtime abstraction, canaries, N-of-M | No arbitrary imports from network |
| **M7** Operator & public UX | Authoritative pages, privacy displays, a11y tests | No fake stats |
| **M8** Production deployment | Replicas, PG, object store, Helm/Docker, backup/restore | Load/fault report |
| **M9** HA shared state + TLS + CI | SQL reputation/geo/rounds, TLS edge, k6 CI | Cross-replica tests + CI green |

## M0 → M1 handoff (status)

Implemented:

- `coordinator/src/persistence/` — protocols, JSON adapters, SQLAlchemy models, SQL repos
- `coordinator/src/artifacts/` — content-addressed local store
- `METADATA_BACKEND=json` default; `postgres`/`sqlite`/`sql` select SQL repos
- Alembic: `coordinator/alembic/` + `001_initial` (nodes, rounds, jobs, artifacts)
- Classic rounds persisted via `JsonRoundRepository` / SQL and restored on boot
- `python -m tools.import_json_state` from `coordinator/src`
- Compose profile `durable`: Postgres 16 + MinIO (`docker compose --profile durable up`)
- Tests: `tests/test_milestone1_persistence.py`, `tests/test_milestone1_sql_and_rounds.py`

Default `docker compose up` / `./run.sh` remains JSON + local artifacts (quick start unchanged).

## M1 → M2 handoff (status)

Implemented:

- `coordinator/src/protocol/` — version negotiation, credential extraction, Ed25519 identity helpers, idempotency store, V2 schemas
- `GET /protocol`, `POST /v2/node/register`, `POST /v2/artifacts`, `GET /v2/artifacts/{hash}`, `POST /v2/updates`
- Legacy `/task` and `/update` accept `X-Api-Key` / `Authorization: Bearer` (query/body remain adapters)
- Optional `public_key` on register (v1 + v2); persisted in `state.json`
- Client sends `X-Api-Key` + `X-Protocol-Version: 2.0` by default
- Tests: `tests/test_milestone2_protocol.py`

Nested JSON `/update` remains for demo clients until fully migrated.

## M2 → M3 handoff (status)

Implemented:

- `aggregation/strategies.py` — `FedAvgStrategy`, `AdaptiveFedAvgStrategy` (sample-weighted), `RobustTrimmedMeanStrategy`
- `AGGREGATION_STRATEGY=fedavg|adaptive|robust` (default `fedavg`); optional `ROBUST_TRIM_RATIO`
- Idempotent `Aggregator.aggregate`: CLOSED rounds return published model; `reserved_version` / `published_version` on round metadata
- Crash mid-aggregate: `AGGREGATING` restores as `COLLECTING` + `resume_after_crash`; `reconcile_after_restart()` on boot
- Atomic model writes (`tmp` + `os.replace`)
- Tests: `tests/test_milestone3_rounds.py`

## M3 → M4 handoff (status)

Implemented:

- Default LoRA agg `LORA_AGG_STRATEGY=delta_svd` — FedAvg on ΔW=B@A then SVD re-factor (legacy `param_fedavg` kept)
- `AdapterManifest` + artifact registry on publish; adapter JSON includes `manifest`
- Task-aware eval: `task_type=causal_lm|seq_cls` on rounds; seq_cls reports accuracy when labels present
- Isolated merge helpers (`aggregation/merge.py`); eval loads adapters onto a fresh PEFT copy
- LoRA states: `EVALUATING` / `REJECTED`; idempotent re-aggregate when `published_version` set
- Persist lock on `lora_rounds.json`
- Tests: `tests/test_milestone4_lora.py` (numerical ΔW vs param FedAvg)

## M4 → M5 handoff (status)

Implemented:

- `JobSpec` validation on create (inference/label/compute payload requirements)
- Leases: `lease_expires_at`, `JOB_LEASE_SECONDS`, per-job `lease_seconds`, `POST /jobs/{id}/lease` heartbeat
- `JobAttempt` history on claim / expire / complete / cancel / fail
- Worker lease heartbeat thread (`JOB_LEASE_HEARTBEAT_SECONDS`)
- Dataset aliases: coordinator registry (`/datasets/aliases`); workers resolve via `DATASET_ALIASES` / `DATASET_ALIAS_<name>`
- Cancel + lease-expiry → requeue / dead-letter; failed results requeue until `max_attempts`
- Tests: `tests/test_milestone5_jobs.py`

## M5 → M6 handoff (status)

Implemented:

- `client/src/runtime/` — `ComputeRuntime` with `LocalImportRuntime` (default) and `ContainerRuntime` stub (`COMPUTE_RUNTIME`)
- Entrypoint sanitization: reject paths/URLs/`..`; allowlist still required
- Canary verification: `payload.verification.mode=canary` + `expected_fingerprint`
- N-of-M quorum stub: `mode=n_of_m` with `n`/`m`; requeue until agree or fail
- Job fields: `validation`, `candidate_results`
- Tests: `tests/test_milestone6_sandbox.py`

## M6 → M7 handoff (status)

Implemented:

- `#/status` — live “this instance” status from `/dashboard/overview` + activity
- `#/privacy` — privacy model page with workload disclosures + capability flags
- `PrivacyDisclosure` — stays/leaves/visible copy; truthful SecAgg/DP “not available”
- Landing — truthful stats (jobs queued vs tracked); arcs labeled decorative; privacy/status footer links
- Jobs panel — inline disclosure; redacted payload/result UX
- Vitest + jest-axe: `PrivacyDisclosure.test.tsx`, `PublicPages.test.tsx`
- Exit: no fake fleet-wide stats; public pages state instance scope

## M7 → M8 handoff (status)

Implemented:

- `ARTIFACT_STORE=s3|minio` — `S3ArtifactStore` (boto3; MinIO via `S3_ENDPOINT_URL`)
- `GET /ready` — metadata DB + artifact store readiness (k8s/Compose probes)
- `REQUIRE_OPERATOR_KEY=true` refuses start without `OPERATOR_API_KEY`
- Compose production profile: `docker-compose.prod.yml` (Postgres, MinIO, coordinator-a/b, nginx `:8080`)
- Helm chart: `deploy/helm/fed-compute` (replicas, secrets, probes, optional Ingress)
- Backup/restore: `scripts/backup.sh`, `scripts/restore.sh`
- Docs: `docs/deploy/PRODUCTION_DEPLOY.md`, `docs/deploy/LOAD_AND_FAULT_REPORT.md`
- Tests: `tests/test_milestone8_deploy.py`

Default `docker compose up` / JSON local demo unchanged.

## M8 → M9 handoff (status)

Implemented:

- Alembic `002_ha_shared_state` — reputation, incentives, geo_presence, rate_limit_buckets, ha_locks
- SQL-backed shared state when `SHARED_STATE=auto` + SQL `METADATA_BACKEND`
- RoundManager refresh/write-through + `try_begin_aggregating` row lock
- Job claim uses named lock under shared state
- TLS lab edge: `scripts/gen-dev-certs.sh`, nginx HTTPS `:8443`
- Helm ingress TLS values + cert-manager annotation docs
- CI: `.github/workflows/ci.yml` (pytest, ui, k6 load-fault)
- Tests: `tests/test_milestone9_ha_state.py`
