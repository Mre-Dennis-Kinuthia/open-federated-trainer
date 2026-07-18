# Production deploy

How to run fed-compute with Postgres, S3-compatible object storage, and multiple API replicas.

## Quick start stays unchanged

```bash
docker compose up          # JSON + local artifacts, single coordinator :8000
./run.sh                   # local demo
```

## Production Compose profile (TLS lab edge)

```bash
./scripts/stop-local-trainers.sh   # stop host ui-train clients that talk to :8000
./scripts/gen-dev-certs.sh         # self-signed → deploy/certs/
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d --build
curl -sk https://localhost:8443/ready
curl -sk 'https://localhost:8443/dashboard/overview?limit=5'
```

Lab default leaves **operator auth off** (empty `OPERATOR_API_KEY`). To require a key:

```bash
export OPERATOR_API_KEY="$(openssl rand -hex 16)"
export REQUIRE_OPERATOR_KEY=true
docker compose -f docker-compose.yml -f docker-compose.prod.yml --profile production up -d
```

The production profile **does not** start the JSON single-node `coordinator` on `:8000`.
Train traffic goes to `coordinator-a` / `coordinator-b` via `client-1` / `client-2`
(synthetic demo data by default; set `ALLOW_SYNTHETIC_DATA=false` + `DATASET_PATH` for real corpora).

| Component | Role |
|-----------|------|
| `postgres` | Metadata (`METADATA_BACKEND=postgres`) + HA shared state |
| `minio` + `minio-init` | Artifact bucket `fedcompute-artifacts` |
| `coordinator-a` / `coordinator-b` | API replicas (`SHARED_STATE=auto`) |
| `edge` (nginx) | TLS on `:8443`; HTTP `:8080` redirects to HTTPS |
| `client-1` / `client-2` | Demo trainers (synthetic unless `DATASET_PATH` set) |

Compose TLS uses **self-signed** certs for lab use. Production should terminate TLS at a cloud LB / Ingress with a real CA (see Helm).

Env flags:

| Variable | Production value |
|----------|------------------|
| `METADATA_BACKEND` | `postgres` |
| `DATABASE_URL` | `postgresql+psycopg2://…` |
| `ARTIFACT_STORE` | `s3` (alias `minio`) |
| `S3_ENDPOINT_URL` | MinIO or cloud endpoint |
| `S3_BUCKET` / `S3_PREFIX` | bucket + key prefix |
| `S3_ACCESS_KEY` / `S3_SECRET_KEY` | credentials |
| `REQUIRE_OPERATOR_KEY` | `true` |
| `OPERATOR_API_KEY` | required non-empty |
| `SHARED_STATE` | `auto` (SQL-backed reputation/geo/rounds) |

## Helm

```bash
docker build -t fed-compute/coordinator:latest ./coordinator
helm upgrade --install fed-compute deploy/helm/fed-compute \
  --set secrets.OPERATOR_API_KEY="$OPERATOR_API_KEY" \
  --set secrets.DATABASE_URL="$DATABASE_URL" \
  --set replicaCount=2 \
  --set ingress.enabled=true \
  --set ingress.className=nginx \
  --set-string ingress.annotations."cert-manager\.io/cluster-issuer"=letsencrypt-prod
```

Chart expects **external** Postgres and object store. Ingress `tls` in values points at `fed-compute-tls` secret (cert-manager or manual).

## Backup / restore

```bash
./scripts/backup.sh
BACKUP_MINIO=1 ./scripts/backup.sh
./scripts/restore.sh backups/<BACKUP_ID>
```

See also [LOAD_AND_FAULT_REPORT.md](./LOAD_AND_FAULT_REPORT.md).

## Readiness

- `GET /health` — process liveness
- `GET /ready` — metadata DB + artifact store configuration checks
