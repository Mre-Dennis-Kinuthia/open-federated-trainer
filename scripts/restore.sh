#!/usr/bin/env bash
# Restore from a backup directory created by scripts/backup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" || ! -d "$SRC" ]]; then
  echo "Usage: $0 <backup-directory>" >&2
  exit 1
fi

echo "==> Restoring from $SRC"

if [[ -f "$SRC/metadata.dump" ]]; then
  if [[ -n "${DATABASE_URL:-}" ]]; then
    PGURL="${DATABASE_URL/postgresql+psycopg2:\/\//postgresql:\/\/}"
    echo "==> pg_restore (DATABASE_URL)"
    pg_restore --clean --if-exists --no-owner --dbname="$PGURL" "$SRC/metadata.dump"
  else
    echo "==> pg_restore via docker postgres"
    docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.prod.yml" exec -T postgres \
      pg_restore -U fedcompute -d fedcompute --clean --if-exists --no-owner <"$SRC/metadata.dump"
  fi
elif [[ -d "$SRC/coordinator" ]]; then
  echo "==> Restoring local coordinator data dirs (stop coordinator first)"
  for d in data models adapters artifacts metrics; do
    if [[ -d "$SRC/coordinator/$d" ]]; then
      mkdir -p "$ROOT/coordinator"
      rsync -a --delete "$SRC/coordinator/$d/" "$ROOT/coordinator/$d/"
    fi
  done
fi

if [[ -d "$SRC/artifacts" && "${RESTORE_MINIO:-0}" == "1" ]]; then
  echo "==> MinIO restore"
  mc mirror --overwrite "$SRC/artifacts" "local/${S3_BUCKET:-fedcompute-artifacts}"
fi

echo "OK: restore complete. Restart coordinator replicas and verify GET /ready"
