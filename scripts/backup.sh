#!/usr/bin/env bash
# Backup Postgres metadata (+ optional MinIO sync) for fed-compute production stacks.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${BACKUP_DIR:-$ROOT/backups}/$STAMP"
mkdir -p "$OUT_DIR"

echo "==> Backup directory: $OUT_DIR"

if [[ -n "${DATABASE_URL:-}" ]]; then
  # Strip SQLAlchemy driver prefix for pg_dump
  PGURL="${DATABASE_URL/postgresql+psycopg2:\/\//postgresql:\/\/}"
  echo "==> pg_dump"
  pg_dump --no-owner --format=custom --file="$OUT_DIR/metadata.dump" "$PGURL"
elif docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.prod.yml" ps postgres 2>/dev/null | grep -q Up; then
  echo "==> pg_dump via docker (postgres service)"
  docker compose -f "$ROOT/docker-compose.yml" -f "$ROOT/docker-compose.prod.yml" exec -T postgres \
    pg_dump -U fedcompute -d fedcompute --no-owner --format=custom >"$OUT_DIR/metadata.dump"
else
  echo "==> No DATABASE_URL / running postgres — copying local JSON data dirs"
  for d in data models adapters artifacts metrics; do
    src="$ROOT/coordinator/$d"
    if [[ -d "$src" ]]; then
      mkdir -p "$OUT_DIR/coordinator"
      cp -a "$src" "$OUT_DIR/coordinator/$d"
    fi
  done
  if [[ -d "$ROOT/adapters" ]]; then
    cp -a "$ROOT/adapters" "$OUT_DIR/adapters"
  fi
fi

if [[ "${BACKUP_MINIO:-0}" == "1" ]]; then
  echo "==> MinIO mirror (requires mc configured as alias 'local')"
  mc mirror --overwrite "local/${S3_BUCKET:-fedcompute-artifacts}" "$OUT_DIR/artifacts"
fi

echo "$STAMP" >"$OUT_DIR/BACKUP_ID"
echo "OK: backup written to $OUT_DIR"
