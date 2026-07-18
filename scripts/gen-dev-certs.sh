#!/usr/bin/env bash
# Generate self-signed lab certs for Compose TLS edge (HTTPS :8443).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CERT_DIR="${CERT_DIR:-$ROOT/deploy/certs}"
mkdir -p "$CERT_DIR"

DAYS="${CERT_DAYS:-825}"
CN="${CERT_CN:-localhost}"

if command -v mkcert >/dev/null 2>&1; then
  echo "==> mkcert (trusted local CA)"
  (cd "$CERT_DIR" && mkcert -cert-file fullchain.pem -key-file privkey.pem "$CN" 127.0.0.1 ::1)
else
  echo "==> openssl self-signed (use curl -k)"
  openssl req -x509 -nodes -newkey rsa:2048 -days "$DAYS" \
    -keyout "$CERT_DIR/privkey.pem" \
    -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=$CN" \
    -addext "subjectAltName=DNS:$CN,IP:127.0.0.1"
fi

chmod 640 "$CERT_DIR/privkey.pem" || true
echo "OK: $CERT_DIR/fullchain.pem + privkey.pem"
