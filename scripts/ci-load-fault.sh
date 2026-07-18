#!/usr/bin/env bash
# CI / local load+fault harness: 2 coordinator processes + RR proxy + k6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${TMPDIR:-/tmp}/fedcompute-ci-load-$$"
mkdir -p "$WORKDIR/artifacts" "$WORKDIR/data" "$WORKDIR/models"
export PYTHONPATH="$ROOT/coordinator/src"
export METADATA_BACKEND=sqlite
export SHARED_STATE=auto
export DATABASE_URL="sqlite:///$WORKDIR/ha.db"
export ARTIFACT_STORE=local
export ARTIFACT_STORE_ROOT="$WORKDIR/artifacts"
export REQUIRE_OPERATOR_KEY=false
export OPERATOR_API_KEY=""
export ENABLE_ASYNC_ROUNDS=false
export GEO_LOOKUP_DISABLED=true
export CORS_ORIGINS="*"

PY="${PYTHON:-$ROOT/client/venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
if ! "$PY" -c "import uvicorn" 2>/dev/null; then
  echo "Installing uvicorn into $PY environment..."
  "$PY" -m pip install 'uvicorn[standard]>=0.24.0' -q
fi

BASE_URL="${BASE_URL:-http://127.0.0.1:18080}"
export BASE_URL

cleanup() {
  [[ -n "${PROXY_PID:-}" ]] && kill "$PROXY_PID" 2>/dev/null || true
  [[ -n "${PID_A:-}" ]] && kill "$PID_A" 2>/dev/null || true
  [[ -n "${PID_B:-}" ]] && kill "$PID_B" 2>/dev/null || true
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "==> Starting coordinator replicas (logs under $WORKDIR)"
export PYTHONPATH="$ROOT/coordinator/src"
export METADATA_BACKEND=sqlite
export SHARED_STATE=auto
export DATABASE_URL="sqlite:///$WORKDIR/ha.db"
export ARTIFACT_STORE=local
export ARTIFACT_STORE_ROOT="$WORKDIR/artifacts"
export ENABLE_ASYNC_ROUNDS=false
export GEO_LOOKUP_DISABLED=true
export CORS_ORIGINS="*"

(
  cd "$ROOT/coordinator"
  exec "$PY" -m uvicorn main:app --host 127.0.0.1 --port 18001
) >"$WORKDIR/a.log" 2>&1 &
PID_A=$!
(
  cd "$ROOT/coordinator"
  exec "$PY" -m uvicorn main:app --host 127.0.0.1 --port 18002
) >"$WORKDIR/b.log" 2>&1 &
PID_B=$!

"$PY" - <<'PY' &
import itertools, http.client, socketserver
from http.server import BaseHTTPRequestHandler

UPSTREAMS = [("127.0.0.1", 18001), ("127.0.0.1", 18002)]
cycle = itertools.cycle(UPSTREAMS)

class Handler(BaseHTTPRequestHandler):
    def _proxy(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        last_err = None
        for _ in range(len(UPSTREAMS)):
            host, port = next(cycle)
            try:
                conn = http.client.HTTPConnection(host, port, timeout=30)
                conn.request(self.command, self.path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() in ("transfer-encoding", "connection"):
                        continue
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)
                return
            except OSError as exc:
                last_err = exc
                continue
        self.send_error(502, str(last_err or "no upstream"))
    def do_GET(self): self._proxy()
    def do_POST(self): self._proxy()
    def do_PUT(self): self._proxy()
    def log_message(self, *args): pass

class Reusable(socketserver.ThreadingTCPServer):
    allow_reuse_address = True

Reusable(("127.0.0.1", 18080), Handler).serve_forever()
PY
PROXY_PID=$!

echo "==> Waiting for /ready"
for i in $(seq 1 90); do
  if curl -sf "$BASE_URL/ready" >/dev/null; then
    echo "ready after ${i}s"
    break
  fi
  sleep 1
  if [[ "$i" -eq 90 ]]; then
    echo "timeout waiting for ready" >&2
    echo "--- a.log ---" >&2
    cat "$WORKDIR/a.log" >&2 || true
    echo "--- b.log ---" >&2
    cat "$WORKDIR/b.log" >&2 || true
    exit 1
  fi
done

run_k6() {
  local script="$1"
  if command -v k6 >/dev/null 2>&1; then
    k6 run -e "BASE_URL=$BASE_URL" "$script"
    return
  fi
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker run --rm --network host -v "$ROOT/deploy/load:/scripts:ro" \
      -e "BASE_URL=$BASE_URL" grafana/k6:0.54.0 run "/scripts/$(basename "$script")"
    return
  fi
  echo "k6/docker unavailable — running curl smoke for $(basename "$script")"
  case "$(basename "$script")" in
    health_ready.js)
      for _ in $(seq 1 20); do
        curl -sf "$BASE_URL/health" >/dev/null
        curl -sf "$BASE_URL/ready" >/dev/null
      done
      ;;
    artifact_roundtrip.js)
      REG=$(curl -sf -X POST "$BASE_URL/v2/node/register" \
        -H 'Content-Type: application/json' -H 'X-Protocol-Version: 2.0' \
        -d "{\"client_name\":\"curl-smoke-$RANDOM\",\"protocol_version\":\"2.0\"}")
      KEY=$(echo "$REG" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['api_key'])")
      CID=$(echo "$REG" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['client_id'])")
      UP=$(curl -sf -X POST "$BASE_URL/v2/artifacts?client_id=$CID&artifact_type=weight_delta" \
        -H "X-Api-Key: $KEY" -H 'X-Protocol-Version: 2.0' \
        -H 'Content-Type: application/octet-stream' --data-binary 'smoke-bytes')
      HASH=$(echo "$UP" | "$PY" -c "import sys,json; print(json.load(sys.stdin)['manifest']['content_hash'])")
      curl -sf "$BASE_URL/v2/artifacts/$HASH?client_id=$CID" \
        -H "X-Api-Key: $KEY" -H 'X-Protocol-Version: 2.0' >/dev/null
      ;;
    *)
      echo "No curl fallback for $script" >&2
      exit 1
      ;;
  esac
}

echo "==> k6 health_ready"
run_k6 "$ROOT/deploy/load/health_ready.js"

echo "==> k6 artifact_roundtrip"
run_k6 "$ROOT/deploy/load/artifact_roundtrip.js"

echo "==> Fault: kill replica A (port 18001)"
kill "$PID_A" 2>/dev/null || true
sleep 2
# Ensure port is down even if PID was a wrapper
if command -v fuser >/dev/null 2>&1; then
  fuser -k 18001/tcp 2>/dev/null || true
fi
PID_A=""

echo "==> k6 health_ready after fault"
run_k6 "$ROOT/deploy/load/health_ready.js"

echo "OK: load + fault harness passed"
