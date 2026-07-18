#!/usr/bin/env bash
# Stop host-side train/worker processes started from the operator console
# (ui-train-*) or local venv clients that otherwise keep posting to :8000.
set -euo pipefail

patterns=(
  "client/src/client.py"
  "client/src/worker.py"
  "open-federated-trainer/client/venv/bin/python.*/client.py"
  "open-federated-trainer/client/venv/bin/python.*/worker.py"
)

stopped=0
for pat in "${patterns[@]}"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    # shellcheck disable=SC2009
    pgrep -af "$pat" || true
    pkill -f "$pat" || true
    stopped=1
  fi
done

if [[ "$stopped" -eq 1 ]]; then
  echo "Stopped local train/worker processes."
else
  echo "No matching local train/worker processes."
fi
