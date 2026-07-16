#!/bin/bash
# Run the fed-compute ops UI (Vite dev server on :5173)
set -euo pipefail
cd "$(dirname "$0")"

export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  # Prefer Linux Node via nvm (avoid Windows npm on WSL UNC paths)
  # shellcheck disable=SC1090
  . "$NVM_DIR/nvm.sh"
  nvm use 24 >/dev/null 2>&1 || nvm use --lts >/dev/null 2>&1 || true
fi

if [ ! -d node_modules ]; then
  npm install
fi

npm run dev
