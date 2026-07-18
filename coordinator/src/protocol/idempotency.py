"""Simple durable idempotency store for retryable mutations."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class IdempotencyStore:
    def __init__(self, path: Optional[str] = None):
        default = Path(__file__).resolve().parents[2] / "data" / "idempotency.json"
        self.path = Path(path or os.getenv("IDEMPOTENCY_PATH", str(default)))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {"entries": {}}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._save_unlocked()
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = raw
                    self._data.setdefault("entries", {})
            except (json.JSONDecodeError, OSError):
                self._save_unlocked()

    def _save_unlocked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def get(self, scope: str, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return (self._data.get("entries") or {}).get(f"{scope}:{key}")

    def put(self, scope: str, key: str, outcome: Dict[str, Any]) -> None:
        with self._lock:
            self._data.setdefault("entries", {})[f"{scope}:{key}"] = {
                "outcome": outcome,
                "stored_at": time.time(),
            }
            self._save_unlocked()
