"""
Durable JSON state store for coordinator identity and pending updates.

Persists across process restarts so volunteer/edge clients can reconnect
without losing API keys or in-flight round updates.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional


class StateStore:
    """Thread-safe JSON file persistence for coordinator durable state."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            coordinator_dir = Path(__file__).resolve().parent.parent.parent
            path = str(coordinator_dir / "data" / "state.json")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {
            "auth": {"client_keys": {}, "public_keys": {}},
            "clients": [],
            "pending_updates": {},
            "next_round_id": 1,
        }
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._save_unlocked()
                return
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data.update(raw)
                    self._data.setdefault("auth", {"client_keys": {}, "public_keys": {}})
                    self._data["auth"].setdefault("client_keys", {})
                    self._data["auth"].setdefault("public_keys", {})
                    self._data.setdefault("clients", [])
                    self._data.setdefault("pending_updates", {})
                    self._data.setdefault("next_round_id", 1)
            except (json.JSONDecodeError, OSError):
                self._save_unlocked()

    def _save_unlocked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def save(self) -> None:
        with self._lock:
            self._save_unlocked()

    def get_client_keys(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._data.get("auth", {}).get("client_keys", {}))

    def set_client_key(self, client_id: str, api_key: str) -> None:
        with self._lock:
            self._data.setdefault("auth", {}).setdefault("client_keys", {})[client_id] = api_key
            clients = set(self._data.get("clients", []))
            clients.add(client_id)
            self._data["clients"] = sorted(clients)
            self._save_unlocked()

    def get_public_keys(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._data.get("auth", {}).get("public_keys", {}))

    def set_public_key(self, client_id: str, public_key: str) -> None:
        with self._lock:
            self._data.setdefault("auth", {}).setdefault("public_keys", {})[client_id] = public_key
            self._save_unlocked()

    def remove_client(self, client_id: str) -> None:
        with self._lock:
            keys = self._data.setdefault("auth", {}).setdefault("client_keys", {})
            keys.pop(client_id, None)
            pubs = self._data.setdefault("auth", {}).setdefault("public_keys", {})
            pubs.pop(client_id, None)
            clients = set(self._data.get("clients", []))
            clients.discard(client_id)
            self._data["clients"] = sorted(clients)
            self._save_unlocked()

    def get_clients(self) -> list[str]:
        with self._lock:
            return list(self._data.get("clients", []))

    def get_pending_updates(self) -> Dict[str, list]:
        with self._lock:
            return dict(self._data.get("pending_updates", {}))

    def set_pending_updates(self, updates: Dict[str, list]) -> None:
        with self._lock:
            self._data["pending_updates"] = updates
            self._save_unlocked()

    def get_next_round_id(self) -> int:
        with self._lock:
            return int(self._data.get("next_round_id", 1))

    def set_next_round_id(self, value: int) -> None:
        with self._lock:
            self._data["next_round_id"] = value
            self._save_unlocked()
