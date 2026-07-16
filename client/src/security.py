"""
Security Module for Client

Handles API key management and persistence for volunteer/edge reconnect.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _default_key_path() -> Path:
    override = os.getenv("CLIENT_API_KEY_FILE", "").strip()
    if override:
        return Path(override)
    # Persist under client/data next to src/
    client_root = Path(__file__).resolve().parent.parent
    return client_root / "data" / "api_key"


class ClientSecurity:
    """Manages client-side API key load/save."""

    def __init__(self):
        self.key_path = _default_key_path()
        self.api_key: Optional[str] = self._load_api_key()

    def _load_api_key(self) -> Optional[str]:
        env_key = os.getenv("CLIENT_API_KEY", "").strip()
        if env_key:
            return env_key
        try:
            if self.key_path.exists():
                key = self.key_path.read_text(encoding="utf-8").strip()
                return key or None
        except OSError:
            return None
        return None

    def save_api_key(self, api_key: str) -> None:
        """Persist API key to disk for reconnect after restart."""
        self.api_key = api_key.strip()
        try:
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_text(self.api_key + "\n", encoding="utf-8")
            try:
                os.chmod(self.key_path, 0o600)
            except OSError:
                pass
        except OSError as e:
            print(f"[Security] Warning: could not persist API key: {e}")

    def get_api_key(self) -> Optional[str]:
        return self.api_key

    def has_api_key(self) -> bool:
        return self.api_key is not None

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ValueError(
                "CLIENT_API_KEY not set and no key file found. "
                "Register once or set CLIENT_API_KEY / CLIENT_API_KEY_FILE."
            )
        return self.api_key


_security = ClientSecurity()


def get_api_key() -> Optional[str]:
    return _security.get_api_key()


def require_api_key() -> str:
    return _security.require_api_key()


def has_api_key() -> bool:
    return _security.has_api_key()


def save_api_key(api_key: str) -> None:
    _security.save_api_key(api_key)
