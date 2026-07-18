"""
Authentication Module

Provides API key-based authentication for federated learning clients.
Keys are persisted to disk for volunteer/edge reconnect after restarts.
"""

import os
import secrets
from typing import Dict, Optional, Set

from .state_store import StateStore


class ClientAlreadyRegisteredError(Exception):
    """Raised when a client_id exists and no valid proof-of-possession key was given."""

    def __init__(self, client_id: str):
        self.client_id = client_id
        super().__init__(
            f"Client '{client_id}' is already registered; "
            "present the existing api_key to resume."
        )


class AuthManager:
    """
    Manages API key authentication for clients.

    Keys are kept in memory and mirrored to StateStore when available.
    """

    def __init__(self, state_store: Optional[StateStore] = None):
        self.state_store = state_store
        self.client_keys: Dict[str, str] = {}
        self.key_to_client: Dict[str, str] = {}
        self.public_keys: Dict[str, str] = {}
        self.registered_clients: Set[str] = set()
        self._restore()

    def _restore(self) -> None:
        if not self.state_store:
            return
        for client_id, api_key in self.state_store.get_client_keys().items():
            self.client_keys[client_id] = api_key
            self.key_to_client[api_key] = client_id
            self.registered_clients.add(client_id)
        for client_id, public_key in self.state_store.get_public_keys().items():
            self.public_keys[client_id] = public_key

    def generate_api_key(self) -> str:
        return secrets.token_hex(16)

    def register_client(
        self,
        client_id: str,
        api_key: Optional[str] = None,
        presented_key: Optional[str] = None,
    ) -> str:
        """
        Register a client and assign an API key.

        Idempotent only with proof of possession: if already registered,
        ``presented_key`` must match the stored key. Never returns an
        existing key to an unauthenticated caller.
        """
        if client_id in self.registered_clients:
            existing = self.client_keys[client_id]
            if presented_key and secrets.compare_digest(presented_key, existing):
                return existing
            raise ClientAlreadyRegisteredError(client_id)

        if api_key is None:
            api_key = self.generate_api_key()

        if api_key in self.key_to_client and self.key_to_client[api_key] != client_id:
            api_key = self.generate_api_key()

        self.client_keys[client_id] = api_key
        self.key_to_client[api_key] = client_id
        self.registered_clients.add(client_id)

        if self.state_store:
            self.state_store.set_client_key(client_id, api_key)

        return api_key

    def set_public_key(self, client_id: str, public_key: str) -> None:
        """Attach an Ed25519 public key to a registered node (Protocol V2 identity)."""
        from protocol.identity import encode_raw_key, normalize_public_key

        if client_id not in self.registered_clients:
            raise ValueError(f"Client '{client_id}' is not registered")
        raw = normalize_public_key(public_key)
        encoded = encode_raw_key(raw)
        self.public_keys[client_id] = encoded
        if self.state_store:
            self.state_store.set_public_key(client_id, encoded)

    def get_public_key(self, client_id: str) -> Optional[str]:
        return self.public_keys.get(client_id)

    def verify_node_signature(
        self,
        client_id: str,
        message: bytes,
        signature: str,
    ) -> bool:
        public_key = self.public_keys.get(client_id)
        if not public_key:
            return False
        from protocol.identity import verify

        return verify(public_key, message, signature)

    def validate_api_key(self, api_key: Optional[str], client_id: Optional[str] = None) -> bool:
        if api_key is None or api_key == "":
            return False
        if api_key not in self.key_to_client:
            return False
        if client_id is not None and self.key_to_client.get(api_key) != client_id:
            return False
        return True

    def get_client_id_from_key(self, api_key: str) -> Optional[str]:
        return self.key_to_client.get(api_key)

    def revoke_client(self, client_id: str) -> bool:
        if client_id not in self.registered_clients:
            return False
        api_key = self.client_keys.get(client_id)
        if api_key:
            del self.key_to_client[api_key]
            del self.client_keys[client_id]
        self.public_keys.pop(client_id, None)
        self.registered_clients.discard(client_id)
        if self.state_store:
            self.state_store.remove_client(client_id)
        return True

    def is_registered(self, client_id: str) -> bool:
        return client_id in self.registered_clients


def get_operator_api_key() -> Optional[str]:
    """Operator/admin key for privileged coordinator actions."""
    key = os.getenv("OPERATOR_API_KEY", "").strip()
    return key or None


def validate_operator_key(provided: Optional[str]) -> bool:
    """
    Validate operator key.

    If OPERATOR_API_KEY is unset, privileged endpoints stay open (dev mode).
    When set, the provided key must match.
    """
    expected = get_operator_api_key()
    if expected is None:
        return True
    return provided is not None and secrets.compare_digest(provided, expected)
