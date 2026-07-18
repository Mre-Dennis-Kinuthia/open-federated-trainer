"""Ed25519 node identity helpers (optional alongside API keys)."""

from __future__ import annotations

import base64
import hashlib
from typing import Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


def encode_raw_key(data: bytes) -> str:
    return _b64u(data)


def generate_keypair() -> Tuple[str, str]:
    """Return (public_key_b64url, private_key_b64url) raw 32-byte keys."""
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes_raw()
    public_bytes = private.public_key().public_bytes_raw()
    return encode_raw_key(public_bytes), encode_raw_key(private_bytes)


def normalize_public_key(public_key: str) -> bytes:
    raw = _decode_key(public_key)
    if len(raw) != 32:
        raise ValueError("Ed25519 public key must be 32 bytes")
    # Validate by constructing
    Ed25519PublicKey.from_public_bytes(raw)
    return raw


def public_key_fingerprint(public_key: str) -> str:
    return hashlib.sha256(normalize_public_key(public_key)).hexdigest()[:16]


def sign(private_key_b64: str, message: bytes) -> str:
    private = Ed25519PrivateKey.from_private_bytes(_decode_key(private_key_b64))
    return _b64u(private.sign(message))


def verify(public_key: str, message: bytes, signature_b64: str) -> bool:
    try:
        public = Ed25519PublicKey.from_public_bytes(normalize_public_key(public_key))
        public.verify(_decode_key(signature_b64), message)
        return True
    except (ValueError, InvalidSignature):
        return False


def canonical_auth_message(
    *,
    client_id: str,
    method: str,
    path: str,
    body_sha256: str,
    timestamp: str,
) -> bytes:
    return f"{client_id}|{method.upper()}|{path}|{body_sha256}|{timestamp}".encode("utf-8")


def public_key_spki_pem(public_key: str) -> str:
    public = Ed25519PublicKey.from_public_bytes(normalize_public_key(public_key))
    return public.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo).decode("ascii")


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _decode_key(value: str) -> bytes:
    text = (value or "").strip()
    if not text:
        raise ValueError("empty key")
    # Hex
    if all(c in "0123456789abcdefABCDEF" for c in text) and len(text) % 2 == 0:
        try:
            return bytes.fromhex(text)
        except ValueError:
            pass
    # Base64 / base64url
    pad = "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(text + pad)
    except Exception:
        return base64.b64decode(text + pad)
