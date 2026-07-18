"""Milestone 2: Protocol V2 negotiation, header auth, identity, binary artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COORD_SRC = ROOT / "coordinator" / "src"


@pytest.fixture()
def coord_path(monkeypatch):
    sys.path.insert(0, str(COORD_SRC))
    yield


def test_negotiate_protocol_version(coord_path):
    from protocol.version import (
        ProtocolIncompatibleError,
        negotiate_protocol_version,
    )

    assert negotiate_protocol_version(None) == "1.0"
    assert negotiate_protocol_version("2.0") == "2.0"
    assert negotiate_protocol_version("1.9") == "1.9"
    with pytest.raises(ProtocolIncompatibleError):
        negotiate_protocol_version("9.0")


def test_extract_api_key_prefers_header(coord_path):
    from protocol.credentials import extract_api_key

    assert (
        extract_api_key(
            x_api_key="hdr",
            authorization="Bearer tok",
            query_api_key="q",
            body_api_key="b",
        )
        == "hdr"
    )
    assert extract_api_key(authorization="Bearer tok", query_api_key="q") == "tok"
    assert extract_api_key(query_api_key="q", body_api_key="b") == "b"
    assert extract_api_key(query_api_key="q") == "q"


def test_ed25519_identity_roundtrip(coord_path):
    from protocol.identity import canonical_auth_message, generate_keypair, sign, verify

    pub, priv = generate_keypair()
    msg = canonical_auth_message(
        client_id="n1",
        method="POST",
        path="/v2/artifacts",
        body_sha256="abc",
        timestamp="1",
    )
    sig = sign(priv, msg)
    assert verify(pub, msg, sig)
    assert not verify(pub, msg + b"x", sig)


def test_auth_manager_stores_public_key(tmp_path, monkeypatch, coord_path):
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    from core.auth import AuthManager
    from core.state_store import StateStore
    from protocol.identity import generate_keypair

    store = StateStore(path=str(tmp_path / "state.json"))
    auth = AuthManager(state_store=store)
    auth.register_client("node-a")
    pub, _ = generate_keypair()
    auth.set_public_key("node-a", pub)
    assert auth.get_public_key("node-a")

    auth2 = AuthManager(state_store=StateStore(path=str(tmp_path / "state.json")))
    assert auth2.get_public_key("node-a") == auth.get_public_key("node-a")


def test_artifact_upload_and_idempotent_update(tmp_path, monkeypatch, coord_path):
    monkeypatch.setenv("STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("IDEMPOTENCY_PATH", str(tmp_path / "idem.json"))
    monkeypatch.setenv("CLASSIC_ROUNDS_PATH", str(tmp_path / "rounds.json"))

    from core.auth import AuthManager
    from core.round_manager import RoundManager
    from core.state_store import StateStore
    from core.update_validator import UpdateValidator
    from persistence.json_repos import JsonRoundRepository
    from protocol.idempotency import IdempotencyStore
    from protocol import routes as routes_mod
    from protocol.routes import bind_dependencies
    from artifacts import LocalFilesystemArtifactStore

    store = StateStore(path=str(tmp_path / "state.json"))
    auth = AuthManager(state_store=store)
    key = auth.register_client("worker1")
    rm = RoundManager(
        state_store=store,
        round_repo=JsonRoundRepository(rounds_path=str(tmp_path / "rounds.json")),
    )
    rm.register_client("worker1")
    rid = rm.assign_client_to_round("worker1", "v1")
    assert rid is not None

    validator = UpdateValidator(rm, auth_manager=auth)
    routes_mod._idempotency = IdempotencyStore(path=str(tmp_path / "idem.json"))
    bind_dependencies(
        auth_manager=auth,
        round_manager=rm,
        update_validator=validator,
        metrics_collector=None,
        geo_presence=type("G", (), {"record": lambda *a, **k: None})(),
        client_ip_fn=lambda r: "127.0.0.1",
        register_legacy_fn=None,
    )

    art_store = LocalFilesystemArtifactStore(root=str(tmp_path / "artifacts"))
    payload = json.dumps({"weight_delta": [[0.1, 0.2]], "num_samples": 2}).encode()
    content_hash = art_store.put_bytes(payload)
    assert art_store.exists(content_hash)

    from protocol.schemas import UpdateByArtifactRequest
    from protocol.routes import submit_update_by_artifact
    import asyncio

    req = UpdateByArtifactRequest(
        client_id="worker1",
        round_id=rid,
        content_hash=content_hash,
        idempotency_key="upd-1",
        api_key=key,
    )

    async def _run():
        first = await submit_update_by_artifact(
            req,
            x_api_key=key,
            authorization=None,
            x_protocol_version="2.0",
        )
        second = await submit_update_by_artifact(
            req,
            x_api_key=key,
            authorization=None,
            x_protocol_version="2.0",
        )
        return first, second

    first, second = asyncio.run(_run())
    assert first.success and not first.replayed
    assert second.success and second.replayed
    assert "worker1" in rm.rounds[rid].updates_received


def test_protocol_info_schema(coord_path):
    from protocol.schemas import ProtocolInfoResponse
    from protocol.version import PROTOCOL_VERSION, SUPPORTED_MAJORS, protocol_v2_enabled

    info = ProtocolInfoResponse(
        protocol_version=PROTOCOL_VERSION,
        supported_majors=sorted(SUPPORTED_MAJORS),
        v2_enabled=protocol_v2_enabled(),
    )
    assert info.protocol_version.startswith("2.")
    assert 1 in info.supported_majors and 2 in info.supported_majors
