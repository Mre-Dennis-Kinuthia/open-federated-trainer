"""Job result verification: canaries and N-of-M quorum (Milestone 6)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple


def sanitize_entrypoint_string(entrypoint: str) -> str:
    """Coordinator-side entrypoint sanitize (mirrors worker rules; no import)."""
    import re

    text = (entrypoint or "").strip()
    if not text:
        raise ValueError("empty entrypoint")
    if any(bad in text for bad in ("..", "/", "\\", "\x00", "://", " ")):
        raise ValueError(
            "entrypoint must be a dotted module:function (no paths or URLs)"
        )
    if text.count(":") != 1:
        raise ValueError("entrypoint must be exactly module.path:function")
    module_name, function_name = text.split(":", 1)
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$", module_name):
        raise ValueError(f"invalid module identifier: {module_name!r}")
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", function_name):
        raise ValueError(f"invalid function identifier: {function_name!r}")
    return text


def result_fingerprint(value: Any) -> str:
    """Stable hash of a JSON-serializable result payload."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verification_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize verification settings from job payload."""
    raw = payload.get("verification")
    if isinstance(raw, dict):
        mode = str(raw.get("mode") or "").strip().lower()
        cfg = dict(raw)
        cfg["mode"] = mode or ("canary" if payload.get("canary") else "none")
        return cfg
    if payload.get("canary"):
        return {"mode": "canary"}
    return {"mode": "none"}


def evaluate_canary(
    result: Dict[str, Any],
    *,
    expected_fingerprint: Optional[str] = None,
    expected_result: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Compare worker result against a known-answer canary.

    Prefers ``expected_fingerprint``; else fingerprints ``expected_result``.
    Fingerprint covers ``result["result"]`` when present, else the whole body.
    """
    body = result.get("result", result)
    actual = result_fingerprint(body)
    if expected_fingerprint:
        expected = str(expected_fingerprint).lower()
    elif expected_result is not None:
        expected = result_fingerprint(expected_result)
    else:
        return {
            "mode": "canary",
            "status": "failed",
            "reason": "canary job missing expected_fingerprint / expected_result",
            "actual_fingerprint": actual,
        }
    passed = actual == expected
    return {
        "mode": "canary",
        "status": "passed" if passed else "failed",
        "actual_fingerprint": actual,
        "expected_fingerprint": expected,
        "reason": None if passed else "canary fingerprint mismatch",
    }


def n_of_m_status(
    candidates: List[Dict[str, Any]],
    *,
    n: int,
    m: int,
) -> Dict[str, Any]:
    """
    Quorum over candidate submissions.

    Each candidate: ``{"client_id", "fingerprint", "result"}``.
    Returns status agreed|pending|failed.
    """
    n = max(1, int(n))
    m = max(n, int(m))
    counts: Dict[str, List[Dict[str, Any]]] = {}
    for item in candidates:
        fp = item.get("fingerprint") or result_fingerprint(item.get("result"))
        counts.setdefault(fp, []).append(item)

    for fp, group in counts.items():
        if len(group) >= n:
            return {
                "mode": "n_of_m",
                "status": "agreed",
                "n": n,
                "m": m,
                "agreed_fingerprint": fp,
                "submissions": len(candidates),
                "agreeing_clients": [g.get("client_id") for g in group],
                "result": group[0].get("result"),
            }

    if len(candidates) >= m:
        return {
            "mode": "n_of_m",
            "status": "failed",
            "n": n,
            "m": m,
            "submissions": len(candidates),
            "reason": f"no quorum of {n} after {m} submissions",
            "fingerprints": {fp: len(g) for fp, g in counts.items()},
        }

    return {
        "mode": "n_of_m",
        "status": "pending",
        "n": n,
        "m": m,
        "submissions": len(candidates),
        "reason": "awaiting additional independent results",
    }


def apply_verification_on_success(
    payload: Dict[str, Any],
    result: Dict[str, Any],
    *,
    client_id: str,
    candidates: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Returns (outcome, validation, updated_candidates, result_to_store).

    outcome: completed | requeue | failed
    """
    cfg = verification_config(payload)
    mode = cfg.get("mode") or "none"

    if mode == "canary":
        validation = evaluate_canary(
            result,
            expected_fingerprint=cfg.get("expected_fingerprint")
            or payload.get("expected_fingerprint"),
            expected_result=cfg.get("expected_result")
            or payload.get("expected_result"),
        )
        if validation["status"] == "passed":
            return "completed", validation, candidates, result
        return "failed", validation, candidates, result

    if mode == "n_of_m":
        body = result.get("result", result)
        fp = result_fingerprint(body)
        updated = list(candidates)
        updated.append(
            {
                "client_id": client_id,
                "fingerprint": fp,
                "result": result,
            }
        )
        n = int(cfg.get("n") or 2)
        m = int(cfg.get("m") or 3)
        validation = n_of_m_status(updated, n=n, m=m)
        if validation["status"] == "agreed":
            return "completed", validation, updated, validation.get("result") or result
        if validation["status"] == "failed":
            return "failed", validation, updated, result
        return "requeue", validation, updated, result

    return "completed", {"mode": "none", "status": "skipped"}, candidates, result
