"""
Client-side handlers for general (non-training) jobs.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, Optional

from datasets import load_local_dataset


def handle_inference(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """
    Run local inference on private inputs.

    Payload:
      inputs: list[str|dict]
      model_id: optional tag (no weights downloaded — demo scoring)
    """
    payload = job.get("payload") or {}
    inputs = payload.get("inputs") or []
    model_id = payload.get("model_id", "local-scorer")

    predictions = []
    for i, item in enumerate(inputs):
        text = item if isinstance(item, str) else json.dumps(item, sort_keys=True)
        # Deterministic local "score" without shipping data
        digest = hashlib.sha256(f"{model_id}:{text}".encode()).hexdigest()
        score = int(digest[:8], 16) / 0xFFFFFFFF
        predictions.append(
            {
                "index": i,
                "label": "pos" if score >= 0.5 else "neg",
                "score": round(score, 6),
            }
        )

    return {
        "job_type": "inference",
        "client_id": client_id,
        "model_id": model_id,
        "num_inputs": len(inputs),
        "predictions": predictions,
        "data_exported": False,
    }


def handle_label(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """
    Auto-label a chunk from the local private dataset (or payload items).

    In production this could open a UI; here we apply a deterministic heuristic
    so volunteer nodes can process labeling jobs unattended.
    """
    payload = job.get("payload") or {}
    items = payload.get("items")
    if not items:
        ds = load_local_dataset()
        start = int(payload.get("offset", 0))
        limit = int(payload.get("limit", 16))
        items = ds.texts[start : start + limit] or ds.rows[start : start + limit]

    labeled = []
    for i, item in enumerate(items):
        text = item if isinstance(item, str) else str(item.get("text", item))
        # Heuristic: length + keyword
        label = "long" if len(text) > 40 else "short"
        if "federated" in text.lower() or "edge" in text.lower():
            label = "domain"
        labeled.append({"index": i, "text_preview": text[:80], "label": label})

    return {
        "job_type": "label",
        "client_id": client_id,
        "num_labeled": len(labeled),
        "labels": labeled,
        "data_exported": False,
    }


def handle_compute(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """
    Folding@home-style science chunk.

    Payload:
      work_unit: { seed, steps, formula }
      formula: "mandelbrot" | "monte_carlo_pi" | "hash_work"
    """
    payload = job.get("payload") or {}
    wu = payload.get("work_unit") or payload
    formula = (wu.get("formula") or payload.get("formula") or "monte_carlo_pi").lower()
    seed = int(wu.get("seed", 42))
    steps = int(wu.get("steps", 50_000))
    started = time.time()

    if formula == "monte_carlo_pi":
        # Deterministic PRNG
        x = seed % (2**31 - 1) or 1
        inside = 0
        for _ in range(steps):
            x = (1103515245 * x + 12345) % (2**31)
            a = (x % 10000) / 10000.0
            x = (1103515245 * x + 12345) % (2**31)
            b = (x % 10000) / 10000.0
            if a * a + b * b <= 1.0:
                inside += 1
        result_value = 4.0 * inside / steps
        detail = {"pi_estimate": result_value, "inside": inside}

    elif formula == "mandelbrot":
        # Escape-time checksum over a small grid (CPU work)
        max_iter = min(steps, 200)
        total = 0
        grid = 64
        for yi in range(grid):
            for xi in range(grid):
                zr = cr = -2.0 + 3.0 * xi / grid
                zi = ci = -1.5 + 3.0 * yi / grid
                n = 0
                while zr * zr + zi * zi <= 4.0 and n < max_iter:
                    zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
                    n += 1
                total += n
        detail = {"escape_sum": total, "grid": grid, "max_iter": max_iter}
        result_value = total

    else:  # hash_work
        digest = hashlib.sha256(f"{seed}".encode()).hexdigest()
        for i in range(min(steps, 10_000)):
            digest = hashlib.sha256(f"{digest}:{i}".encode()).hexdigest()
        detail = {"digest": digest}
        result_value = digest

    elapsed = time.time() - started
    return {
        "job_type": "compute",
        "client_id": client_id,
        "formula": formula,
        "seed": seed,
        "steps": steps,
        "result": result_value,
        "detail": detail,
        "elapsed_seconds": round(elapsed, 4),
        "data_exported": False,
    }


HANDLERS = {
    "inference": handle_inference,
    "label": handle_label,
    "compute": handle_compute,
}


def run_job(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    job_type = job.get("job_type")
    handler = HANDLERS.get(job_type)
    if not handler:
        raise ValueError(f"Unsupported job_type: {job_type}")
    return handler(job, client_id)
