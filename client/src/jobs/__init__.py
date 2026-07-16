"""Production job handlers for inference, auto-labeling, and compute plugins."""

from __future__ import annotations

import importlib
import json
import os
import time
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Sequence

from private_datasets import load_local_dataset


class JobConfigurationError(ValueError):
    """A job does not name the real backend or inputs it requires."""


def _device() -> int:
    configured = os.getenv("INFERENCE_DEVICE", "").strip()
    if configured:
        return int(configured)
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


@lru_cache(maxsize=4)
def _pipeline(task: str, model_id: str, revision: str):
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Real inference requires transformers: pip install transformers"
        ) from exc
    return pipeline(
        task=task,
        model=model_id,
        revision=revision or None,
        device=_device(),
        trust_remote_code=False,
    )


def _private_inputs(payload: Dict[str, Any]) -> List[Any]:
    supplied = payload.get("inputs")
    if supplied is not None:
        if not isinstance(supplied, list) or not supplied:
            raise JobConfigurationError("payload.inputs must be a non-empty list")
        return supplied

    dataset = load_local_dataset(
        path=payload.get("dataset_path"),
        fmt=payload.get("dataset_format"),
    )
    start = max(0, int(payload.get("offset", 0)))
    limit = max(1, min(int(payload.get("limit", 16)), 1000))
    values: Sequence[Any] = dataset.texts or dataset.rows
    selected = list(values[start : start + limit])
    if not selected:
        raise JobConfigurationError("The selected local dataset range is empty")
    return selected


def _text_values(values: Iterable[Any]) -> List[str]:
    return [
        value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        for value in values
    ]


def handle_inference(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Run an actual Transformers pipeline against supplied or local inputs."""
    payload = job.get("payload") or {}
    model_id = (
        str(payload.get("model_id") or os.getenv("INFERENCE_MODEL_ID", "")).strip()
    )
    if not model_id:
        raise JobConfigurationError(
            "Inference requires payload.model_id or INFERENCE_MODEL_ID"
        )
    task = str(payload.get("task") or os.getenv("INFERENCE_TASK", "text-classification"))
    revision = str(payload.get("revision") or os.getenv("INFERENCE_MODEL_REVISION", ""))
    inputs = _text_values(_private_inputs(payload))
    runner = _pipeline(task, model_id, revision)

    options: Dict[str, Any] = {}
    if task == "text-generation":
        options["max_new_tokens"] = max(
            1, min(int(payload.get("max_new_tokens", 64)), 2048)
        )
        options["do_sample"] = bool(payload.get("do_sample", False))
    predictions = runner(inputs, batch_size=max(1, int(payload.get("batch_size", 8))), **options)
    return {
        "job_type": "inference",
        "client_id": client_id,
        "backend": "transformers",
        "task": task,
        "model_id": model_id,
        "model_revision": revision or None,
        "num_inputs": len(inputs),
        "predictions": predictions,
        "input_source": "payload" if payload.get("inputs") is not None else "local_dataset",
        "raw_inputs_returned": False,
        "inputs_were_coordinator_supplied": payload.get("inputs") is not None,
    }


def handle_label(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Auto-label private text using a real classification model."""
    payload = job.get("payload") or {}
    model_id = str(
        payload.get("model_id") or os.getenv("LABEL_MODEL_ID", "")
    ).strip()
    if not model_id:
        raise JobConfigurationError(
            "Labeling requires payload.model_id or LABEL_MODEL_ID"
        )

    candidate_labels = payload.get("candidate_labels")
    task = "zero-shot-classification" if candidate_labels else str(
        payload.get("task") or "text-classification"
    )
    if candidate_labels and (
        not isinstance(candidate_labels, list) or len(candidate_labels) < 2
    ):
        raise JobConfigurationError("candidate_labels must contain at least two labels")

    values = _text_values(_private_inputs(payload))
    revision = str(payload.get("revision") or os.getenv("LABEL_MODEL_REVISION", ""))
    runner = _pipeline(task, model_id, revision)
    if candidate_labels:
        outputs = runner(
            values,
            candidate_labels=candidate_labels,
            multi_label=bool(payload.get("multi_label", False)),
        )
    else:
        outputs = runner(values, batch_size=max(1, int(payload.get("batch_size", 8))))

    labels = []
    for index, output in enumerate(outputs):
        if isinstance(output, list):
            best = output[0]
            labels.append(
                {
                    "index": index,
                    "label": best.get("label"),
                    "score": best.get("score"),
                    "candidates": output if payload.get("return_all_scores") else None,
                }
            )
        else:
            labels.append(
                {
                    "index": index,
                    "label": output.get("labels", [None])[0],
                    "score": output.get("scores", [None])[0],
                    "candidates": output if payload.get("return_all_scores") else None,
                }
            )
    return {
        "job_type": "label",
        "client_id": client_id,
        "backend": "transformers",
        "model_id": model_id,
        "task": task,
        "num_labeled": len(labels),
        "labels": labels,
        "raw_inputs_returned": False,
        "inputs_were_coordinator_supplied": payload.get("inputs") is not None,
    }


def _allowed_compute_modules() -> List[str]:
    return [
        item.strip()
        for item in os.getenv("COMPUTE_PLUGIN_ALLOWLIST", "").split(",")
        if item.strip()
    ]


def _load_compute_plugin(entrypoint: str) -> Callable[[Dict[str, Any]], Any]:
    if ":" not in entrypoint:
        raise JobConfigurationError("compute entrypoint must be module.path:function")
    module_name, function_name = entrypoint.split(":", 1)
    allowed = _allowed_compute_modules()
    if not allowed or not any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in allowed
    ):
        raise JobConfigurationError(
            f"Compute module {module_name!r} is not allowlisted. Set "
            "COMPUTE_PLUGIN_ALLOWLIST on the worker."
        )
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise JobConfigurationError(f"Compute entrypoint is not callable: {entrypoint}")
    return function


def handle_compute(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    """Run an operator-installed, explicitly allowlisted compute work unit."""
    payload = job.get("payload") or {}
    entrypoint = str(payload.get("entrypoint") or "").strip()
    if not entrypoint:
        raise JobConfigurationError(
            "Compute jobs require payload.entrypoint=module:function; built-in toy "
            "formulas are not used in production mode."
        )
    work_unit = payload.get("work_unit")
    if not isinstance(work_unit, dict):
        raise JobConfigurationError("payload.work_unit must be a JSON object")

    started = time.monotonic()
    result = _load_compute_plugin(entrypoint)(work_unit)
    try:
        json.dumps(result)
    except (TypeError, ValueError) as exc:
        raise TypeError("Compute plugin result must be JSON serializable") from exc
    return {
        "job_type": "compute",
        "client_id": client_id,
        "backend": "python-plugin",
        "entrypoint": entrypoint,
        "result": result,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


HANDLERS: Dict[str, Callable[[Dict[str, Any], str], Dict[str, Any]]] = {
    "inference": handle_inference,
    "label": handle_label,
    "compute": handle_compute,
}


def run_job(job: Dict[str, Any], client_id: str) -> Dict[str, Any]:
    job_type = str(job.get("job_type") or "")
    handler = HANDLERS.get(job_type)
    if handler is None:
        raise ValueError(f"Unsupported job_type: {job_type}")
    return handler(job, client_id)
