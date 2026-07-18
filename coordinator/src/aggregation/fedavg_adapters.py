"""
Federated Averaging for LoRA Adapters

Default strategy ``delta_svd`` averages reconstructed ΔW = B @ A then
re-factors to rank-r via SVD. Legacy ``param_fedavg`` averages A and B
elementwise (incorrect for the product, kept for compatibility).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from utils.logger import get_logger

from .lora_math import (
    infer_rank_from_pair,
    pair_lora_ab_keys,
    reconstruct_delta,
    svd_factorize_delta,
    to_tensor,
)

logger = get_logger("aggregation")


def validate_adapter(adapter_state_dict: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate an adapter state dict.

    Checks for:
    - Valid structure
    - No NaN or Inf values
    - Reasonable value ranges
    """
    if not isinstance(adapter_state_dict, dict):
        return False, "Adapter must be a dictionary"

    if len(adapter_state_dict) == 0:
        return False, "Adapter state dict is empty"

    for key, value in adapter_state_dict.items():
        if not isinstance(key, str):
            return False, f"Invalid key type: {type(key)}"

        if isinstance(value, (list, np.ndarray)):
            tensor = torch.tensor(value) if isinstance(value, list) else torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            return False, f"Invalid value type for key {key}: {type(value)}"

        if torch.isnan(tensor).any():
            return False, f"NaN values found in {key}"

        if torch.isinf(tensor).any():
            return False, f"Inf values found in {key}"

        if tensor.numel() > 0:
            max_val = torch.abs(tensor).max().item()
            if max_val > 100.0:
                logger.warning(f"Large values in {key}: max={max_val}")

    return True, None


def _resolve_strategy(name: Optional[str] = None) -> str:
    key = (
        name
        or os.getenv("LORA_AGG_STRATEGY", "delta_svd")
        or "delta_svd"
    ).strip().lower()
    if key not in {"delta_svd", "param_fedavg"}:
        raise ValueError(
            f"Unknown LORA_AGG_STRATEGY={key!r}; use delta_svd or param_fedavg"
        )
    return key


def _filter_compatible(
    adapter_submissions: Dict[str, Dict],
    *,
    weight_by_samples: bool,
) -> Dict[str, Dict]:
    valid_submissions: Dict[str, Dict] = {}
    for client_id, submission in adapter_submissions.items():
        adapter_state_dict = submission.get("adapter_state_dict")
        if adapter_state_dict is None:
            logger.warning(f"Client {client_id} submission missing adapter_state_dict")
            continue
        is_valid, error_msg = validate_adapter(adapter_state_dict)
        if not is_valid:
            logger.warning(f"Client {client_id} adapter validation failed: {error_msg}")
            continue
        valid_submissions[client_id] = submission

    if not valid_submissions:
        return {}

    first_client = next(iter(valid_submissions))
    first_adapter = valid_submissions[first_client]["adapter_state_dict"]
    param_keys = list(first_adapter.keys())
    expected_shapes = {
        key: tuple(torch.as_tensor(value).shape)
        for key, value in first_adapter.items()
    }
    compatible: Dict[str, Dict] = {}
    for client_id, submission in valid_submissions.items():
        adapter = submission["adapter_state_dict"]
        actual_shapes = {
            key: tuple(torch.as_tensor(value).shape)
            for key, value in adapter.items()
        }
        if set(adapter) != set(param_keys) or actual_shapes != expected_shapes:
            logger.warning(f"Client {client_id} adapter has mismatched keys or shapes")
            continue
        if weight_by_samples and int(submission.get("num_samples", 0)) <= 0:
            logger.warning(f"Client {client_id} has invalid num_samples")
            continue
        compatible[client_id] = submission
    return compatible


def _client_weights(
    submissions: Dict[str, Dict],
    *,
    weight_by_samples: bool,
) -> List[float]:
    if weight_by_samples:
        raw = [float(s.get("num_samples", 1)) for s in submissions.values()]
        total = sum(raw)
        if total <= 0:
            n = len(raw)
            return [1.0 / n] * n
        return [w / total for w in raw]
    n = len(submissions)
    return [1.0 / n] * n


def _aggregate_param_fedavg(
    valid_submissions: Dict[str, Dict],
    *,
    weight_by_samples: bool,
) -> Dict:
    param_keys = list(next(iter(valid_submissions.values()))["adapter_state_dict"].keys())
    weights = _client_weights(valid_submissions, weight_by_samples=weight_by_samples)
    aggregated_adapter: Dict = {}
    for param_key in param_keys:
        param_values = []
        for submission in valid_submissions.values():
            param_values.append(to_tensor(submission["adapter_state_dict"][param_key]))
        aggregated_param = torch.zeros_like(param_values[0])
        for tensor, weight in zip(param_values, weights):
            aggregated_param += weight * tensor
        aggregated_adapter[param_key] = aggregated_param.cpu().numpy().tolist()
    return aggregated_adapter


def _aggregate_delta_svd(
    valid_submissions: Dict[str, Dict],
    *,
    weight_by_samples: bool,
) -> Dict:
    first = next(iter(valid_submissions.values()))["adapter_state_dict"]
    pairs, unpaired = pair_lora_ab_keys(list(first.keys()))
    weights = _client_weights(valid_submissions, weight_by_samples=weight_by_samples)
    clients = list(valid_submissions.values())
    aggregated: Dict = {}

    for stem, (a_key, b_key) in pairs.items():
        deltas = []
        rank = None
        for submission in clients:
            adapter = submission["adapter_state_dict"]
            a = to_tensor(adapter[a_key])
            b = to_tensor(adapter[b_key])
            if rank is None:
                rank = infer_rank_from_pair(a, b)
            deltas.append(reconstruct_delta(a, b))
        avg_delta = torch.zeros_like(deltas[0])
        for delta, weight in zip(deltas, weights):
            avg_delta += weight * delta
        a_new, b_new = svd_factorize_delta(avg_delta, rank or 1)
        aggregated[a_key] = a_new.cpu().numpy().tolist()
        aggregated[b_key] = b_new.cpu().numpy().tolist()

    # Unpaired tensors (biases, modules_to_save): fall back to param FedAvg
    if unpaired:
        for param_key in unpaired:
            param_values = [
                to_tensor(s["adapter_state_dict"][param_key]) for s in clients
            ]
            aggregated_param = torch.zeros_like(param_values[0])
            for tensor, weight in zip(param_values, weights):
                aggregated_param += weight * tensor
            aggregated[param_key] = aggregated_param.cpu().numpy().tolist()

    return aggregated


def aggregate_lora_adapters(
    adapter_submissions: Dict[str, Dict],
    weight_by_samples: bool = True,
    strategy: Optional[str] = None,
) -> Optional[Dict]:
    """
    Aggregate LoRA adapters.

    Strategies:
    - ``delta_svd`` (default): FedAvg on ΔW = B@A, then SVD back to A/B
    - ``param_fedavg``: elementwise FedAvg on each parameter (legacy)
    """
    if not adapter_submissions:
        logger.warning("No adapter submissions to aggregate")
        return None

    try:
        strategy_name = _resolve_strategy(strategy)
    except ValueError as exc:
        logger.error(str(exc))
        return None

    valid_submissions = _filter_compatible(
        adapter_submissions, weight_by_samples=weight_by_samples
    )
    if not valid_submissions:
        logger.error("No valid adapter submissions after validation")
        return None

    logger.info(
        f"Aggregating {len(valid_submissions)} adapters with strategy={strategy_name}",
        extra={
            "component": "aggregation",
            "event": "aggregation_started",
            "num_adapters": len(valid_submissions),
            "strategy": strategy_name,
        },
    )

    if strategy_name == "param_fedavg":
        aggregated_adapter = _aggregate_param_fedavg(
            valid_submissions, weight_by_samples=weight_by_samples
        )
    else:
        aggregated_adapter = _aggregate_delta_svd(
            valid_submissions, weight_by_samples=weight_by_samples
        )

    logger.info(
        f"Aggregation completed: {len(aggregated_adapter)} parameters",
        extra={
            "component": "aggregation",
            "event": "aggregation_completed",
            "num_parameters": len(aggregated_adapter),
            "num_clients": len(valid_submissions),
            "strategy": strategy_name,
        },
    )
    return aggregated_adapter
