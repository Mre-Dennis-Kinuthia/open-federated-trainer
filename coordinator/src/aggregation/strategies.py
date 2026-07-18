"""Classic FL aggregation strategies (Milestone 3).

FedAvg remains the default. Adaptive and robust strategies are selectable via
``AGGREGATION_STRATEGY=fedavg|adaptive|robust``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class ClientContribution:
    """One client's parsed update for strategy input."""

    client_id: str
    weight_delta: List[List[float]]
    num_samples: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyResult:
    averaged_delta: List[List[float]]
    strategy_name: str
    details: Dict = field(default_factory=dict)


@runtime_checkable
class AggregationStrategy(Protocol):
    name: str

    def aggregate(self, contributions: List[ClientContribution]) -> StrategyResult:
        """Combine client deltas into one averaged delta."""


def _require_consistent_shapes(deltas: List[List[List[float]]]) -> None:
    if not deltas:
        raise ValueError("No deltas to aggregate")
    num_layers = len(deltas[0])
    for d in deltas:
        if len(d) != num_layers:
            raise ValueError("Inconsistent number of parameter layers across clients")
    for layer_idx in range(num_layers):
        layer_len = len(deltas[0][layer_idx])
        for d in deltas:
            if len(d[layer_idx]) != layer_len:
                raise ValueError(f"Inconsistent layer {layer_idx} sizes across clients")


class FedAvgStrategy:
    """Uniform mean over clients (classic FedAvg on deltas)."""

    name = "fedavg"

    def aggregate(self, contributions: List[ClientContribution]) -> StrategyResult:
        deltas = [c.weight_delta for c in contributions]
        _require_consistent_shapes(deltas)
        num_clients = len(deltas)
        num_layers = len(deltas[0])
        averaged: List[List[float]] = []
        for layer_idx in range(num_layers):
            layer_len = len(deltas[0][layer_idx])
            avg_layer = [0.0] * layer_len
            for client_delta in deltas:
                for i, v in enumerate(client_delta[layer_idx]):
                    avg_layer[i] += float(v)
            averaged.append([v / num_clients for v in avg_layer])
        return StrategyResult(
            averaged_delta=averaged,
            strategy_name=self.name,
            details={"num_clients": num_clients, "weighting": "uniform"},
        )


class AdaptiveFedAvgStrategy:
    """Sample-weighted FedAvg (adaptive to local dataset size)."""

    name = "adaptive"

    def aggregate(self, contributions: List[ClientContribution]) -> StrategyResult:
        deltas = [c.weight_delta for c in contributions]
        _require_consistent_shapes(deltas)
        weights = [max(float(c.num_samples), 1e-9) for c in contributions]
        total = sum(weights)
        num_layers = len(deltas[0])
        averaged: List[List[float]] = []
        for layer_idx in range(num_layers):
            layer_len = len(deltas[0][layer_idx])
            avg_layer = [0.0] * layer_len
            for client_delta, w in zip(deltas, weights):
                scale = w / total
                for i, v in enumerate(client_delta[layer_idx]):
                    avg_layer[i] += float(v) * scale
            averaged.append(avg_layer)
        return StrategyResult(
            averaged_delta=averaged,
            strategy_name=self.name,
            details={
                "num_clients": len(contributions),
                "weighting": "num_samples",
                "sample_weights": {
                    c.client_id: max(float(c.num_samples), 1e-9) for c in contributions
                },
            },
        )


class RobustTrimmedMeanStrategy:
    """
    Coordinate-wise trimmed mean (robust aggregation).

    Trims ``trim_ratio`` fraction from each tail per coordinate. Falls back to
    coordinate-wise median when fewer than 3 clients.
    """

    name = "robust"

    def __init__(self, trim_ratio: float = 0.1):
        self.trim_ratio = max(0.0, min(0.4, float(trim_ratio)))

    def aggregate(self, contributions: List[ClientContribution]) -> StrategyResult:
        deltas = [c.weight_delta for c in contributions]
        _require_consistent_shapes(deltas)
        n = len(deltas)
        num_layers = len(deltas[0])
        averaged: List[List[float]] = []
        method = "median" if n < 3 else "trimmed_mean"
        for layer_idx in range(num_layers):
            layer_len = len(deltas[0][layer_idx])
            avg_layer: List[float] = []
            for i in range(layer_len):
                values = sorted(float(d[layer_idx][i]) for d in deltas)
                if method == "median":
                    mid = n // 2
                    if n % 2:
                        avg_layer.append(values[mid])
                    else:
                        avg_layer.append(0.5 * (values[mid - 1] + values[mid]))
                else:
                    k = int(n * self.trim_ratio)
                    trimmed = values[k : n - k] if n - 2 * k > 0 else values
                    avg_layer.append(sum(trimmed) / len(trimmed))
            averaged.append(avg_layer)
        return StrategyResult(
            averaged_delta=averaged,
            strategy_name=self.name,
            details={
                "num_clients": n,
                "method": method,
                "trim_ratio": self.trim_ratio,
            },
        )


_STRATEGIES = {
    "fedavg": FedAvgStrategy,
    "adaptive": AdaptiveFedAvgStrategy,
    "robust": RobustTrimmedMeanStrategy,
}


def get_strategy(name: Optional[str] = None) -> AggregationStrategy:
    """Resolve strategy from argument or ``AGGREGATION_STRATEGY`` env (default fedavg)."""
    key = (name or os.getenv("AGGREGATION_STRATEGY", "fedavg") or "fedavg").strip().lower()
    if key not in _STRATEGIES:
        raise ValueError(
            f"Unknown AGGREGATION_STRATEGY={key!r}; choose from {sorted(_STRATEGIES)}"
        )
    if key == "robust":
        trim = float(os.getenv("ROBUST_TRIM_RATIO", "0.1"))
        return RobustTrimmedMeanStrategy(trim_ratio=trim)
    return _STRATEGIES[key]()


def list_strategies() -> List[str]:
    return sorted(_STRATEGIES)
