"""
Aggregation Module

LoRA adapter FedAvg (ΔW+SVD), classic FL strategies, manifests, isolated merge.
"""

from .fedavg_adapters import aggregate_lora_adapters, validate_adapter
from .strategies import (
    AdaptiveFedAvgStrategy,
    AggregationStrategy,
    ClientContribution,
    FedAvgStrategy,
    RobustTrimmedMeanStrategy,
    StrategyResult,
    get_strategy,
    list_strategies,
)
from .adapter_manifest import AdapterManifest, build_adapter_manifest, register_adapter_manifest
from .merge import isolated_merge_state_dicts, merge_delta_into_weight

__all__ = [
    "aggregate_lora_adapters",
    "validate_adapter",
    "AggregationStrategy",
    "ClientContribution",
    "StrategyResult",
    "FedAvgStrategy",
    "AdaptiveFedAvgStrategy",
    "RobustTrimmedMeanStrategy",
    "get_strategy",
    "list_strategies",
    "AdapterManifest",
    "build_adapter_manifest",
    "register_adapter_manifest",
    "isolated_merge_state_dicts",
    "merge_delta_into_weight",
]
