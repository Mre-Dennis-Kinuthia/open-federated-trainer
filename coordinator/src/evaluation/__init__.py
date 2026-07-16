"""
Evaluation Module for LoRA Adapters

Evaluates aggregated adapters on held-out data.
"""

from .evaluate_adapter import evaluate_adapter, AdapterEvaluationResult

__all__ = [
    "evaluate_adapter",
    "AdapterEvaluationResult",
]

