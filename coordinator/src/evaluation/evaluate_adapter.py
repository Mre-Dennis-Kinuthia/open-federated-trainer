"""
Evaluate LoRA Adapter

Evaluates an aggregated LoRA adapter on a small held-out dataset.
Compares performance against previous adapter version.
"""

from typing import Dict, Optional
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger("evaluation")


@dataclass
class AdapterEvaluationResult:
    """Result of adapter evaluation."""
    round_id: int
    adapter_version: str
    evaluation_loss: float
    num_eval_samples: int
    passed: bool  # True if loss improved or stayed same
    previous_loss: Optional[float] = None
    improvement: Optional[float] = None  # Negative if improved


def evaluate_adapter(
    round_id: int,
    adapter_version: str,
    aggregated_adapter: Dict,
    previous_adapter_loss: Optional[float] = None
) -> AdapterEvaluationResult:
    """
    Evaluate an aggregated LoRA adapter.
    
    This is a minimal evaluation hook. In production, this would:
    1. Load base model + aggregated adapter
    2. Run on held-out evaluation dataset
    3. Compute loss/metrics
    4. Compare with previous adapter
    
    For MVP, we use a simplified check:
    - Validate adapter structure
    - Check for reasonable parameter norms
    - Compare with previous loss if available
    
    Args:
        round_id: Round identifier
        adapter_version: Version string for the adapter
        aggregated_adapter: Aggregated adapter state dict
        previous_adapter_loss: Loss from previous adapter version (optional)
        
    Returns:
        AdapterEvaluationResult with evaluation metrics
    """
    logger.info(f"Evaluating adapter {adapter_version} for round {round_id}", extra={
        "component": "evaluation",
        "event": "evaluation_started",
        "round_id": round_id,
        "adapter_version": adapter_version
    })
    
    # Minimal validation: check adapter structure
    if not aggregated_adapter or len(aggregated_adapter) == 0:
        logger.error("Empty adapter for evaluation")
        return AdapterEvaluationResult(
            round_id=round_id,
            adapter_version=adapter_version,
            evaluation_loss=float('inf'),
            num_eval_samples=0,
            passed=False
        )
    
    # Compute average parameter norm as a proxy for "quality"
    # In production, this would be actual evaluation loss
    import numpy as np
    
    total_norm = 0.0
    num_params = 0
    
    for key, value in aggregated_adapter.items():
        if isinstance(value, list):
            arr = np.array(value)
            norm = np.linalg.norm(arr)
            total_norm += norm
            num_params += 1
    
    avg_norm = total_norm / num_params if num_params > 0 else float('inf')
    
    # Use norm as proxy for loss (lower is better, but this is just a placeholder)
    # In real implementation, this would be actual evaluation loss
    evaluation_loss = avg_norm
    
    # Compare with previous adapter
    passed = True
    improvement = None
    
    if previous_adapter_loss is not None:
        improvement = evaluation_loss - previous_adapter_loss
        # Pass if loss improved (decreased) or stayed same (within tolerance)
        if improvement > 0.1:  # Loss increased significantly
            passed = False
            logger.warning(f"Adapter evaluation failed: loss increased by {improvement:.4f}", extra={
                "component": "evaluation",
                "event": "evaluation_failed",
                "round_id": round_id,
                "improvement": improvement
            })
        else:
            logger.info(f"Adapter evaluation passed: improvement={improvement:.4f}", extra={
                "component": "evaluation",
                "event": "evaluation_passed",
                "round_id": round_id,
                "improvement": improvement
            })
    else:
        # First adapter, no comparison
        logger.info(f"First adapter evaluation (no previous to compare)", extra={
            "component": "evaluation",
            "event": "evaluation_passed",
            "round_id": round_id
        })
    
    result = AdapterEvaluationResult(
        round_id=round_id,
        adapter_version=adapter_version,
        evaluation_loss=evaluation_loss,
        num_eval_samples=0,  # Placeholder
        passed=passed,
        previous_loss=previous_adapter_loss,
        improvement=improvement
    )
    
    return result

