"""
Federated Averaging for LoRA Adapters

Implements FedAvg specifically for LoRA adapter weights.
Only adapter parameters are aggregated, base model weights remain fixed.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from utils.logger import get_logger

logger = get_logger("aggregation")


def validate_adapter(adapter_state_dict: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate an adapter state dict.
    
    Checks for:
    - Valid structure
    - No NaN or Inf values
    - Reasonable value ranges
    
    Args:
        adapter_state_dict: LoRA adapter state dict
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(adapter_state_dict, dict):
        return False, "Adapter must be a dictionary"
    
    if len(adapter_state_dict) == 0:
        return False, "Adapter state dict is empty"
    
    # Check for NaN and Inf
    for key, value in adapter_state_dict.items():
        if not isinstance(key, str):
            return False, f"Invalid key type: {type(key)}"
        
        # Convert to tensor if it's a list/array
        if isinstance(value, (list, np.ndarray)):
            tensor = torch.tensor(value) if isinstance(value, list) else torch.from_numpy(value)
        elif isinstance(value, torch.Tensor):
            tensor = value
        else:
            return False, f"Invalid value type for key {key}: {type(value)}"
        
        # Check for NaN
        if torch.isnan(tensor).any():
            return False, f"NaN values found in {key}"
        
        # Check for Inf
        if torch.isinf(tensor).any():
            return False, f"Inf values found in {key}"
        
        # Check for reasonable ranges (adapter weights should be small)
        if tensor.numel() > 0:
            max_val = torch.abs(tensor).max().item()
            if max_val > 100.0:  # Reasonable threshold
                logger.warning(f"Large values in {key}: max={max_val}")
    
    return True, None


def aggregate_lora_adapters(
    adapter_submissions: Dict[str, Dict],
    weight_by_samples: bool = True
) -> Optional[Dict]:
    """
    Aggregate LoRA adapters using federated averaging.
    
    Implements FedAvg:
    w_global = Σ(n_i * w_i) / Σ(n_i)
    
    where n_i is the number of samples for client i.
    
    Args:
        adapter_submissions: Dictionary mapping client_id to submission data
                           Each submission should have:
                           - adapter_state_dict: LoRA adapter weights
                           - num_samples: Number of training samples
        weight_by_samples: If True, weight by num_samples; otherwise uniform weighting
        
    Returns:
        Aggregated adapter state dict, or None if aggregation fails
    """
    if not adapter_submissions:
        logger.warning("No adapter submissions to aggregate")
        return None
    
    # Validate all adapters first
    valid_submissions = {}
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
        logger.error("No valid adapter submissions after validation")
        return None
    
    logger.info(f"Aggregating {len(valid_submissions)} adapters", extra={
        "component": "aggregation",
        "event": "aggregation_started",
        "num_adapters": len(valid_submissions)
    })
    
    # Get all parameter keys (should be consistent across adapters)
    first_client = list(valid_submissions.keys())[0]
    first_adapter = valid_submissions[first_client]["adapter_state_dict"]
    param_keys = list(first_adapter.keys())
    
    # Verify all adapters have the same keys
    for client_id, submission in valid_submissions.items():
        adapter_keys = set(submission["adapter_state_dict"].keys())
        if adapter_keys != set(param_keys):
            logger.warning(f"Client {client_id} adapter has mismatched keys")
            logger.warning(f"Expected: {set(param_keys)}, Got: {adapter_keys}")
            # Remove from valid submissions
            del valid_submissions[client_id]
    
    if not valid_submissions:
        logger.error("No valid adapters after key validation")
        return None
    
    # Aggregate each parameter
    aggregated_adapter = {}
    total_samples = 0
    
    for param_key in param_keys:
        # Collect all values for this parameter
        param_values = []
        sample_weights = []
        
        for client_id, submission in valid_submissions.items():
            adapter_state_dict = submission["adapter_state_dict"]
            param_value = adapter_state_dict[param_key]
            
            # Convert to tensor if needed
            if isinstance(param_value, (list, np.ndarray)):
                tensor = torch.tensor(param_value) if isinstance(param_value, list) else torch.from_numpy(param_value)
            elif isinstance(param_value, torch.Tensor):
                tensor = param_value.clone()
            else:
                logger.warning(f"Invalid parameter type for {param_key} from {client_id}")
                continue
            
            param_values.append(tensor)
            
            if weight_by_samples:
                num_samples = submission.get("num_samples", 1)
                sample_weights.append(num_samples)
                total_samples += num_samples
            else:
                sample_weights.append(1.0)
        
        if not param_values:
            logger.warning(f"No valid values for parameter {param_key}")
            continue
        
        # Normalize weights
        if weight_by_samples and total_samples > 0:
            weights = [w / total_samples for w in sample_weights]
        else:
            # Uniform weighting
            uniform_weight = 1.0 / len(sample_weights)
            weights = [uniform_weight] * len(sample_weights)
        
        # Weighted average
        aggregated_param = torch.zeros_like(param_values[0])
        for tensor, weight in zip(param_values, weights):
            aggregated_param += weight * tensor
        
        # Convert to list for JSON serialization
        aggregated_adapter[param_key] = aggregated_param.cpu().numpy().tolist()
    
    logger.info(f"Aggregation completed: {len(aggregated_adapter)} parameters", extra={
        "component": "aggregation",
        "event": "aggregation_completed",
        "num_parameters": len(aggregated_adapter),
        "num_clients": len(valid_submissions)
    })
    
    return aggregated_adapter

