"""
Trainer module for local model training.

Simulates local training and returns deterministic weight deltas.
This module is designed to be replaceable with real PyTorch/TensorFlow training.
"""

import hashlib
import json
from typing import Dict, Any


def train_local_model(task: Dict[str, Any]) -> str:
    """
    Simulate local model training and return a weight delta.
    
    This is a placeholder implementation that returns a deterministic
    fake weight delta based on the task parameters. In production,
    this would perform actual model training using PyTorch, TensorFlow, etc.
    
    Args:
        task: Task dictionary containing round_id, model_version, task, description
        
    Returns:
        Weight delta as a string (in MVP). In production, this would be
        actual model weights or gradients.
    """
    round_id = task.get("round_id", 0)
    model_version = task.get("model_version", 0)
    task_type = task.get("task", "train")
    
    # Create a deterministic "weight delta" based on task parameters
    # This ensures the same task always produces the same delta (for testing)
    delta_data = {
        "round_id": round_id,
        "model_version": model_version,
        "task_type": task_type,
        "client_id": "simulated",  # Will be replaced by actual client_id in client.py
        "training_steps": 10,  # Simulated training steps
        "loss": 0.5 - (round_id * 0.01),  # Simulated decreasing loss
    }
    
    # Create a deterministic hash-based weight delta
    delta_str = json.dumps(delta_data, sort_keys=True)
    delta_hash = hashlib.md5(delta_str.encode()).hexdigest()
    
    # Return a formatted weight delta string
    weight_delta = f"delta_r{round_id}_v{model_version}_{delta_hash[:8]}"
    
    return weight_delta


def train_local_model_with_client_id(task: Dict[str, Any], client_id: str) -> str:
    """
    Simulate local model training with client ID included in the delta.
    
    This version includes the client_id in the weight delta for better
    traceability and deterministic behavior.
    
    Args:
        task: Task dictionary containing round_id, model_version, task, description
        client_id: Identifier of the client performing training
        
    Returns:
        Weight delta as a string
    """
    round_id = task.get("round_id", 0)
    model_version = task.get("model_version", 0)
    task_type = task.get("task", "train")
    
    # Create a deterministic "weight delta" based on task parameters and client_id
    delta_data = {
        "round_id": round_id,
        "model_version": model_version,
        "task_type": task_type,
        "client_id": client_id,
        "training_steps": 10,
        "loss": 0.5 - (round_id * 0.01),
    }
    
    # Create a deterministic hash-based weight delta
    delta_str = json.dumps(delta_data, sort_keys=True)
    delta_hash = hashlib.md5(delta_str.encode()).hexdigest()
    
    # Return a formatted weight delta string
    weight_delta = f"delta_{client_id}_r{round_id}_v{model_version}_{delta_hash[:8]}"
    
    return weight_delta

