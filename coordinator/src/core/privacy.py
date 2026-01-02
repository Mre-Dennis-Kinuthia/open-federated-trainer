"""
Privacy Safeguards Module

Provides lightweight privacy protections for federated learning updates.
Includes gradient clipping and optional noise injection.
"""

import os
import json
import random
import math
from typing import Dict, Any, Optional, List


class PrivacyProtector:
    """
    Applies privacy safeguards to client updates.
    
    Implements gradient clipping and optional noise injection.
    """
    
    def __init__(
        self,
        max_norm: Optional[float] = None,
        noise_scale: Optional[float] = None,
        enable_noise: bool = False
    ):
        """
        Initialize the privacy protector.
        
        Args:
            max_norm: Maximum L2 norm for gradient clipping (None = disabled)
            noise_scale: Standard deviation for Gaussian noise (None = disabled)
            enable_noise: Whether to enable noise injection
        """
        # Get from environment if not provided
        self.max_norm = max_norm or float(os.getenv("PRIVACY_MAX_NORM", "10.0"))
        self.noise_scale = noise_scale or float(os.getenv("PRIVACY_NOISE_SCALE", "0.01"))
        self.enable_noise = enable_noise or os.getenv("PRIVACY_ENABLE_NOISE", "false").lower() == "true"
    
    def clip_gradients(self, weight_delta: List[List[float]]) -> List[List[float]]:
        """
        Apply gradient clipping to weight deltas.
        
        Clips each parameter tensor to have L2 norm <= max_norm.
        
        Args:
            weight_delta: List of parameter tensors (each as list of floats)
            
        Returns:
            Clipped weight deltas
        """
        if self.max_norm <= 0:
            return weight_delta
        
        clipped = []
        for param_tensor in weight_delta:
            # Calculate L2 norm
            norm = math.sqrt(sum(x * x for x in param_tensor))
            
            if norm > self.max_norm:
                # Clip: scale down to max_norm
                scale = self.max_norm / norm
                clipped_tensor = [x * scale for x in param_tensor]
            else:
                clipped_tensor = param_tensor.copy()
            
            clipped.append(clipped_tensor)
        
        return clipped
    
    def add_noise(self, weight_delta: List[List[float]]) -> List[List[float]]:
        """
        Add Gaussian noise to weight deltas.
        
        Args:
            weight_delta: List of parameter tensors (each as list of floats)
            
        Returns:
            Noisy weight deltas
        """
        if not self.enable_noise or self.noise_scale <= 0:
            return weight_delta
        
        noisy = []
        for param_tensor in weight_delta:
            # Add Gaussian noise to each parameter
            noisy_tensor = [
                x + random.gauss(0.0, self.noise_scale)
                for x in param_tensor
            ]
            noisy.append(noisy_tensor)
        
        return noisy
    
    def protect_update(self, weight_delta_str: str) -> str:
        """
        Apply privacy protections to a weight delta update.
        
        This is the main entry point. It:
        1. Parses the JSON weight delta
        2. Applies gradient clipping
        3. Optionally adds noise
        4. Re-serializes to JSON
        
        Args:
            weight_delta_str: JSON-serialized weight delta
            
        Returns:
            Protected weight delta as JSON string
        """
        try:
            # Parse the update
            update_data = json.loads(weight_delta_str)
            
            # Extract weight delta
            if "weight_delta" in update_data:
                weight_delta = update_data["weight_delta"]
            else:
                # If structure is different, try to handle it
                # For now, assume it's the weight_delta directly
                weight_delta = update_data if isinstance(update_data, list) else []
            
            # Apply clipping
            clipped = self.clip_gradients(weight_delta)
            
            # Apply noise (if enabled)
            protected = self.add_noise(clipped)
            
            # Update the update_data structure
            if isinstance(update_data, dict):
                update_data["weight_delta"] = protected
                # Add privacy metadata
                update_data["privacy_applied"] = {
                    "clipping": self.max_norm > 0,
                    "noise": self.enable_noise,
                    "max_norm": self.max_norm if self.max_norm > 0 else None,
                    "noise_scale": self.noise_scale if self.enable_noise else None
                }
                return json.dumps(update_data, sort_keys=True)
            else:
                # If it was just a list, return the protected list
                return json.dumps(protected, sort_keys=True)
        
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # If parsing fails, return original (validation will catch it)
            return weight_delta_str
    
    def validate_update_values(self, weight_delta: List[List[float]]) -> tuple[bool, Optional[str]]:
        """
        Validate that update values are finite (no NaN or Inf).
        
        Args:
            weight_delta: List of parameter tensors
            
        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
        """
        for i, param_tensor in enumerate(weight_delta):
            for j, value in enumerate(param_tensor):
                if not math.isfinite(value):
                    return False, f"Non-finite value found in parameter {i}, element {j}: {value}"
        
        return True, None

