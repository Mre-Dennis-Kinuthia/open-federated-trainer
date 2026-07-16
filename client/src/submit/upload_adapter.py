"""
Upload Adapter Module

Submits LoRA adapters to the coordinator.
"""

import requests
import hashlib
import json
from typing import Dict, Optional
from config import config
from security import get_api_key
from utils.logger import get_logger

logger = get_logger("upload_adapter")


def upload_adapter(
    client_id: str,
    round_id: int,
    adapter_state_dict: Dict,
    num_samples: int,
    training_loss: float,
    api_key: Optional[str] = None
) -> bool:
    """
    Upload LoRA adapter to coordinator.
    
    Args:
        client_id: Client identifier
        round_id: Round identifier
        adapter_state_dict: LoRA adapter state dict
        num_samples: Number of training samples
        training_loss: Final training loss
        api_key: Optional API key (uses security.get_api_key() if not provided)
        
    Returns:
        True if upload successful, False otherwise
        
    Raises:
        requests.RequestException: If request fails
    """
    if api_key is None:
        api_key = get_api_key()
    
    url = f"{config.COORDINATOR_URL}/rounds/{round_id}/submit"
    
    # Compute adapter hash for verification
    adapter_json = json.dumps(adapter_state_dict, sort_keys=True)
    adapter_hash = hashlib.sha256(adapter_json.encode()).hexdigest()[:16]
    
    payload = {
        "client_id": client_id,
        "round_id": round_id,
        "adapter_state_dict": adapter_state_dict,
        "num_samples": num_samples,
        "training_loss": training_loss,
        "api_key": api_key
    }
    
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=60.0
        )
        response.raise_for_status()
        
        result = response.json()
        success = result.get("success", False)
        
        if success:
            logger.info(f"Adapter uploaded successfully for round {round_id}", extra={
                "component": "client",
                "event": "adapter_uploaded",
                "round_id": round_id,
                "adapter_hash": adapter_hash
            })
        else:
            logger.warning(f"Adapter upload returned success=False", extra={
                "component": "client",
                "event": "adapter_upload_failed",
                "round_id": round_id
            })
        
        return success
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to upload adapter: {e}", extra={
            "component": "client",
            "event": "adapter_upload_error",
            "round_id": round_id,
            "error": str(e)
        })
        raise


def submit_lora_adapter(
    client_id: str,
    round_id: int,
    adapter_state_dict: Dict,
    num_samples: int,
    training_loss: float,
    api_key: Optional[str] = None
) -> bool:
    """
    Alias for upload_adapter for consistency.
    
    Args:
        client_id: Client identifier
        round_id: Round identifier
        adapter_state_dict: LoRA adapter state dict
        num_samples: Number of training samples
        training_loss: Final training loss
        api_key: Optional API key
        
    Returns:
        True if submission successful, False otherwise
    """
    return upload_adapter(
        client_id=client_id,
        round_id=round_id,
        adapter_state_dict=adapter_state_dict,
        num_samples=num_samples,
        training_loss=training_loss,
        api_key=api_key
    )

