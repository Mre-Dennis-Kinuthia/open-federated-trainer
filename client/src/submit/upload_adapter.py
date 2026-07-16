"""
Upload Adapter Module

Submits LoRA adapters to the coordinator.
"""

import requests
import hashlib
import json
import time
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
    
    last_error: Optional[requests.exceptions.RequestException] = None
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=max(config.REQUEST_TIMEOUT, 60.0),
            )
            if response.status_code < 500:
                response.raise_for_status()
            elif attempt < config.MAX_RETRIES:
                raise requests.exceptions.HTTPError(
                    f"Coordinator returned {response.status_code}",
                    response=response,
                )
            else:
                response.raise_for_status()

            success = response.json().get("success", False)
            if success:
                logger.info(f"Adapter uploaded successfully for round {round_id}", extra={
                    "component": "client",
                    "event": "adapter_uploaded",
                    "round_id": round_id,
                    "adapter_hash": adapter_hash
                })
            return success
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if (
                getattr(exc, "response", None) is not None
                and exc.response.status_code < 500
            ):
                raise
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_DELAY * (2**attempt))
                continue
            logger.error(f"Failed to upload adapter: {exc}", extra={
                "component": "client",
                "event": "adapter_upload_error",
                "round_id": round_id,
                "error": str(exc)
            })
    if last_error:
        raise last_error
    return False


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

