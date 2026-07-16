"""
Get LoRA Round Configuration

Retrieves LoRA round configuration for clients.
"""

from typing import Optional
from .create_round import get_lora_round_manager, LoRARoundConfig


def get_lora_round(round_id: int) -> Optional[LoRARoundConfig]:
    """
    Get LoRA round configuration.
    
    Args:
        round_id: Round identifier
        
    Returns:
        LoRARoundConfig if found, None otherwise
    """
    manager = get_lora_round_manager()
    return manager.get_round(round_id)

