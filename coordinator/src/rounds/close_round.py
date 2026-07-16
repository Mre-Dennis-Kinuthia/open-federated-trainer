"""
Close LoRA Round

Closes a LoRA fine-tuning round after aggregation.
"""

from .create_round import get_lora_round_manager


def close_lora_round(round_id: int) -> bool:
    """
    Close a LoRA round.
    
    Args:
        round_id: Round identifier
        
    Returns:
        True if closed successfully, False otherwise
    """
    manager = get_lora_round_manager()
    return manager.close_round(round_id)

