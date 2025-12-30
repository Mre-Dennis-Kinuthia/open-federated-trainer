"""
Update Validator Module

Validates client updates before aggregation.
"""

from typing import Optional
from .round_manager import RoundManager


class UpdateValidator:
    """
    Validates client updates for federated learning.
    
    Performs basic validation checks on weight deltas.
    """
    
    def __init__(self, round_manager: RoundManager):
        """
        Initialize the update validator.
        
        Args:
            round_manager: Round manager instance to coordinate with
        """
        self.round_manager = round_manager
    
    def validate(self, client_id: str, round_id: int, weight_delta: str) -> bool:
        """
        Validate a client update.
        
        Args:
            client_id: Identifier of the client submitting the update
            round_id: Identifier of the round
            weight_delta: The weight delta update (as string in MVP)
            
        Returns:
            True if update is valid, False otherwise
        """
        # Check if client is registered
        if client_id not in self.round_manager.clients:
            return False
        
        # Check if round exists and client is assigned to it
        if not self.round_manager.validate_update(client_id, round_id):
            return False
        
        # Basic validation: weight_delta should not be empty
        if not weight_delta or not isinstance(weight_delta, str):
            return False
        
        # Additional validation can be added here (e.g., size checks, format validation)
        # For MVP, we just check that it's a non-empty string
        
        return True

