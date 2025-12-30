"""
Aggregator Module

Collects and aggregates client updates for federated learning rounds.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from .round_manager import RoundManager, RoundState
from .model_store import ModelStore
from .versioning import next_version


@dataclass
class ClientUpdate:
    """Represents a client update."""
    client_id: str
    round_id: int
    weight_delta: str


class Aggregator:
    """
    Aggregates client updates for federated learning.
    
    Collects updates from clients and performs aggregation.
    """
    
    def __init__(self, round_manager: RoundManager, model_store: ModelStore = None, task_assigner = None):
        """
        Initialize the aggregator.
        
        Args:
            round_manager: Round manager instance to coordinate with
            model_store: Model store instance for persisting aggregated models
            task_assigner: Optional task assigner to update with new model version
        """
        self.round_manager = round_manager
        self.model_store = model_store or ModelStore()
        self.task_assigner = task_assigner
        self.updates: Dict[int, List[ClientUpdate]] = {}
    
    def submit_update(self, client_id: str, round_id: int, weight_delta: str) -> bool:
        """
        Submit a client update for aggregation.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
            weight_delta: The weight delta update (as string in MVP)
            
        Returns:
            True if update was submitted successfully, False otherwise
        """
        # Validate and record update in round manager
        if not self.round_manager.add_update(client_id, round_id, weight_delta):
            return False
        
        # Store update for aggregation
        if round_id not in self.updates:
            self.updates[round_id] = []
        
        # Check if client already submitted an update for this round
        existing_update = next(
            (u for u in self.updates[round_id] if u.client_id == client_id),
            None
        )
        
        if existing_update:
            # Update existing update
            existing_update.weight_delta = weight_delta
        else:
            # Add new update
            update = ClientUpdate(
                client_id=client_id,
                round_id=round_id,
                weight_delta=weight_delta
            )
            self.updates[round_id].append(update)
        
        return True
    
    def aggregate(self, round_id: int) -> Optional[Dict]:
        """
        Aggregate all updates for a round.
        
        Args:
            round_id: Identifier of the round to aggregate
            
        Returns:
            Dictionary with aggregated model information, or None if aggregation fails
        """
        # Check if round exists
        round_status = self.round_manager.get_round_status(round_id)
        if round_status is None:
            return None
        
        # Check if round is in a valid state for aggregation
        round_obj = self.round_manager.rounds.get(round_id)
        if round_obj is None:
            return None
        
        # Update round state to AGGREGATING
        self.round_manager.set_round_state(round_id, RoundState.AGGREGATING)
        
        # Get all updates for this round
        round_updates = self.updates.get(round_id, [])
        
        if not round_updates:
            return {
                "round_id": round_id,
                "status": "no_updates",
                "aggregated_model": None,
                "num_updates": 0
            }
        
        # Get the model version used for this round
        round_model_version = round_obj.model_version
        
        # Simple aggregation: collect all weight deltas
        # In a real implementation, this would perform federated averaging
        aggregated_weight_deltas = [update.weight_delta for update in round_updates]
        
        # Generate new model version
        new_model_version = next_version(round_model_version)
        
        # Create aggregated model data
        aggregated_model_data = {
            "version": new_model_version,
            "base_version": round_model_version,
            "round_id": round_id,
            "weight_deltas": aggregated_weight_deltas,
            "num_updates": len(aggregated_weight_deltas),
            "client_ids": [update.client_id for update in round_updates],
            "aggregation_timestamp": None  # Could add timestamp if needed
        }
        
        # Persist the new model version to disk
        try:
            self.model_store.save_model(new_model_version, aggregated_model_data)
            
            # Update task assigner with new model version for future assignments
            if self.task_assigner:
                self.task_assigner.set_model_version(new_model_version)
        except Exception as e:
            # If persistence fails, still mark round as closed but log error
            # In production, you might want to handle this differently
            print(f"Warning: Failed to persist model {new_model_version}: {e}")
        
        # Mark round as closed after aggregation
        self.round_manager.set_round_state(round_id, RoundState.CLOSED)
        
        return {
            "round_id": round_id,
            "model_version": new_model_version,
            "status": "aggregated",
            "aggregated_model": aggregated_model_data,
            "num_updates": len(round_updates)
        }
    
    def get_updates_for_round(self, round_id: int) -> List[ClientUpdate]:
        """
        Get all updates for a specific round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            List of client updates for the round
        """
        return self.updates.get(round_id, [])

