"""
Task Assigner Module

Assigns training tasks to registered clients and prevents duplicate assignments.
"""

from typing import Dict, Optional
from .round_manager import RoundManager
from .versioning import initial_version
from .model_store import ModelStore


class TaskAssigner:
    """
    Assigns tasks to clients for federated learning rounds.
    
    Prevents duplicate task assignments and tracks model versions.
    """
    
    def __init__(self, round_manager: RoundManager, model_store: ModelStore = None):
        """
        Initialize the task assigner.
        
        Args:
            round_manager: Round manager instance to coordinate with
            model_store: Model store instance for version management
        """
        self.round_manager = round_manager
        self.model_store = model_store or ModelStore()
        
        # Initialize model version from latest on disk, or use initial version
        latest_version = self.model_store.latest_model_version()
        self.model_version: str = latest_version if latest_version else initial_version()
        
        self.client_assignments: Dict[str, Dict] = {}
    
    def assign_task(self, client_id: str) -> Optional[Dict]:
        """
        Assign a task to a client.
        
        Args:
            client_id: Identifier of the client requesting a task
            
        Returns:
            Dictionary with round_id, model_version, and task information,
            or None if assignment fails
        """
        # Check if client is registered
        if client_id not in self.round_manager.clients:
            return None
        
        # Check if client already has an active assignment
        if client_id in self.client_assignments:
            existing_assignment = self.client_assignments[client_id]
            round_id = existing_assignment.get("round_id")
            round_status = self.round_manager.get_round_status(round_id)
            
            # If round is still active and not all updates received, return existing assignment
            if round_status and round_status["state"] in ["OPEN", "COLLECTING"]:
                # Check if all clients have submitted (round is ready for aggregation)
                if round_status["total_updates"] < round_status["total_clients"]:
                    return existing_assignment
                # Round is complete, clear assignment and get new task
                del self.client_assignments[client_id]
        
        # Assign client to a round with current model version
        round_id = self.round_manager.assign_client_to_round(client_id, self.model_version)
        if round_id is None:
            return None
        
        # Create task assignment
        task = {
            "round_id": round_id,
            "model_version": self.model_version,
            "task": "train",
            "description": f"Train model version {self.model_version} for round {round_id}"
        }
        
        # Store assignment
        self.client_assignments[client_id] = task
        
        return task
    
    def get_client_assignment(self, client_id: str) -> Optional[Dict]:
        """
        Get the current assignment for a client.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            Current assignment dictionary or None if no assignment exists
        """
        return self.client_assignments.get(client_id)
    
    def set_model_version(self, version: str) -> None:
        """
        Set the current model version.
        
        Args:
            version: Model version string (e.g., "v1", "v2")
        """
        self.model_version = version
    
    def get_model_version(self) -> str:
        """
        Get the current model version.
        
        Returns:
            Current model version string
        """
        return self.model_version

