"""
Round Manager Module

Manages federated learning rounds, tracks clients, and maintains round states.
"""

from enum import Enum
from typing import Dict, Set, Optional
from dataclasses import dataclass, field


class RoundState(Enum):
    """Round state enumeration."""
    OPEN = "OPEN"
    COLLECTING = "COLLECTING"
    AGGREGATING = "AGGREGATING"
    CLOSED = "CLOSED"


@dataclass
class Round:
    """Represents a federated learning round."""
    round_id: int
    model_version: str  # Model version used for this round (e.g., "v1", "v2")
    state: RoundState = RoundState.OPEN
    assigned_clients: Set[str] = field(default_factory=set)
    updates_received: Set[str] = field(default_factory=set)


class RoundManager:
    """
    Manages rounds and client registrations for federated learning.
    
    Tracks registered clients, rounds, and their states.
    """
    
    def __init__(self):
        """Initialize the round manager."""
        self.clients: Set[str] = set()
        self.rounds: Dict[int, Round] = {}
        self.client_round_assignments: Dict[str, int] = {}
        self.next_round_id: int = 1
    
    def register_client(self, client_name: str) -> bool:
        """
        Register a new client.
        
        Args:
            client_name: Unique identifier for the client
            
        Returns:
            True if client was registered, False if already exists
        """
        if client_name in self.clients:
            return False
        self.clients.add(client_name)
        return True
    
    def assign_client_to_round(self, client_id: str, model_version: str) -> Optional[int]:
        """
        Assign a client to the current active round.
        
        Args:
            client_id: Identifier of the client to assign
            model_version: Model version to use for this round (e.g., "v1", "v2")
            
        Returns:
            Round ID if assignment successful, None otherwise
        """
        if client_id not in self.clients:
            return None
        
        # Check if client is already assigned to an active round
        if client_id in self.client_round_assignments:
            assigned_round_id = self.client_round_assignments[client_id]
            assigned_round = self.rounds.get(assigned_round_id)
            if assigned_round:
                # If round is complete (all updates received), clear assignment
                if len(assigned_round.updates_received) >= len(assigned_round.assigned_clients) and len(assigned_round.assigned_clients) > 0:
                    # Round is complete, clear assignment
                    del self.client_round_assignments[client_id]
                elif assigned_round.state in [RoundState.OPEN, RoundState.COLLECTING]:
                    # Round is still active and not complete
                    # Verify model version matches
                    if assigned_round.model_version == model_version:
                        return None
                    else:
                        # Model version mismatch, clear assignment
                        del self.client_round_assignments[client_id]
        
        # Find or create an active round with matching model version
        active_round = None
        for round_id, round_obj in self.rounds.items():
            if round_obj.state in [RoundState.OPEN, RoundState.COLLECTING]:
                # Must match model version
                if round_obj.model_version != model_version:
                    continue
                # Check if all assigned clients have submitted updates
                if len(round_obj.updates_received) >= len(round_obj.assigned_clients) and len(round_obj.assigned_clients) > 0:
                    # All clients have submitted, skip this round and create a new one
                    continue
                active_round = round_obj
                break
        
        if active_round is None:
            # Create new round with specified model version
            active_round = Round(round_id=self.next_round_id, model_version=model_version)
            self.rounds[self.next_round_id] = active_round
            self.next_round_id += 1
        
        active_round.assigned_clients.add(client_id)
        self.client_round_assignments[client_id] = active_round.round_id
        
        if active_round.state == RoundState.OPEN:
            active_round.state = RoundState.COLLECTING
        
        return active_round.round_id
    
    def validate_update(self, client_id: str, round_id: int) -> bool:
        """
        Validate if a client update is allowed for a given round.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
            
        Returns:
            True if update is valid, False otherwise
        """
        if client_id not in self.clients:
            return False
        
        round_obj = self.rounds.get(round_id)
        if round_obj is None:
            return False
        
        if client_id not in round_obj.assigned_clients:
            return False
        
        if round_obj.state not in [RoundState.COLLECTING, RoundState.AGGREGATING]:
            return False
        
        return True
    
    def add_update(self, client_id: str, round_id: int, weight_delta: str) -> bool:
        """
        Record that a client has submitted an update.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
            weight_delta: The weight delta update (as string in MVP)
            
        Returns:
            True if update was recorded, False otherwise
        """
        if not self.validate_update(client_id, round_id):
            return False
        
        round_obj = self.rounds[round_id]
        round_obj.updates_received.add(client_id)
        
        return True
    
    def get_round_status(self, round_id: int) -> Optional[Dict]:
        """
        Get the status of a round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            Dictionary with round status information, None if round doesn't exist
        """
        round_obj = self.rounds.get(round_id)
        if round_obj is None:
            return None
        
        return {
            "round_id": round_obj.round_id,
            "model_version": round_obj.model_version,
            "state": round_obj.state.value,
            "assigned_clients": list(round_obj.assigned_clients),
            "updates_received": list(round_obj.updates_received),
            "total_clients": len(round_obj.assigned_clients),
            "total_updates": len(round_obj.updates_received)
        }
    
    def set_round_state(self, round_id: int, state: RoundState) -> bool:
        """
        Set the state of a round.
        
        Args:
            round_id: Identifier of the round
            state: New state for the round
            
        Returns:
            True if state was updated, False otherwise
        """
        round_obj = self.rounds.get(round_id)
        if round_obj is None:
            return False
        
        round_obj.state = state
        return True

