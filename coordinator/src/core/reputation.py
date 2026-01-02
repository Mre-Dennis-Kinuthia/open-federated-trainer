"""
Client Reputation System

Tracks client reliability and performance for federated learning.
Provides reputation scores to guide task assignment and identify reliable clients.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ClientReputation:
    """Reputation data for a single client."""
    client_id: str
    
    # Participation metrics
    rounds_participated: int = 0
    rounds_completed: int = 0
    rounds_dropped: int = 0
    
    # Update metrics
    updates_submitted: int = 0
    updates_accepted: int = 0
    updates_rejected: int = 0
    
    # Performance metrics
    total_latency_seconds: float = 0.0
    latency_samples: int = 0
    
    # Timestamps
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    
    @property
    def dropout_rate(self) -> float:
        """Calculate dropout rate (0-1)."""
        total_rounds = self.rounds_participated
        if total_rounds == 0:
            return 0.0
        return self.rounds_dropped / total_rounds
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate update acceptance rate (0-1)."""
        total_updates = self.updates_submitted
        if total_updates == 0:
            return 1.0
        return self.updates_accepted / total_updates
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency in seconds."""
        if self.latency_samples == 0:
            return 0.0
        return self.total_latency_seconds / self.latency_samples
    
    @property
    def completion_rate(self) -> float:
        """Calculate round completion rate (0-1)."""
        total_rounds = self.rounds_participated
        if total_rounds == 0:
            return 0.0
        return self.rounds_completed / total_rounds
    
    @property
    def reputation_score(self) -> float:
        """
        Calculate overall reputation score (0-1).
        
        Higher score = more reliable client.
        
        Factors:
        - Completion rate (40%)
        - Acceptance rate (30%)
        - Low dropout rate (20%)
        - Low latency (10% - inverse)
        """
        # Completion rate (higher is better)
        completion_weight = 0.4
        completion_score = self.completion_rate
        
        # Acceptance rate (higher is better)
        acceptance_weight = 0.3
        acceptance_score = self.acceptance_rate
        
        # Dropout rate (lower is better)
        dropout_weight = 0.2
        dropout_score = 1.0 - self.dropout_rate
        
        # Latency (lower is better, normalized)
        latency_weight = 0.1
        # Normalize latency: assume max reasonable latency is 60 seconds
        max_latency = 60.0
        normalized_latency = min(1.0, max(0.0, 1.0 - (self.average_latency / max_latency)))
        latency_score = normalized_latency
        
        # Weighted sum
        score = (
            completion_score * completion_weight +
            acceptance_score * acceptance_weight +
            dropout_score * dropout_weight +
            latency_score * latency_weight
        )
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))
    
    def to_dict(self) -> Dict:
        """Convert reputation to dictionary."""
        return {
            "client_id": self.client_id,
            "reputation_score": self.reputation_score,
            "rounds_participated": self.rounds_participated,
            "rounds_completed": self.rounds_completed,
            "rounds_dropped": self.rounds_dropped,
            "completion_rate": self.completion_rate,
            "updates_submitted": self.updates_submitted,
            "updates_accepted": self.updates_accepted,
            "updates_rejected": self.updates_rejected,
            "acceptance_rate": self.acceptance_rate,
            "dropout_rate": self.dropout_rate,
            "average_latency_seconds": self.average_latency,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen
        }


class ReputationManager:
    """
    Manages client reputation across the federated learning system.
    
    Tracks client behavior and calculates reputation scores.
    """
    
    def __init__(self):
        """Initialize the reputation manager."""
        self.reputations: Dict[str, ClientReputation] = {}
        # Track round participation (client_id -> set of round_ids)
        self.client_rounds: Dict[str, set] = defaultdict(set)
        # Track round start times for latency calculation
        self.round_start_times: Dict[int, float] = {}
    
    def register_client(self, client_id: str) -> None:
        """
        Register a new client or update last seen time.
        
        Args:
            client_id: Identifier of the client
        """
        if client_id not in self.reputations:
            self.reputations[client_id] = ClientReputation(
                client_id=client_id,
                first_seen=time.time()
            )
        
        self.reputations[client_id].last_seen = time.time()
    
    def record_round_participation(self, client_id: str, round_id: int) -> None:
        """
        Record that a client is participating in a round.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        self.reputations[client_id].rounds_participated += 1
        self.client_rounds[client_id].add(round_id)
    
    def record_round_start(self, round_id: int) -> None:
        """
        Record the start time of a round (for latency calculation).
        
        Args:
            round_id: Identifier of the round
        """
        self.round_start_times[round_id] = time.time()
    
    def record_update_submitted(self, client_id: str, round_id: int) -> None:
        """
        Record that a client submitted an update.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        self.reputations[client_id].updates_submitted += 1
        
        # Calculate latency if round start time is known
        if round_id in self.round_start_times:
            latency = time.time() - self.round_start_times[round_id]
            self.reputations[client_id].total_latency_seconds += latency
            self.reputations[client_id].latency_samples += 1
    
    def record_update_accepted(self, client_id: str, round_id: int) -> None:
        """
        Record that a client's update was accepted.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        self.reputations[client_id].updates_accepted += 1
    
    def record_update_rejected(self, client_id: str, round_id: int) -> None:
        """
        Record that a client's update was rejected.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        self.reputations[client_id].updates_rejected += 1
    
    def record_round_completion(self, client_id: str, round_id: int) -> None:
        """
        Record that a client completed a round.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        if round_id in self.client_rounds[client_id]:
            self.reputations[client_id].rounds_completed += 1
    
    def record_round_dropout(self, client_id: str, round_id: int) -> None:
        """
        Record that a client dropped out of a round.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        self.register_client(client_id)
        if round_id in self.client_rounds[client_id]:
            self.reputations[client_id].rounds_dropped += 1
    
    def get_reputation(self, client_id: str) -> Optional[ClientReputation]:
        """
        Get reputation for a client.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            ClientReputation object or None if client not found
        """
        return self.reputations.get(client_id)
    
    def get_reputation_score(self, client_id: str) -> float:
        """
        Get reputation score for a client (0-1).
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            Reputation score (0.0 if client not found)
        """
        rep = self.reputations.get(client_id)
        if rep:
            return rep.reputation_score
        return 0.0
    
    def get_all_reputations(self) -> Dict[str, Dict]:
        """
        Get all client reputations.
        
        Returns:
            Dictionary mapping client_id to reputation data
        """
        return {
            client_id: rep.to_dict()
            for client_id, rep in self.reputations.items()
        }
    
    def get_top_clients(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N clients by reputation score.
        
        Args:
            n: Number of top clients to return
            
        Returns:
            List of (client_id, score) tuples, sorted by score (descending)
        """
        scores = [
            (client_id, rep.reputation_score)
            for client_id, rep in self.reputations.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

