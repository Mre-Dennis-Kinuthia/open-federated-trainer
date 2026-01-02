"""
Incentive Mechanism (Simulation)

Implements a token-based reward system for federated learning clients.
This is a simulation for research purposes, not real currency.
"""

import time
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ClientIncentives:
    """Incentive data for a single client."""
    client_id: str
    
    # Token balances
    total_tokens_earned: float = 0.0
    tokens_spent: float = 0.0
    
    # Reward history
    rewards_received: list[Dict] = field(default_factory=list)
    
    # Performance bonuses
    speed_bonuses: int = 0
    consistency_bonuses: int = 0
    
    @property
    def current_balance(self) -> float:
        """Calculate current token balance."""
        return self.total_tokens_earned - self.tokens_spent
    
    def to_dict(self) -> Dict:
        """Convert incentives to dictionary."""
        return {
            "client_id": self.client_id,
            "total_tokens_earned": self.total_tokens_earned,
            "tokens_spent": self.tokens_spent,
            "current_balance": self.current_balance,
            "speed_bonuses": self.speed_bonuses,
            "consistency_bonuses": self.consistency_bonuses,
            "total_rewards": len(self.rewards_received)
        }


class IncentiveManager:
    """
    Manages token-based incentives for federated learning clients.
    
    This is a simulation system for research, not real currency.
    """
    
    def __init__(
        self,
        base_reward_per_update: float = 10.0,
        speed_bonus_threshold: float = 30.0,  # seconds
        consistency_bonus_threshold: int = 5  # consecutive rounds
    ):
        """
        Initialize the incentive manager.
        
        Args:
            base_reward_per_update: Base tokens awarded per accepted update
            speed_bonus_threshold: Latency threshold for speed bonus (seconds)
            consistency_bonus_threshold: Consecutive rounds for consistency bonus
        """
        self.base_reward_per_update = base_reward_per_update
        self.speed_bonus_threshold = speed_bonus_threshold
        self.consistency_bonus_threshold = consistency_bonus_threshold
        
        self.client_incentives: Dict[str, ClientIncentives] = {}
        # Track consecutive completions for consistency bonus
        self.consecutive_completions: Dict[str, int] = defaultdict(int)
        # Track last completion time for speed bonus
        self.last_completion_times: Dict[str, float] = {}
    
    def _get_or_create_client(self, client_id: str) -> ClientIncentives:
        """Get or create incentive record for a client."""
        if client_id not in self.client_incentives:
            self.client_incentives[client_id] = ClientIncentives(client_id=client_id)
        return self.client_incentives[client_id]
    
    def award_update_reward(
        self,
        client_id: str,
        round_id: int,
        latency_seconds: Optional[float] = None
    ) -> float:
        """
        Award tokens for a successful update submission.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
            latency_seconds: Optional latency for speed bonus calculation
            
        Returns:
            Total tokens awarded (base + bonuses)
        """
        client = self._get_or_create_client(client_id)
        
        # Base reward
        tokens = self.base_reward_per_update
        
        # Speed bonus (if update was fast)
        speed_bonus = 0.0
        if latency_seconds is not None and latency_seconds < self.speed_bonus_threshold:
            speed_bonus = self.base_reward_per_update * 0.5  # 50% bonus
            tokens += speed_bonus
            client.speed_bonuses += 1
        
        # Consistency bonus (if client has been consistent)
        consistency_bonus = 0.0
        consecutive = self.consecutive_completions[client_id]
        if consecutive >= self.consistency_bonus_threshold:
            consistency_bonus = self.base_reward_per_update * 0.3  # 30% bonus
            tokens += consistency_bonus
            client.consistency_bonuses += 1
            # Reset counter after bonus
            self.consecutive_completions[client_id] = 0
        
        # Record reward
        client.total_tokens_earned += tokens
        client.rewards_received.append({
            "round_id": round_id,
            "tokens": tokens,
            "base": self.base_reward_per_update,
            "speed_bonus": speed_bonus,
            "consistency_bonus": consistency_bonus,
            "timestamp": time.time()
        })
        
        # Update consecutive completions
        self.consecutive_completions[client_id] += 1
        self.last_completion_times[client_id] = time.time()
        
        return tokens
    
    def record_dropout(self, client_id: str) -> None:
        """
        Record a client dropout (resets consistency counter).
        
        Args:
            client_id: Identifier of the client
        """
        self.consecutive_completions[client_id] = 0
    
    def get_client_incentives(self, client_id: str) -> Optional[ClientIncentives]:
        """
        Get incentive data for a client.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            ClientIncentives object or None if not found
        """
        return self.client_incentives.get(client_id)
    
    def get_client_balance(self, client_id: str) -> float:
        """
        Get current token balance for a client.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            Current token balance (0.0 if client not found)
        """
        client = self.client_incentives.get(client_id)
        if client:
            return client.current_balance
        return 0.0
    
    def get_all_incentives(self) -> Dict[str, Dict]:
        """
        Get all client incentive data.
        
        Returns:
            Dictionary mapping client_id to incentive data
        """
        return {
            client_id: client.to_dict()
            for client_id, client in self.client_incentives.items()
        }
    
    def get_top_earners(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N clients by token earnings.
        
        Args:
            n: Number of top earners to return
            
        Returns:
            List of (client_id, total_earned) tuples, sorted by earnings (descending)
        """
        earners = [
            (client_id, client.total_tokens_earned)
            for client_id, client in self.client_incentives.items()
        ]
        earners.sort(key=lambda x: x[1], reverse=True)
        return earners[:n]

