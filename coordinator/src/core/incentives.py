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
    Optional SQL repo enables multi-replica shared state.
    """
    
    def __init__(
        self,
        base_reward_per_update: float = 10.0,
        speed_bonus_threshold: float = 30.0,  # seconds
        consistency_bonus_threshold: int = 5,  # consecutive rounds
        repo=None,
    ):
        """
        Initialize the incentive manager.
        
        Args:
            base_reward_per_update: Base tokens awarded per accepted update
            speed_bonus_threshold: Latency threshold for speed bonus (seconds)
            consistency_bonus_threshold: Consecutive rounds for consistency bonus
            repo: Optional shared SQL repository
        """
        self.base_reward_per_update = base_reward_per_update
        self.speed_bonus_threshold = speed_bonus_threshold
        self.consistency_bonus_threshold = consistency_bonus_threshold
        self.repo = repo
        
        self.client_incentives: Dict[str, ClientIncentives] = {}
        # Track consecutive completions for consistency bonus
        self.consecutive_completions: Dict[str, int] = defaultdict(int)
        # Track last completion time for speed bonus
        self.last_completion_times: Dict[str, float] = {}
        if self.repo is not None:
            self._reload_from_repo()

    def _reload_from_repo(self) -> None:
        if self.repo is None:
            return
        for data in self.repo.list_all():
            client_id = data["client_id"]
            self.client_incentives[client_id] = ClientIncentives(
                client_id=client_id,
                total_tokens_earned=float(data.get("total_tokens_earned", 0.0)),
                tokens_spent=float(data.get("tokens_spent", 0.0)),
                rewards_received=list(data.get("rewards_received") or []),
                speed_bonuses=int(data.get("speed_bonuses", 0)),
                consistency_bonuses=int(data.get("consistency_bonuses", 0)),
            )
            self.consecutive_completions[client_id] = int(
                data.get("consecutive_completions", 0)
            )
            if data.get("last_completion_time") is not None:
                self.last_completion_times[client_id] = float(
                    data["last_completion_time"]
                )

    def _persist(self, client_id: str) -> None:
        if self.repo is None or client_id not in self.client_incentives:
            return
        client = self.client_incentives[client_id]
        self.repo.save(
            {
                "client_id": client.client_id,
                "total_tokens_earned": client.total_tokens_earned,
                "tokens_spent": client.tokens_spent,
                "speed_bonuses": client.speed_bonuses,
                "consistency_bonuses": client.consistency_bonuses,
                "rewards_received": list(client.rewards_received),
                "consecutive_completions": int(
                    self.consecutive_completions.get(client_id, 0)
                ),
                "last_completion_time": self.last_completion_times.get(client_id),
            }
        )
    
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
        self._persist(client_id)
        
        return tokens
    
    def record_dropout(self, client_id: str) -> None:
        """
        Record a client dropout (resets consistency counter).
        
        Args:
            client_id: Identifier of the client
        """
        self.consecutive_completions[client_id] = 0
        if client_id in self.client_incentives:
            self._persist(client_id)
    
    def get_client_incentives(self, client_id: str) -> Optional[ClientIncentives]:
        if self.repo is not None:
            self._reload_from_repo()
        return self.client_incentives.get(client_id)
    
    def get_client_balance(self, client_id: str) -> float:
        if self.repo is not None:
            self._reload_from_repo()
        client = self.client_incentives.get(client_id)
        if client:
            return client.current_balance
        return 0.0
    
    def get_all_incentives(self) -> Dict[str, Dict]:
        if self.repo is not None:
            self._reload_from_repo()
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

