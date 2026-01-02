"""
Rate Limiting Module

Provides basic rate limiting to prevent abuse and spam.
Simple in-memory rate limiting suitable for research environments.
"""

import time
from typing import Dict, Optional, Tuple
from collections import defaultdict


class RateLimiter:
    """
    Simple in-memory rate limiter for federated learning.
    
    Tracks per-client request rates and update frequencies.
    """
    
    def __init__(
        self,
        max_updates_per_round: int = 5,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000
    ):
        """
        Initialize the rate limiter.
        
        Args:
            max_updates_per_round: Maximum updates a client can submit per round
            max_requests_per_minute: Maximum requests per minute per client
            max_requests_per_hour: Maximum requests per hour per client
        """
        self.max_updates_per_round = max_updates_per_round
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        
        # Per-client, per-round update counts
        # Structure: {client_id: {round_id: count}}
        self.updates_per_round: Dict[str, Dict[int, int]] = defaultdict(dict)
        
        # Per-client request timestamps
        # Structure: {client_id: [timestamp1, timestamp2, ...]}
        self.request_timestamps: Dict[str, list] = defaultdict(list)
        
        # Current round tracking (for cleanup)
        self.current_rounds: Dict[int, set] = {}  # round_id -> set of client_ids
    
    def check_update_rate(self, client_id: str, round_id: int) -> tuple[bool, Optional[str]]:
        """
        Check if a client can submit an update for a round.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
            
        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        # Get current count for this client/round
        client_rounds = self.updates_per_round[client_id]
        current_count = client_rounds.get(round_id, 0)
        
        if current_count >= self.max_updates_per_round:
            return False, f"Client {client_id} exceeded max updates per round ({self.max_updates_per_round})"
        
        # Track this round
        if round_id not in self.current_rounds:
            self.current_rounds[round_id] = set()
        self.current_rounds[round_id].add(client_id)
        
        return True, None
    
    def record_update(self, client_id: str, round_id: int) -> None:
        """
        Record that a client submitted an update.
        
        Args:
            client_id: Identifier of the client
            round_id: Identifier of the round
        """
        client_rounds = self.updates_per_round[client_id]
        client_rounds[round_id] = client_rounds.get(round_id, 0) + 1
    
    def check_request_rate(self, client_id: str) -> tuple[bool, Optional[str]]:
        """
        Check if a client can make a request (rate limit).
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        now = time.time()
        timestamps = self.request_timestamps[client_id]
        
        # Clean old timestamps (older than 1 hour)
        one_hour_ago = now - 3600
        timestamps[:] = [ts for ts in timestamps if ts > one_hour_ago]
        
        # Check hourly limit
        if len(timestamps) >= self.max_requests_per_hour:
            return False, f"Client {client_id} exceeded max requests per hour ({self.max_requests_per_hour})"
        
        # Check minute limit (last 60 seconds)
        one_minute_ago = now - 60
        recent_requests = [ts for ts in timestamps if ts > one_minute_ago]
        
        if len(recent_requests) >= self.max_requests_per_minute:
            return False, f"Client {client_id} exceeded max requests per minute ({self.max_requests_per_minute})"
        
        # Record this request
        timestamps.append(now)
        
        return True, None
    
    def record_request(self, client_id: str) -> None:
        """
        Record that a client made a request.
        
        Note: This is called automatically by check_request_rate,
        but can be called separately if needed.
        
        Args:
            client_id: Identifier of the client
        """
        now = time.time()
        self.request_timestamps[client_id].append(now)
    
    def reset_round(self, round_id: int) -> None:
        """
        Reset rate limiting for a completed round.
        
        Args:
            round_id: Identifier of the round to reset
        """
        if round_id in self.current_rounds:
            client_ids = self.current_rounds[round_id]
            for client_id in client_ids:
                if client_id in self.updates_per_round:
                    self.updates_per_round[client_id].pop(round_id, None)
            del self.current_rounds[round_id]
    
    def get_client_stats(self, client_id: str) -> Dict:
        """
        Get rate limiting statistics for a client.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            Dictionary with rate limiting stats
        """
        now = time.time()
        timestamps = self.request_timestamps.get(client_id, [])
        
        # Count requests in last minute
        one_minute_ago = now - 60
        requests_last_minute = len([ts for ts in timestamps if ts > one_minute_ago])
        
        # Count requests in last hour
        one_hour_ago = now - 3600
        requests_last_hour = len([ts for ts in timestamps if ts > one_hour_ago])
        
        return {
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "total_rounds_with_updates": len(self.updates_per_round.get(client_id, {}))
        }

