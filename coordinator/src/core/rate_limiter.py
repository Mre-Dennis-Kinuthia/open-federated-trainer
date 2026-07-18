"""
Rate Limiting Module

Provides basic rate limiting to prevent abuse and spam.
In-memory by default; SQL-backed when shared state is enabled.
"""

import time
from typing import Dict, Optional
from collections import defaultdict


class RateLimiter:
    """
    Simple rate limiter for federated learning.
    
    Tracks per-client request rates and update frequencies.
    """
    
    def __init__(
        self,
        max_updates_per_round: int = 5,
        max_requests_per_minute: int = 60,
        max_requests_per_hour: int = 1000,
        repo=None,
    ):
        self.max_updates_per_round = max_updates_per_round
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.repo = repo
        
        self.updates_per_round: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.request_timestamps: Dict[str, list] = defaultdict(list)
        self.current_rounds: Dict[int, set] = {}
    
    def check_update_rate(self, client_id: str, round_id: int) -> tuple[bool, Optional[str]]:
        if self.repo is not None:
            current_count = self.repo.get_update_count(client_id, round_id)
        else:
            client_rounds = self.updates_per_round[client_id]
            current_count = client_rounds.get(round_id, 0)
        
        if current_count >= self.max_updates_per_round:
            return False, f"Client {client_id} exceeded max updates per round ({self.max_updates_per_round})"
        
        if round_id not in self.current_rounds:
            self.current_rounds[round_id] = set()
        self.current_rounds[round_id].add(client_id)
        
        return True, None
    
    def record_update(self, client_id: str, round_id: int) -> None:
        if self.repo is not None:
            self.repo.incr_update_count(client_id, round_id)
            return
        client_rounds = self.updates_per_round[client_id]
        client_rounds[round_id] = client_rounds.get(round_id, 0) + 1
    
    def check_request_rate(self, client_id: str) -> tuple[bool, Optional[str]]:
        now = time.time()
        bucket = f"req:{client_id}"
        if self.repo is not None:
            timestamps = self.repo.get_timestamps(bucket)
        else:
            timestamps = self.request_timestamps[client_id]
        
        one_hour_ago = now - 3600
        timestamps = [ts for ts in timestamps if ts > one_hour_ago]
        
        if len(timestamps) >= self.max_requests_per_hour:
            return False, f"Client {client_id} exceeded max requests per hour ({self.max_requests_per_hour})"
        
        one_minute_ago = now - 60
        recent_requests = [ts for ts in timestamps if ts > one_minute_ago]
        
        if len(recent_requests) >= self.max_requests_per_minute:
            return False, f"Client {client_id} exceeded max requests per minute ({self.max_requests_per_minute})"
        
        timestamps.append(now)
        if self.repo is not None:
            self.repo.set_timestamps(bucket, timestamps)
        else:
            self.request_timestamps[client_id] = timestamps
        
        return True, None
    
    def record_request(self, client_id: str) -> None:
        now = time.time()
        if self.repo is not None:
            bucket = f"req:{client_id}"
            timestamps = self.repo.get_timestamps(bucket)
            timestamps.append(now)
            self.repo.set_timestamps(bucket, timestamps)
            return
        self.request_timestamps[client_id].append(now)
    
    def reset_round(self, round_id: int) -> None:
        if round_id in self.current_rounds:
            client_ids = self.current_rounds[round_id]
            for client_id in client_ids:
                if client_id in self.updates_per_round:
                    self.updates_per_round[client_id].pop(round_id, None)
            del self.current_rounds[round_id]
    
    def get_client_stats(self, client_id: str) -> Dict:
        now = time.time()
        if self.repo is not None:
            timestamps = self.repo.get_timestamps(f"req:{client_id}")
        else:
            timestamps = self.request_timestamps.get(client_id, [])
        
        one_minute_ago = now - 60
        requests_last_minute = len([ts for ts in timestamps if ts > one_minute_ago])
        one_hour_ago = now - 3600
        requests_last_hour = len([ts for ts in timestamps if ts > one_hour_ago])
        
        return {
            "requests_last_minute": requests_last_minute,
            "requests_last_hour": requests_last_hour,
            "total_rounds_with_updates": len(self.updates_per_round.get(client_id, {}))
        }
