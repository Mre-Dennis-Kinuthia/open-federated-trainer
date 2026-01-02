"""
Asynchronous Round Manager Module

Manages federated learning rounds with asynchronous completion.
Allows rounds to progress without waiting for all clients (straggler handling).
"""

import time
import threading
from typing import Dict, Set, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from .round_manager import RoundManager, RoundState, Round


class AsyncRoundState(Enum):
    """Extended round state for async operations."""
    OPEN = "OPEN"
    COLLECTING = "COLLECTING"
    READY_FOR_AGGREGATION = "READY_FOR_AGGREGATION"  # Min updates reached
    AGGREGATING = "AGGREGATING"
    CLOSED = "CLOSED"
    TIMEOUT = "TIMEOUT"  # Round timed out


@dataclass
class AsyncRoundConfig:
    """Configuration for async round behavior."""
    minimum_updates_required: int = 2
    max_round_duration_seconds: float = 300.0  # 5 minutes default
    enable_async: bool = True


@dataclass
class StragglerInfo:
    """Information about straggler clients."""
    client_id: str
    round_id: int
    arrived_after_close: bool
    timestamp: float


class AsyncRoundManager:
    """
    Manages asynchronous federated learning rounds.
    
    Extends RoundManager with async capabilities:
    - Rounds can complete with partial participation
    - Timeout-based round closure
    - Straggler tracking
    """
    
    def __init__(
        self,
        base_round_manager: RoundManager,
        config: Optional[AsyncRoundConfig] = None,
        on_round_ready: Optional[Callable[[int], None]] = None
    ):
        """
        Initialize the async round manager.
        
        Args:
            base_round_manager: Base RoundManager instance
            config: Async configuration (uses defaults if None)
            on_round_ready: Optional callback when round is ready for aggregation
        """
        self.base_round_manager = base_round_manager
        self.config = config or AsyncRoundConfig()
        self.on_round_ready = on_round_ready
        
        # Track round start times
        self.round_start_times: Dict[int, float] = {}
        
        # Track stragglers
        self.stragglers: Dict[int, list[StragglerInfo]] = {}
        
        # Track closed rounds (for straggler detection)
        self.closed_rounds: Set[int] = set()
        
        # Background thread for timeout checking
        self._timeout_thread: Optional[threading.Thread] = None
        self._stop_timeout_thread = False
        
        if self.config.enable_async:
            self._start_timeout_monitor()
    
    def _start_timeout_monitor(self) -> None:
        """Start background thread to monitor round timeouts."""
        def monitor_timeouts():
            while not self._stop_timeout_thread:
                self._check_timeouts()
                time.sleep(5.0)  # Check every 5 seconds
        
        self._timeout_thread = threading.Thread(target=monitor_timeouts, daemon=True)
        self._timeout_thread.start()
    
    def _check_timeouts(self) -> None:
        """Check for rounds that have exceeded their timeout."""
        now = time.time()
        for round_id, start_time in list(self.round_start_times.items()):
            if round_id in self.closed_rounds:
                continue
            
            elapsed = now - start_time
            if elapsed > self.config.max_round_duration_seconds:
                # Round has timed out
                round_obj = self.base_round_manager.rounds.get(round_id)
                if round_obj and round_obj.state not in [RoundState.AGGREGATING, RoundState.CLOSED]:
                    # Mark as ready for aggregation due to timeout
                    if self.on_round_ready:
                        self.on_round_ready(round_id)
    
    def start_round(self, round_id: int) -> None:
        """
        Mark the start of a round for timeout tracking.
        
        Args:
            round_id: Identifier of the round
        """
        if self.config.enable_async:
            self.round_start_times[round_id] = time.time()
    
    def check_round_ready(self, round_id: int) -> bool:
        """
        Check if a round is ready for aggregation.
        
        A round is ready if:
        - Minimum updates received, OR
        - Round has timed out
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            True if round is ready for aggregation
        """
        if not self.config.enable_async:
            # Sync mode: round is ready when all clients submit
            round_obj = self.base_round_manager.rounds.get(round_id)
            if round_obj:
                return len(round_obj.updates_received) >= len(round_obj.assigned_clients) and len(round_obj.assigned_clients) > 0
            return False
        
        round_obj = self.base_round_manager.rounds.get(round_id)
        if not round_obj:
            return False
        
        # Check if already closed
        if round_id in self.closed_rounds:
            return False
        
        # Check minimum updates
        updates_received = len(round_obj.updates_received)
        if updates_received >= self.config.minimum_updates_required:
            return True
        
        # Check timeout
        if round_id in self.round_start_times:
            elapsed = time.time() - self.round_start_times[round_id]
            if elapsed > self.config.max_round_duration_seconds:
                return True
        
        return False
    
    def record_straggler(self, client_id: str, round_id: int) -> None:
        """
        Record a straggler client (update arrived after round closed).
        
        Args:
            client_id: Identifier of the straggler client
            round_id: Identifier of the round
        """
        if round_id not in self.stragglers:
            self.stragglers[round_id] = []
        
        self.stragglers[round_id].append(StragglerInfo(
            client_id=client_id,
            round_id=round_id,
            arrived_after_close=True,
            timestamp=time.time()
        ))
    
    def get_stragglers_for_round(self, round_id: int) -> list[StragglerInfo]:
        """
        Get all stragglers for a round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            List of straggler information
        """
        return self.stragglers.get(round_id, [])
    
    def mark_round_closed(self, round_id: int) -> None:
        """
        Mark a round as closed (for straggler detection).
        
        Args:
            round_id: Identifier of the round
        """
        self.closed_rounds.add(round_id)
        # Clean up start time
        self.round_start_times.pop(round_id, None)
    
    def get_round_stats(self, round_id: int) -> Dict:
        """
        Get statistics for an async round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            Dictionary with round statistics
        """
        round_obj = self.base_round_manager.rounds.get(round_id)
        if not round_obj:
            return {}
        
        stats = {
            "round_id": round_id,
            "assigned_clients": len(round_obj.assigned_clients),
            "updates_received": len(round_obj.updates_received),
            "minimum_required": self.config.minimum_updates_required if self.config.enable_async else len(round_obj.assigned_clients),
            "is_ready": self.check_round_ready(round_id),
            "stragglers": len(self.stragglers.get(round_id, []))
        }
        
        if round_id in self.round_start_times:
            elapsed = time.time() - self.round_start_times[round_id]
            stats["elapsed_seconds"] = elapsed
            stats["timeout_seconds"] = self.config.max_round_duration_seconds
            stats["timeout_remaining"] = max(0, self.config.max_round_duration_seconds - elapsed)
        
        return stats
    
    def shutdown(self) -> None:
        """Shutdown the async round manager (stop timeout thread)."""
        self._stop_timeout_thread = True
        if self._timeout_thread:
            self._timeout_thread.join(timeout=2.0)

