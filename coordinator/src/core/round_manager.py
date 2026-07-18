"""
Round Manager Module

Manages federated learning rounds, tracks clients, and maintains round states.
Classic rounds are persisted via an optional RoundRepository (JSON or SQL).
"""

from enum import Enum
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, field
import os

from utils.logger import get_logger

logger = get_logger("round_manager")


def _async_rounds_enabled() -> bool:
    return os.getenv("ENABLE_ASYNC_ROUNDS", "true").lower() in {"1", "true", "yes"}


def _async_min_updates() -> int:
    try:
        return max(1, int(os.getenv("ASYNC_MIN_UPDATES", "2")))
    except ValueError:
        return 2


def _round_still_accepts_clients(round_obj: "Round") -> bool:
    """True if more clients may join this COLLECTING/OPEN round."""
    n_updates = len(round_obj.updates_received)
    n_assigned = len(round_obj.assigned_clients)
    if _async_rounds_enabled():
        # Keep the round open until async min-updates (or max duration elsewhere).
        return n_updates < _async_min_updates()
    # Sync: close join once every assigned client has submitted.
    if n_assigned == 0:
        return True
    return n_updates < n_assigned


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
    metadata: Dict[str, Any] = field(default_factory=dict)


class RoundManager:
    """
    Manages rounds and client registrations for federated learning.
    
    Tracks registered clients, rounds, and their states.
    """
    
    def __init__(self, state_store=None, round_repo=None):
        """Initialize the round manager."""
        self.state_store = state_store
        self.round_repo = round_repo
        self.clients: Set[str] = set()
        self.rounds: Dict[int, Round] = {}
        self.client_round_assignments: Dict[str, int] = {}
        self.next_round_id: int = 1
        if state_store:
            self.clients = set(state_store.get_clients())
            self.next_round_id = max(1, state_store.get_next_round_id())
        if round_repo:
            self._restore_rounds()

    def _round_to_record(self, round_obj: Round) -> Any:
        from persistence import RoundRecord

        return RoundRecord(
            round_id=round_obj.round_id,
            state=round_obj.state.value,
            model_version=round_obj.model_version,
            assigned_clients=sorted(round_obj.assigned_clients),
            updates_received=sorted(round_obj.updates_received),
            metadata=dict(round_obj.metadata or {}),
        )

    def _persist_round(self, round_obj: Round) -> None:
        if not self.round_repo:
            return
        try:
            self.round_repo.save_round(self._round_to_record(round_obj))
        except Exception as exc:
            logger.warning(
                f"Failed to persist round {round_obj.round_id}: {exc}",
                extra={"component": "coordinator", "event": "round_persist_failed"},
            )

    def _apply_record(self, rec: Any, *, crash_recover: bool = False) -> Round:
        try:
            state = RoundState(rec.state)
        except ValueError:
            state = RoundState.CLOSED
        round_obj = Round(
            round_id=rec.round_id,
            model_version=rec.model_version or "v1",
            state=state,
            assigned_clients=set(rec.assigned_clients or []),
            updates_received=set(rec.updates_received or []),
            metadata=dict(rec.metadata or {}),
        )
        # Only on boot restore: incomplete AGGREGATING → COLLECTING for reconcile.
        if (
            crash_recover
            and state == RoundState.AGGREGATING
            and not round_obj.metadata.get("published_version")
        ):
            round_obj.state = RoundState.COLLECTING
            round_obj.metadata["resume_after_crash"] = True
        self.rounds[rec.round_id] = round_obj
        self.next_round_id = max(self.next_round_id, rec.round_id + 1)
        if round_obj.state in {
            RoundState.OPEN,
            RoundState.COLLECTING,
            RoundState.AGGREGATING,
        }:
            for client_id in round_obj.assigned_clients:
                if client_id not in round_obj.updates_received:
                    self.client_round_assignments[client_id] = rec.round_id
        return round_obj

    def refresh_round(self, round_id: int) -> Optional[Round]:
        """Reload a round from the durable repository (multi-replica SoT)."""
        if not self.round_repo:
            return self.rounds.get(round_id)
        try:
            rec = self.round_repo.get_round(round_id)
        except Exception as exc:
            logger.warning(f"Failed to refresh round {round_id}: {exc}")
            return self.rounds.get(round_id)
        if rec is None:
            return self.rounds.get(round_id)
        return self._apply_record(rec)

    def refresh_all_rounds(self) -> None:
        if not self.round_repo:
            return
        try:
            records = self.round_repo.list_rounds(limit=10_000)
        except Exception as exc:
            logger.warning(f"Failed to refresh rounds: {exc}")
            return
        for rec in records:
            self._apply_record(rec)

    def try_begin_aggregating(self, round_id: int) -> bool:
        """
        Claim aggregation for this round across replicas.

        Uses SQL row lock when shared state / SQL round repo is available;
        falls back to local state transition otherwise.
        """
        from persistence.shared_state import shared_state_enabled

        if shared_state_enabled() and self.round_repo is not None:
            try:
                from persistence.ha_repos import try_transition_round_aggregating

                ok, _state = try_transition_round_aggregating(round_id)
                self.refresh_round(round_id)
                return bool(ok)
            except Exception as exc:
                logger.warning(f"Aggregate lock failed, falling back local: {exc}")

        round_obj = self.refresh_round(round_id) or self.rounds.get(round_id)
        if round_obj is None:
            return False
        if round_obj.state in (RoundState.AGGREGATING, RoundState.CLOSED):
            return round_obj.state == RoundState.AGGREGATING
        return self.set_round_state(round_id, RoundState.AGGREGATING)

    def _restore_rounds(self) -> None:
        try:
            records = self.round_repo.list_rounds(limit=10_000)
        except Exception as exc:
            logger.warning(f"Failed to restore rounds: {exc}")
            return
        for rec in records:
            self._apply_record(rec, crash_recover=True)
        if self.state_store:
            self.state_store.set_next_round_id(self.next_round_id)
        if self.rounds:
            logger.info(
                f"Restored {len(self.rounds)} classic round(s) from durable store",
                extra={"component": "coordinator", "event": "rounds_restored"},
            )
            for round_obj in self.rounds.values():
                if (round_obj.metadata or {}).get("resume_after_crash"):
                    self._persist_round(round_obj)
    def register_client(self, client_name: str) -> bool:
        """
        Register a new client.
        
        Args:
            client_name: Unique identifier for the client
            
        Returns:
            True if client was newly registered, False if already exists
        """
        if client_name in self.clients:
            logger.warning(f"Client {client_name} already registered", extra={
                "component": "coordinator",
                "event": "client_registration_failed",
                "client_id": client_name
            })
            return False
        self.clients.add(client_name)
        if self.state_store:
            # Auth store also tracks clients; keep next_round_id durable
            self.state_store.set_next_round_id(self.next_round_id)
        logger.info(f"Client {client_name} registered", extra={
            "component": "coordinator",
            "event": "client_registered",
            "client_id": client_name
        })
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
        self.refresh_all_rounds()
        if client_id not in self.clients:
            return None
        
        # Check if client is already assigned to an active round
        if client_id in self.client_round_assignments:
            assigned_round_id = self.client_round_assignments[client_id]
            assigned_round = self.rounds.get(assigned_round_id)
            if assigned_round:
                if client_id in assigned_round.updates_received:
                    # Already contributed — free them for the next open round.
                    del self.client_round_assignments[client_id]
                elif assigned_round.state in [RoundState.OPEN, RoundState.COLLECTING]:
                    if assigned_round.model_version == model_version:
                        return None
                    del self.client_round_assignments[client_id]
        
        # Find or create an active round with matching model version
        active_round = None
        for round_id, round_obj in self.rounds.items():
            if round_obj.state in [RoundState.OPEN, RoundState.COLLECTING]:
                if round_obj.model_version != model_version:
                    continue
                if not _round_still_accepts_clients(round_obj):
                    continue
                # Don't re-join a round this client already updated.
                if client_id in round_obj.updates_received:
                    continue
                active_round = round_obj
                break
        
        if active_round is None:
            # Create new round with specified model version
            active_round = Round(round_id=self.next_round_id, model_version=model_version)
            self.rounds[self.next_round_id] = active_round
            logger.info(f"Round {self.next_round_id} started", extra={
                "component": "coordinator",
                "event": "round_started",
                "round_id": self.next_round_id,
                "model_version": model_version
            })
            self.next_round_id += 1
            if self.state_store:
                self.state_store.set_next_round_id(self.next_round_id)
        
        active_round.assigned_clients.add(client_id)
        self.client_round_assignments[client_id] = active_round.round_id
        
        logger.info(f"Client {client_id} assigned to round {active_round.round_id}", extra={
            "component": "coordinator",
            "event": "client_assigned",
            "round_id": active_round.round_id,
            "client_id": client_id,
            "model_version": model_version
        })
        
        if active_round.state == RoundState.OPEN:
            active_round.state = RoundState.COLLECTING

        self._persist_round(active_round)
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
        self.refresh_round(round_id)
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
        
        logger.info(f"Update received from client {client_id} for round {round_id}", extra={
            "component": "coordinator",
            "event": "update_received",
            "round_id": round_id,
            "client_id": client_id,
            "update_size_bytes": len(weight_delta.encode('utf-8'))
        })

        self._persist_round(round_obj)
        return True
    
    def get_round_status(self, round_id: int) -> Optional[Dict]:
        """
        Get the status of a round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            Dictionary with round status information, None if round doesn't exist
        """
        round_obj = self.refresh_round(round_id) or self.rounds.get(round_id)
        if round_obj is None:
            return None
        
        return {
            "round_id": round_obj.round_id,
            "model_version": round_obj.model_version,
            "state": round_obj.state.value,
            "assigned_clients": list(round_obj.assigned_clients),
            "updates_received": list(round_obj.updates_received),
            "total_clients": len(round_obj.assigned_clients),
            "total_updates": len(round_obj.updates_received),
            "published_version": (round_obj.metadata or {}).get("published_version"),
            "metadata": dict(round_obj.metadata or {}),
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
        
        old_state = round_obj.state
        round_obj.state = state
        
        # Log round completion
        if state == RoundState.CLOSED:
            logger.info(f"Round {round_id} completed", extra={
                "component": "coordinator",
                "event": "round_completed",
                "round_id": round_id,
                "model_version": round_obj.model_version,
                "total_clients": len(round_obj.assigned_clients),
                "total_updates": len(round_obj.updates_received)
            })

        self._persist_round(round_obj)
        return True

