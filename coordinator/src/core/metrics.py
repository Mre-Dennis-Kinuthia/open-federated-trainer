"""
Metrics Module

Tracks federated learning metrics per round and globally.
Metrics are stored in memory and can be persisted to disk.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict


@dataclass
class RoundMetrics:
    """Metrics for a single federated learning round."""
    
    round_id: int
    model_version: str
    round_start_time: float
    round_end_time: Optional[float] = None
    
    clients_assigned: int = 0
    updates_received: int = 0
    updates_accepted: int = 0
    updates_rejected: int = 0
    
    aggregation_start_time: Optional[float] = None
    aggregation_end_time: Optional[float] = None
    
    # Computed properties
    @property
    def round_duration_seconds(self) -> Optional[float]:
        """Calculate round duration in seconds."""
        if self.round_end_time and self.round_start_time:
            return self.round_end_time - self.round_start_time
        return None
    
    @property
    def aggregation_time_seconds(self) -> Optional[float]:
        """Calculate aggregation time in seconds."""
        if self.aggregation_end_time and self.aggregation_start_time:
            return self.aggregation_end_time - self.aggregation_start_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        data = asdict(self)
        data["round_duration_seconds"] = self.round_duration_seconds
        data["aggregation_time_seconds"] = self.aggregation_time_seconds
        return data


class MetricsCollector:
    """
    Collects and manages federated learning metrics.
    
    Tracks metrics per round and maintains global statistics.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None, logs_dir: Optional[str] = None):
        """
        Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory for metrics files (defaults to coordinator/metrics/)
            logs_dir: Directory for summary logs (defaults to coordinator/logs/)
        """
        if metrics_dir is None:
            # Get coordinator directory (parent of src)
            current_file = Path(__file__)
            coordinator_dir = current_file.parent.parent.parent
            metrics_dir = str(coordinator_dir / "metrics")
            logs_dir = str(coordinator_dir / "logs")
        
        self.metrics_dir = Path(metrics_dir)
        self.logs_dir = Path(logs_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory metrics storage
        self.round_metrics: Dict[int, RoundMetrics] = {}
        self.current_round_id: Optional[int] = None
        
        # Global metrics
        self.total_clients_seen: set = set()
        self.total_failed_updates: int = 0
    
    def start_round(self, round_id: int, model_version: str) -> None:
        """
        Start tracking metrics for a new round.
        
        Args:
            round_id: Identifier of the round
            model_version: Model version used for this round
        """
        self.current_round_id = round_id
        self.round_metrics[round_id] = RoundMetrics(
            round_id=round_id,
            model_version=model_version,
            round_start_time=time.time()
        )
    
    def record_client_assigned(self, round_id: int, client_id: str) -> None:
        """
        Record that a client was assigned to a round.
        
        Args:
            round_id: Identifier of the round
            client_id: Identifier of the client
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].clients_assigned += 1
            self.total_clients_seen.add(client_id)
    
    def record_update_received(self, round_id: int) -> None:
        """
        Record that an update was received for a round.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].updates_received += 1
    
    def record_update_accepted(self, round_id: int) -> None:
        """
        Record that an update was accepted for a round.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].updates_accepted += 1
    
    def record_update_rejected(self, round_id: int) -> None:
        """
        Record that an update was rejected for a round.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].updates_rejected += 1
            self.total_failed_updates += 1
    
    def start_aggregation(self, round_id: int) -> None:
        """
        Record the start of aggregation for a round.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].aggregation_start_time = time.time()
    
    def complete_aggregation(self, round_id: int) -> None:
        """
        Record the completion of aggregation for a round.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].aggregation_end_time = time.time()
    
    def end_round(self, round_id: int) -> None:
        """
        End tracking for a round and persist metrics.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id in self.round_metrics:
            self.round_metrics[round_id].round_end_time = time.time()
            self._persist_round_metrics(round_id)
            self._append_summary_log(round_id)
    
    def _persist_round_metrics(self, round_id: int) -> None:
        """
        Persist round metrics to disk as JSON.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id not in self.round_metrics:
            return
        
        metrics = self.round_metrics[round_id]
        metrics_file = self.metrics_dir / f"round_{round_id}.json"
        
        try:
            with open(metrics_file, "w") as f:
                json.dump(metrics.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to persist metrics for round {round_id}: {e}")
    
    def _append_summary_log(self, round_id: int) -> None:
        """
        Append a human-readable summary to rounds.log.
        
        Args:
            round_id: Identifier of the round
        """
        if round_id not in self.round_metrics:
            return
        
        metrics = self.round_metrics[round_id]
        summary_file = self.logs_dir / "rounds.log"
        
        try:
            with open(summary_file, "a") as f:
                f.write(f"[{datetime.utcnow().isoformat()}Z] Round {round_id} (Model {metrics.model_version})\n")
                f.write(f"  Clients assigned: {metrics.clients_assigned}\n")
                f.write(f"  Updates received: {metrics.updates_received}\n")
                f.write(f"  Updates accepted: {metrics.updates_accepted}\n")
                f.write(f"  Updates rejected: {metrics.updates_rejected}\n")
                if metrics.round_duration_seconds:
                    f.write(f"  Round duration: {metrics.round_duration_seconds:.2f}s\n")
                if metrics.aggregation_time_seconds:
                    f.write(f"  Aggregation time: {metrics.aggregation_time_seconds:.2f}s\n")
                f.write("\n")
        except Exception as e:
            print(f"Warning: Failed to append summary for round {round_id}: {e}")
    
    def get_round_metrics(self, round_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metrics for a specific round.
        
        Args:
            round_id: Identifier of the round
            
        Returns:
            Round metrics as dictionary, or None if round not found
        """
        if round_id in self.round_metrics:
            return self.round_metrics[round_id].to_dict()
        
        # Try to load from disk if not in memory
        metrics_file = self.metrics_dir / f"round_{round_id}.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    def get_latest_round_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get metrics for the most recent round.
        
        Returns:
            Latest round metrics as dictionary, or None if no rounds exist
        """
        if not self.round_metrics:
            return None
        
        latest_round_id = max(self.round_metrics.keys())
        return self.get_round_metrics(latest_round_id)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics (rounds + global).
        
        Returns:
            Dictionary containing all metrics
        """
        return {
            "global": {
                "total_clients_seen": len(self.total_clients_seen),
                "total_failed_updates": self.total_failed_updates,
                "total_rounds": len(self.round_metrics)
            },
            "rounds": {
                round_id: metrics.to_dict()
                for round_id, metrics in self.round_metrics.items()
            }
        }

