"""
Aggregator Module

Collects client updates and performs federated averaging (FedAvg) over
parsed weight deltas. Pending updates are checkpointed for restart recovery.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .round_manager import RoundManager, RoundState
from .model_store import ModelStore
from .versioning import next_version
from utils.logger import get_logger

logger = get_logger("aggregator")


@dataclass
class ClientUpdate:
    """Represents a client update."""
    client_id: str
    round_id: int
    weight_delta: str


def _parse_weight_delta(raw: str) -> Optional[List[List[float]]]:
    """Extract nested weight delta lists from a JSON update payload."""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(payload, dict) and "weight_delta" in payload:
        delta = payload["weight_delta"]
    else:
        delta = payload

    if not isinstance(delta, list) or not delta:
        return None
    if not all(isinstance(layer, list) for layer in delta):
        return None
    return delta


def fedavg_weight_deltas(deltas: List[List[List[float]]]) -> List[List[float]]:
    """
    Federated averaging over a list of per-client weight deltas.

    Each client delta is a list of flattened parameter layers.
    """
    if not deltas:
        return []

    num_clients = len(deltas)
    num_layers = len(deltas[0])
    for d in deltas:
        if len(d) != num_layers:
            raise ValueError("Inconsistent number of parameter layers across clients")

    averaged: List[List[float]] = []
    for layer_idx in range(num_layers):
        layer_len = len(deltas[0][layer_idx])
        for d in deltas:
            if len(d[layer_idx]) != layer_len:
                raise ValueError(f"Inconsistent layer {layer_idx} sizes across clients")
        avg_layer = [0.0] * layer_len
        for client_delta in deltas:
            for i, v in enumerate(client_delta[layer_idx]):
                avg_layer[i] += float(v)
        avg_layer = [v / num_clients for v in avg_layer]
        averaged.append(avg_layer)
    return averaged


class Aggregator:
    """Aggregates client updates with FedAvg and optional persistence."""

    def __init__(
        self,
        round_manager: RoundManager,
        model_store: ModelStore = None,
        task_assigner=None,
        metrics_collector=None,
        rate_limiter=None,
        state_store=None,
        on_aggregated=None,
    ):
        self.round_manager = round_manager
        self.model_store = model_store or ModelStore()
        self.task_assigner = task_assigner
        self.metrics_collector = metrics_collector
        self.rate_limiter = rate_limiter
        self.state_store = state_store
        self.on_aggregated = on_aggregated
        self.updates: Dict[int, List[ClientUpdate]] = {}
        self._restore_pending()

    def _restore_pending(self) -> None:
        if not self.state_store:
            return
        pending = self.state_store.get_pending_updates()
        for round_key, items in pending.items():
            try:
                round_id = int(round_key)
            except (TypeError, ValueError):
                continue
            restored: List[ClientUpdate] = []
            for item in items:
                restored.append(
                    ClientUpdate(
                        client_id=item["client_id"],
                        round_id=round_id,
                        weight_delta=item["weight_delta"],
                    )
                )
                # Ensure round manager knows about the update if round exists
                if round_id in self.round_manager.rounds:
                    self.round_manager.add_update(item["client_id"], round_id, item["weight_delta"])
            if restored:
                self.updates[round_id] = restored

    def _persist_pending(self) -> None:
        if not self.state_store:
            return
        serializable: Dict[str, list] = {}
        for round_id, updates in self.updates.items():
            round_obj = self.round_manager.rounds.get(round_id)
            if round_obj and round_obj.state == RoundState.CLOSED:
                continue
            serializable[str(round_id)] = [
                {
                    "client_id": u.client_id,
                    "weight_delta": u.weight_delta,
                }
                for u in updates
            ]
        self.state_store.set_pending_updates(serializable)

    def submit_update(self, client_id: str, round_id: int, weight_delta: str) -> bool:
        if not self.round_manager.add_update(client_id, round_id, weight_delta):
            return False

        if round_id not in self.updates:
            self.updates[round_id] = []

        existing_update = next(
            (u for u in self.updates[round_id] if u.client_id == client_id),
            None,
        )
        if existing_update:
            existing_update.weight_delta = weight_delta
        else:
            self.updates[round_id].append(
                ClientUpdate(client_id=client_id, round_id=round_id, weight_delta=weight_delta)
            )

        self._persist_pending()
        return True

    def aggregate(self, round_id: int) -> Optional[Dict[str, Any]]:
        round_status = self.round_manager.get_round_status(round_id)
        if round_status is None:
            return None

        round_obj = self.round_manager.rounds.get(round_id)
        if round_obj is None:
            return None

        if round_obj.state == RoundState.CLOSED:
            return {
                "round_id": round_id,
                "status": "already_closed",
                "aggregated_model": None,
                "num_updates": 0,
            }

        self.round_manager.set_round_state(round_id, RoundState.AGGREGATING)

        logger.info(
            f"Aggregation started for round {round_id}",
            extra={
                "component": "coordinator",
                "event": "aggregation_started",
                "round_id": round_id,
                "model_version": round_obj.model_version,
            },
        )

        if self.metrics_collector:
            self.metrics_collector.start_aggregation(round_id)

        round_updates = self.updates.get(round_id, [])
        if not round_updates:
            self.round_manager.set_round_state(round_id, RoundState.CLOSED)
            return {
                "round_id": round_id,
                "status": "no_updates",
                "aggregated_model": None,
                "num_updates": 0,
            }

        parsed: List[List[List[float]]] = []
        valid_clients: List[str] = []
        losses: List[float] = []
        for update in round_updates:
            delta = _parse_weight_delta(update.weight_delta)
            if delta is None:
                logger.warning(
                    f"Skipping unparseable update from {update.client_id} in round {round_id}"
                )
                continue
            parsed.append(delta)
            valid_clients.append(update.client_id)
            try:
                payload = json.loads(update.weight_delta)
                if isinstance(payload, dict) and "final_loss" in payload:
                    losses.append(float(payload["final_loss"]))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        if not parsed:
            self.round_manager.set_round_state(round_id, RoundState.CLOSED)
            return {
                "round_id": round_id,
                "status": "no_valid_updates",
                "aggregated_model": None,
                "num_updates": 0,
            }

        try:
            averaged_delta = fedavg_weight_deltas(parsed)
        except ValueError as e:
            logger.error(f"FedAvg failed for round {round_id}: {e}")
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None

        round_model_version = round_obj.model_version
        new_model_version = next_version(round_model_version)

        aggregated_model_data = {
            "version": new_model_version,
            "base_version": round_model_version,
            "round_id": round_id,
            "aggregation": "fedavg",
            "averaged_weight_delta": averaged_delta,
            "num_updates": len(parsed),
            "client_ids": valid_clients,
            "mean_final_loss": (sum(losses) / len(losses)) if losses else None,
            "aggregation_timestamp": time.time(),
        }

        try:
            self.model_store.save_model(new_model_version, aggregated_model_data)
            if self.task_assigner:
                self.task_assigner.set_model_version(new_model_version)
        except Exception as e:
            logger.error(f"Failed to persist model {new_model_version}: {e}")

        self.round_manager.set_round_state(round_id, RoundState.CLOSED)

        if self.rate_limiter:
            self.rate_limiter.reset_round(round_id)

        # Drop closed round updates from pending checkpoint
        self.updates.pop(round_id, None)
        self._persist_pending()

        logger.info(
            f"Aggregation completed for round {round_id}",
            extra={
                "component": "coordinator",
                "event": "aggregation_completed",
                "round_id": round_id,
                "model_version": new_model_version,
                "num_updates": len(parsed),
            },
        )

        if self.metrics_collector:
            self.metrics_collector.complete_aggregation(round_id)
            self.metrics_collector.end_round(round_id)

        if self.on_aggregated:
            try:
                self.on_aggregated(round_id)
            except Exception as e:
                logger.error(f"on_aggregated callback failed: {e}")

        return {
            "round_id": round_id,
            "model_version": new_model_version,
            "status": "aggregated",
            "aggregated_model": aggregated_model_data,
            "num_updates": len(parsed),
        }

    def get_updates_for_round(self, round_id: int) -> List[ClientUpdate]:
        return self.updates.get(round_id, [])
