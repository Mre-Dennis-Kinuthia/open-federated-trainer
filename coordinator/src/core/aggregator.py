"""
Aggregator Module

Collects client updates and runs a pluggable aggregation strategy (default FedAvg).
Pending updates are checkpointed for restart recovery. Aggregate is idempotent:
re-invoking a CLOSED round returns the previously published model.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from aggregation.strategies import ClientContribution, get_strategy
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

    Kept as a stable helper for tests and callers; delegates to FedAvgStrategy.
    """
    from aggregation.strategies import FedAvgStrategy

    contributions = [
        ClientContribution(client_id=str(i), weight_delta=d) for i, d in enumerate(deltas)
    ]
    return FedAvgStrategy().aggregate(contributions).averaged_delta


def apply_weight_delta(
    base_weights: List[List[float]],
    delta: List[List[float]],
) -> List[List[float]]:
    """Apply a flattened parameter delta to a global model."""
    if len(base_weights) != len(delta):
        raise ValueError("Base model and delta have different layer counts")
    updated: List[List[float]] = []
    for layer_index, (base_layer, delta_layer) in enumerate(
        zip(base_weights, delta)
    ):
        if len(base_layer) != len(delta_layer):
            raise ValueError(
                f"Base model and delta differ at layer {layer_index}"
            )
        updated.append(
            [float(base) + float(change) for base, change in zip(base_layer, delta_layer)]
        )
    return updated


class Aggregator:
    """Aggregates client updates with a selectable strategy and durable pending state."""

    def __init__(
        self,
        round_manager: RoundManager,
        model_store: ModelStore = None,
        task_assigner=None,
        metrics_collector=None,
        rate_limiter=None,
        state_store=None,
        on_aggregated=None,
        strategy=None,
        strategy_name: Optional[str] = None,
    ):
        self.round_manager = round_manager
        self.model_store = model_store or ModelStore()
        self.task_assigner = task_assigner
        self.metrics_collector = metrics_collector
        self.rate_limiter = rate_limiter
        self.state_store = state_store
        self.on_aggregated = on_aggregated
        self.strategy = strategy or get_strategy(strategy_name)
        self.updates: Dict[int, List[ClientUpdate]] = {}
        self._aggregate_lock = threading.Lock()
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
                try:
                    payload = json.loads(item["weight_delta"])
                except (KeyError, TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(payload, dict) or not payload.get("base_weights"):
                    logger.warning(
                        f"Dropping legacy pending update for round {round_id}; "
                        "it has no shared base model"
                    )
                    continue
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
        self._persist_pending()

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

    def _already_closed_result(self, round_id: int, round_obj) -> Dict[str, Any]:
        published = (round_obj.metadata or {}).get("published_version")
        model = None
        if published:
            try:
                model = self.model_store.load_model(published)
            except (FileNotFoundError, ValueError, OSError):
                model = None
        return {
            "round_id": round_id,
            "model_version": published,
            "status": "already_closed",
            "aggregated_model": model,
            "num_updates": len(round_obj.updates_received),
            "replayed": True,
        }

    def reconcile_after_restart(self) -> List[Dict[str, Any]]:
        """
        Finish rounds left mid-flight after a coordinator restart.

        Rounds marked ``resume_after_crash`` (was AGGREGATING) or COLLECTING with
        pending updates that look complete are re-aggregated once.
        """
        results: List[Dict[str, Any]] = []
        for round_id, round_obj in list(self.round_manager.rounds.items()):
            if round_obj.state == RoundState.CLOSED:
                continue
            meta = round_obj.metadata or {}
            pending = self.updates.get(round_id) or []
            should = bool(meta.get("resume_after_crash")) or (
                round_obj.state == RoundState.COLLECTING
                and pending
                and len(round_obj.updates_received) >= len(round_obj.assigned_clients)
                and len(round_obj.assigned_clients) > 0
            )
            if not should:
                continue
            logger.info(
                f"Reconciling round {round_id} after restart",
                extra={
                    "component": "coordinator",
                    "event": "round_reconcile",
                    "round_id": round_id,
                },
            )
            result = self.aggregate(round_id)
            if result:
                results.append(result)
        return results

    def aggregate(self, round_id: int) -> Optional[Dict[str, Any]]:
        with self._aggregate_lock:
            return self._aggregate_unlocked(round_id)

    def _aggregate_unlocked(self, round_id: int) -> Optional[Dict[str, Any]]:
        round_status = self.round_manager.get_round_status(round_id)
        if round_status is None:
            return None

        round_obj = self.round_manager.rounds.get(round_id)
        if round_obj is None:
            return None

        if round_obj.state == RoundState.CLOSED:
            return self._already_closed_result(round_id, round_obj)

        # Idempotency: published version already recorded but state not closed yet
        published = (round_obj.metadata or {}).get("published_version")
        if published and self.model_store.model_exists(published):
            round_obj.metadata["resume_after_crash"] = False
            self.round_manager.set_round_state(round_id, RoundState.CLOSED)
            self.updates.pop(round_id, None)
            self._persist_pending()
            return self._already_closed_result(round_id, round_obj)

        if not self.round_manager.try_begin_aggregating(round_id):
            self.round_manager.refresh_round(round_id)
            round_obj = self.round_manager.rounds.get(round_id)
            if round_obj is None:
                return None
            if round_obj.state == RoundState.CLOSED:
                return self._already_closed_result(round_id, round_obj)
            published2 = (round_obj.metadata or {}).get("published_version")
            if published2 and self.model_store.model_exists(published2):
                return self._already_closed_result(round_id, round_obj)
            return {
                "round_id": round_id,
                "status": "aggregating",
                "aggregated_model": None,
                "num_updates": len(round_obj.updates_received),
                "model_version": round_obj.model_version,
            }

        round_obj = self.round_manager.rounds.get(round_id)
        if round_obj is None:
            return None

        logger.info(
            f"Aggregation started for round {round_id}",
            extra={
                "component": "coordinator",
                "event": "aggregation_started",
                "round_id": round_id,
                "model_version": round_obj.model_version,
                "strategy": getattr(self.strategy, "name", "unknown"),
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

        contributions: List[ClientContribution] = []
        base_models: List[List[List[float]]] = []
        model_ids: List[str] = []
        model_configs: List[Dict[str, Any]] = []
        valid_clients: List[str] = []
        losses: List[float] = []
        for update in round_updates:
            delta = _parse_weight_delta(update.weight_delta)
            if delta is None:
                logger.warning(
                    f"Skipping unparseable update from {update.client_id} in round {round_id}"
                )
                continue
            try:
                payload = json.loads(update.weight_delta)
                if not isinstance(payload, dict):
                    raise ValueError("Update payload must be an object")
                base_weights = payload.get("base_weights")
                if (
                    not isinstance(base_weights, list)
                    or not base_weights
                    or not all(isinstance(layer, list) for layer in base_weights)
                ):
                    raise ValueError("Update is missing base_weights")
                apply_weight_delta(base_weights, delta)  # shape validation
                num_samples = payload.get("num_samples", 1)
                try:
                    num_samples_f = float(num_samples)
                except (TypeError, ValueError):
                    num_samples_f = 1.0
                contributions.append(
                    ClientContribution(
                        client_id=update.client_id,
                        weight_delta=delta,
                        num_samples=num_samples_f,
                    )
                )
                base_models.append(base_weights)
                model_ids.append(str(payload.get("model_id", "simple_mlp")))
                model_configs.append(payload.get("model_config") or {})
                valid_clients.append(update.client_id)
                if "final_loss" in payload:
                    try:
                        losses.append(float(payload["final_loss"]))
                    except (TypeError, ValueError):
                        pass
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                logger.warning(
                    f"Skipping invalid update from {update.client_id}: {exc}"
                )

        if not contributions:
            self.round_manager.set_round_state(round_id, RoundState.CLOSED)
            return {
                "round_id": round_id,
                "status": "no_valid_updates",
                "aggregated_model": None,
                "num_updates": 0,
            }

        canonical_base = json.dumps(base_models[0], separators=(",", ":"))
        if any(
            json.dumps(base, separators=(",", ":")) != canonical_base
            for base in base_models[1:]
        ):
            logger.error("Clients did not train from identical global weights")
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None
        if len(set(model_ids)) != 1:
            logger.error("Clients submitted updates for different model architectures")
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None
        canonical_config = json.dumps(
            model_configs[0],
            sort_keys=True,
            separators=(",", ":"),
        )
        if any(
            json.dumps(config, sort_keys=True, separators=(",", ":"))
            != canonical_config
            for config in model_configs[1:]
        ):
            logger.error("Clients submitted incompatible model configurations")
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None

        try:
            strategy_result = self.strategy.aggregate(contributions)
            averaged_delta = strategy_result.averaged_delta
            global_weights = apply_weight_delta(base_models[0], averaged_delta)
        except ValueError as e:
            logger.error(f"Strategy {self.strategy.name} failed for round {round_id}: {e}")
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None

        # Stable version: prefer metadata reservation if set mid-flight
        reserved = (round_obj.metadata or {}).get("reserved_version")
        round_model_version = round_obj.model_version
        if reserved:
            new_model_version = reserved
        elif self.model_store.model_exists(round_model_version):
            latest_global = self.model_store.latest_model_version()
            new_model_version = next_version(latest_global or round_model_version)
        else:
            new_model_version = round_model_version

        # Reserve version before write so a crash+retry does not mint a new one
        round_obj.metadata["reserved_version"] = new_model_version
        round_obj.metadata["aggregation_strategy"] = strategy_result.strategy_name
        self.round_manager._persist_round(round_obj)

        aggregated_model_data = {
            "version": new_model_version,
            "base_version": (
                round_model_version
                if self.model_store.model_exists(round_model_version)
                else None
            ),
            "model_id": model_ids[0],
            "model_config": model_configs[0],
            "weights": global_weights,
            "round_id": round_id,
            "aggregation": strategy_result.strategy_name,
            "aggregation_details": strategy_result.details,
            "averaged_weight_delta": averaged_delta,
            "num_updates": len(contributions),
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
            self.round_manager.set_round_state(round_id, RoundState.COLLECTING)
            return None

        round_obj.metadata["published_version"] = new_model_version
        round_obj.metadata.pop("resume_after_crash", None)
        self.round_manager.set_round_state(round_id, RoundState.CLOSED)

        if self.rate_limiter:
            self.rate_limiter.reset_round(round_id)

        self.updates.pop(round_id, None)
        self._persist_pending()

        logger.info(
            f"Aggregation completed for round {round_id}",
            extra={
                "component": "coordinator",
                "event": "aggregation_completed",
                "round_id": round_id,
                "model_version": new_model_version,
                "num_updates": len(contributions),
                "strategy": strategy_result.strategy_name,
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
            "num_updates": len(contributions),
            "replayed": False,
            "strategy": strategy_result.strategy_name,
        }

    def get_updates_for_round(self, round_id: int) -> List[ClientUpdate]:
        return self.updates.get(round_id, [])
