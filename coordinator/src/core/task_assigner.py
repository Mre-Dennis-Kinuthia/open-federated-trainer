"""
Task Assigner Module

Assigns training tasks to registered clients and prevents duplicate assignments.
Supports pluggable model_id + model_config for arbitrary architectures.
"""

import os
from typing import Dict, Optional, Any
from .round_manager import RoundManager
from .versioning import initial_version, next_version
from .model_store import ModelStore
from utils.logger import get_logger

logger = get_logger("task_assigner")


class TaskAssigner:
    """
    Assigns tasks to clients for federated learning rounds.
    
    Prevents duplicate task assignments and tracks model versions.
    """
    
    def __init__(self, round_manager: RoundManager, model_store: ModelStore = None):
        self.round_manager = round_manager
        self.model_store = model_store or ModelStore()
        self.model_id: str = os.getenv("DEFAULT_MODEL_ID", "simple_mlp")
        self.model_config: Dict[str, Any] = {}
        self.model_version: str = self._version_for_model(self.model_id, None)
        if self.model_store.model_exists(self.model_version):
            saved = self.model_store.load_model(self.model_version)
            self.model_config = saved.get("model_config") or {}
        self.client_assignments: Dict[str, Dict] = {}
    
    def set_model(self, model_id: str, model_config: Optional[Dict[str, Any]] = None) -> None:
        """Set active architecture for new classic FL assignments."""
        self.model_id = model_id
        self.model_version = self._version_for_model(model_id, model_config)
        if model_config is None and self.model_store.model_exists(self.model_version):
            saved = self.model_store.load_model(self.model_version)
            self.model_config = saved.get("model_config") or {}
        else:
            self.model_config = model_config or {}

    def _version_for_model(
        self,
        model_id: str,
        model_config: Optional[Dict[str, Any]],
    ) -> str:
        matching = self.model_store.latest_model_version(
            model_id=model_id,
            model_config=model_config,
            require_weights=True,
        )
        if matching:
            return matching
        latest_global = self.model_store.latest_model_version()
        return next_version(latest_global) if latest_global else initial_version()
    
    def assign_task(self, client_id: str) -> Optional[Dict]:
        if client_id not in self.round_manager.clients:
            return None
        
        if client_id in self.client_assignments:
            existing_assignment = self.client_assignments[client_id]
            round_id = existing_assignment.get("round_id")
            round_status = self.round_manager.get_round_status(round_id)
            
            if round_status and round_status["state"] in ["OPEN", "COLLECTING"]:
                if round_status["total_updates"] < round_status["total_clients"]:
                    return existing_assignment
                del self.client_assignments[client_id]
        
        round_id = self.round_manager.assign_client_to_round(client_id, self.model_version)
        if round_id is None:
            return None
        
        task = {
            "round_id": round_id,
            "model_version": self.model_version,
            "model_id": self.model_id,
            "model_config": self.model_config,
            "task": "train",
            "description": (
                f"Train {self.model_id} model version {self.model_version} "
                f"for round {round_id}"
            ),
        }
        
        self.client_assignments[client_id] = task
        return task
    
    def get_client_assignment(self, client_id: str) -> Optional[Dict]:
        return self.client_assignments.get(client_id)
    
    def set_model_version(self, version: str) -> None:
        self.model_version = version
    
    def get_model_version(self) -> str:
        return self.model_version
