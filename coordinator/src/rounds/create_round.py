"""
Create LoRA Training Round

Creates a new federated learning round for LoRA fine-tuning.
"""

import json
import os
import threading
from typing import Dict, Optional
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("rounds")


@dataclass
class LoRARoundConfig:
    """Configuration for a LoRA fine-tuning round."""
    round_id: int
    base_model_id: str  # e.g., "tiny-llama"
    adapter_version: Optional[str] = None  # Previous adapter version to start from
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA alpha
    lora_dropout: float = 0.1  # LoRA dropout
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])  # Modules to apply LoRA
    max_steps: int = 100  # Maximum training steps per client
    learning_rate: float = 2e-4  # Learning rate
    batch_size: int = 4  # Batch size
    gradient_accumulation_steps: int = 4  # Gradient accumulation
    warmup_steps: int = 10  # Warmup steps
    max_seq_length: int = 512  # Maximum sequence length
    task_type: str = "causal_lm"  # causal_lm | seq_cls
    published_version: Optional[str] = None
    aggregation_strategy: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # OPEN → COLLECTING → AGGREGATING → EVALUATING → CLOSED | REJECTED
    state: str = "OPEN"


class LoRARoundManager:
    """
    Manages LoRA fine-tuning rounds.
    
    Extends the base round manager with LoRA-specific configuration.
    """
    
    def __init__(self, state_path: Optional[str] = None):
        """Initialize the LoRA round manager."""
        self.rounds: Dict[int, LoRARoundConfig] = {}
        self.next_round_id: int = 1
        self.adapter_submissions: Dict[int, Dict[str, Dict]] = {}  # round_id -> client_id -> submission
        default_path = Path(__file__).resolve().parents[2] / "data" / "lora_rounds.json"
        self.state_path = Path(
            state_path or os.getenv("LORA_STATE_PATH", str(default_path))
        )
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.next_round_id = int(raw.get("next_round_id", 1))
            self.rounds = {
                int(round_id): LoRARoundConfig(**config)
                for round_id, config in raw.get("rounds", {}).items()
            }
            self.adapter_submissions = {
                int(round_id): submissions
                for round_id, submissions in raw.get("adapter_submissions", {}).items()
            }
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot load LoRA state {self.state_path}: {exc}"
            ) from exc

    def _persist(self) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "next_round_id": self.next_round_id,
                "rounds": {
                    str(round_id): asdict(config)
                    for round_id, config in self.rounds.items()
                },
                "adapter_submissions": {
                    str(round_id): submissions
                    for round_id, submissions in self.adapter_submissions.items()
                },
            }
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.state_path)
    
    def create_round(
        self,
        base_model_id: str,
        adapter_version: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[list[str]] = None,
        max_steps: int = 100,
        learning_rate: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 10,
        max_seq_length: int = 512,
        task_type: str = "causal_lm",
    ) -> LoRARoundConfig:
        """
        Create a new LoRA fine-tuning round.
        
        Args:
            base_model_id: Base model identifier from registry
            adapter_version: Previous adapter version (None for first round)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: List of modules to apply LoRA to
            max_steps: Maximum training steps per client
            learning_rate: Learning rate for training
            batch_size: Batch size for training
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            max_seq_length: Maximum sequence length
            task_type: Evaluation/task family (causal_lm or seq_cls)
            
        Returns:
            LoRARoundConfig for the created round
        """
        round_id = self.next_round_id
        self.next_round_id += 1
        
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        config = LoRARoundConfig(
            round_id=round_id,
            base_model_id=base_model_id,
            adapter_version=adapter_version,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_seq_length=max_seq_length,
            task_type=task_type or "causal_lm",
        )
        
        self.rounds[round_id] = config
        self.adapter_submissions[round_id] = {}
        self._persist()
        
        logger.info(f"Created LoRA round {round_id} for base model {base_model_id}", extra={
            "component": "rounds",
            "event": "lora_round_created",
            "round_id": round_id,
            "base_model_id": base_model_id
        })
        
        return config
    
    def get_round(self, round_id: int) -> Optional[LoRARoundConfig]:
        """
        Get round configuration.
        
        Args:
            round_id: Round identifier
            
        Returns:
            LoRARoundConfig if found, None otherwise
        """
        return self.rounds.get(round_id)

    def list_rounds(self, limit: int = 25) -> list[Dict]:
        """
        List recent LoRA rounds with submission counts.

        Args:
            limit: Maximum number of rounds to return (most recent first)

        Returns:
            List of round summary dictionaries
        """
        sorted_ids = sorted(self.rounds.keys(), reverse=True)[:limit]
        summaries: list[Dict] = []
        for round_id in sorted_ids:
            config = self.rounds[round_id]
            submissions = self.adapter_submissions.get(round_id, {})
            summaries.append({
                "round_id": config.round_id,
                "base_model_id": config.base_model_id,
                "adapter_version": config.adapter_version,
                "lora_r": config.lora_r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "target_modules": config.target_modules,
                "max_steps": config.max_steps,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "state": config.state,
                "created_at": config.created_at,
                "submission_count": len(submissions),
                "submitters": list(submissions.keys()),
            })
        return summaries
    
    def submit_adapter(
        self,
        round_id: int,
        client_id: str,
        adapter_state_dict: Dict,
        num_samples: int,
        training_loss: float,
        adapter_hash: str
    ) -> bool:
        """
        Submit an adapter for a round.
        
        Args:
            round_id: Round identifier
            client_id: Client identifier
            adapter_state_dict: LoRA adapter state dict
            num_samples: Number of training samples used
            training_loss: Final training loss
            adapter_hash: Hash of adapter for verification
            
        Returns:
            True if submitted successfully, False otherwise
        """
        if round_id not in self.rounds:
            logger.warning(f"Round {round_id} not found")
            return False
        if self.rounds[round_id].state not in {"OPEN", "COLLECTING"}:
            logger.warning(
                f"Round {round_id} does not accept submissions in "
                f"state {self.rounds[round_id].state}"
            )
            return False
        
        if round_id not in self.adapter_submissions:
            self.adapter_submissions[round_id] = {}
        
        self.adapter_submissions[round_id][client_id] = {
            "adapter_state_dict": adapter_state_dict,
            "num_samples": num_samples,
            "training_loss": training_loss,
            "adapter_hash": adapter_hash,
            "submitted_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Update round state
        if self.rounds[round_id].state == "OPEN":
            self.rounds[round_id].state = "COLLECTING"
        self._persist()
        
        logger.info(f"Adapter submitted for round {round_id} by client {client_id}", extra={
            "component": "rounds",
            "event": "adapter_submitted",
            "round_id": round_id,
            "client_id": client_id,
            "num_samples": num_samples
        })
        
        return True
    
    def get_submissions(self, round_id: int) -> Dict[str, Dict]:
        """
        Get all adapter submissions for a round.
        
        Args:
            round_id: Round identifier
            
        Returns:
            Dictionary mapping client_id to submission data
        """
        return self.adapter_submissions.get(round_id, {})
    
    def close_round(self, round_id: int) -> bool:
        """
        Close a round.
        
        Args:
            round_id: Round identifier
            
        Returns:
            True if closed successfully, False otherwise
        """
        if round_id not in self.rounds:
            return False
        
        self.rounds[round_id].state = "CLOSED"
        self._persist()
        logger.info(f"Closed LoRA round {round_id}", extra={
            "component": "rounds",
            "event": "lora_round_closed",
            "round_id": round_id
        })
        return True

    def set_state(self, round_id: int, state: str) -> bool:
        if round_id not in self.rounds:
            return False
        if state not in {
            "OPEN",
            "COLLECTING",
            "AGGREGATING",
            "EVALUATING",
            "CLOSED",
            "REJECTED",
        }:
            raise ValueError(f"Invalid LoRA round state: {state}")
        self.rounds[round_id].state = state
        self._persist()
        return True


# Global instance
_lora_round_manager: Optional[LoRARoundManager] = None


def get_lora_round_manager() -> LoRARoundManager:
    """Get or create the global LoRA round manager."""
    global _lora_round_manager
    if _lora_round_manager is None:
        _lora_round_manager = LoRARoundManager()
    return _lora_round_manager


def create_lora_round(
    base_model_id: str,
    adapter_version: Optional[str] = None,
    **kwargs
) -> LoRARoundConfig:
    """
    Create a new LoRA fine-tuning round.
    
    Args:
        base_model_id: Base model identifier
        adapter_version: Previous adapter version (optional)
        **kwargs: Additional LoRA configuration parameters
        
    Returns:
        LoRARoundConfig for the created round
    """
    manager = get_lora_round_manager()
    return manager.create_round(base_model_id, adapter_version, **kwargs)

