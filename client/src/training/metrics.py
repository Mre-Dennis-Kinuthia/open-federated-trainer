"""
Training Metrics Module

Tracks training metrics during LoRA fine-tuning.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TrainingMetrics:
    """Training metrics for a LoRA fine-tuning session."""
    num_samples: int
    num_steps: int
    initial_loss: float
    final_loss: float
    learning_rate: float
    step_losses: Optional[List[float]] = None
    
    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "num_samples": self.num_samples,
            "num_steps": self.num_steps,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "learning_rate": self.learning_rate,
            "step_losses": self.step_losses
        }

