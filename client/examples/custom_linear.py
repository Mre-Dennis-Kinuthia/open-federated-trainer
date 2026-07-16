"""
Example custom Trainer plugin.

Point a client at this with:
  export MODEL_ID=custom
  export MODEL_MODULE=examples.custom_linear:CustomLinearTrainer

Or register via models.register_trainer in your own package.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from client/src
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models import Trainer, TrainResult, register_trainer  # noqa: E402


class CustomLinearTrainer(Trainer):
    """Single linear layer — minimal custom architecture example."""

    model_id = "custom_linear"

    def train(
        self,
        task: Dict[str, Any],
        dataset: Any,
        client_id: Optional[str] = None,
    ) -> TrainResult:
        cfg = task.get("model_config") or {}
        dim = int(cfg.get("input_dim", 10))
        epochs = int(cfg.get("num_epochs", 5))
        lr = float(cfg.get("learning_rate", 0.05))

        model = nn.Linear(dim, 1)
        initial = nn.Linear(dim, 1)
        initial.load_state_dict(model.state_dict())

        if dataset is not None and hasattr(dataset, "as_tensors"):
            X, y = dataset.as_tensors(kind="tabular", input_dim=dim)
        else:
            n = int(cfg.get("num_samples", 64))
            X = torch.randn(n, dim)
            y = torch.sum(X, dim=1, keepdim=True)

        opt = optim.SGD(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        loss_val = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())

        deltas = []
        for p0, p1 in zip(initial.parameters(), model.parameters()):
            deltas.append((p1.data - p0.data).detach().cpu().numpy().flatten().tolist())

        payload = {
            "client_id": client_id or "unknown",
            "round_id": task.get("round_id"),
            "model_version": task.get("model_version"),
            "model_id": self.model_id,
            "weight_delta": deltas,
            "final_loss": loss_val,
            "training_config": {"num_epochs": epochs, "num_samples": int(X.size(0))},
        }
        return TrainResult(
            weight_delta=json.dumps(payload, sort_keys=True),
            metrics={"final_loss": loss_val},
            num_samples=int(X.size(0)),
        )


register_trainer("custom_linear", CustomLinearTrainer)
