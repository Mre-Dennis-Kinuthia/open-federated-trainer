"""
Pluggable model trainer interface for arbitrary architectures.

Clients resolve a model_id to a Trainer implementation. Built-ins:
  - simple_mlp (default)
  - custom: load from MODEL_MODULE / entrypoint path

External packages can register via register_trainer().
"""

from __future__ import annotations

import importlib
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class TrainResult:
    """Result of a local training run."""
    weight_delta: str  # JSON-serialized update payload
    metrics: Dict[str, Any] = field(default_factory=dict)
    num_samples: int = 0


class Trainer(ABC):
    """Base class for pluggable local trainers."""

    model_id: str = "base"

    @abstractmethod
    def train(
        self,
        task: Dict[str, Any],
        dataset: Any,
        client_id: Optional[str] = None,
    ) -> TrainResult:
        """Train on a local dataset and return a serializable update."""


_REGISTRY: Dict[str, Callable[[], Trainer]] = {}


def register_trainer(model_id: str, factory: Callable[[], Trainer]) -> None:
    """Register a trainer factory under a model_id."""
    _REGISTRY[model_id] = factory


def list_trainers() -> List[str]:
    return sorted(_REGISTRY.keys())


def get_trainer(model_id: Optional[str] = None) -> Trainer:
    """
    Resolve a trainer by model_id.

    Resolution order:
      1. Explicit model_id if registered
      2. Env MODEL_ID
      3. Env MODEL_MODULE (import path to Trainer subclass)
      4. simple_mlp default
    """
    mid = (model_id or os.getenv("MODEL_ID") or "simple_mlp").strip()

    if mid in _REGISTRY:
        return _REGISTRY[mid]()

    # Dynamic import: MODEL_MODULE=package.module:ClassName
    module_spec = os.getenv("MODEL_MODULE", "").strip()
    if mid == "custom" or module_spec:
        spec = module_spec or mid
        return _load_external_trainer(spec)

    # Fallback: treat model_id as module:Class
    if ":" in mid or "." in mid:
        try:
            return _load_external_trainer(mid)
        except Exception:
            pass

    if "simple_mlp" in _REGISTRY:
        return _REGISTRY["simple_mlp"]()
    raise KeyError(f"Unknown model_id '{mid}'. Registered: {list_trainers()}")


def _load_external_trainer(spec: str) -> Trainer:
    """Load Trainer from 'module.path:ClassName' or 'module.path' (Class=Trainer)."""
    if ":" in spec:
        module_path, class_name = spec.split(":", 1)
    else:
        module_path, class_name = spec, "Trainer"
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    instance = cls()
    if not isinstance(instance, Trainer):
        raise TypeError(f"{spec} is not a Trainer subclass")
    return instance


# --- Built-in: SimpleMLP (wraps existing trainer.py) ---

class SimpleMLPTrainer(Trainer):
    model_id = "simple_mlp"

    def train(
        self,
        task: Dict[str, Any],
        dataset: Any,
        client_id: Optional[str] = None,
    ) -> TrainResult:
        from trainer import train_local_model

        cfg = task.get("model_config") or {}
        input_dim = int(cfg.get("input_dim", 10))
        if dataset is None or not hasattr(dataset, "as_tensors"):
            raise ValueError("simple_mlp requires a configured local dataset")
        data = dataset.as_tensors(kind="tabular", input_dim=input_dim)

        weight_delta = train_local_model(
            task,
            client_id=client_id,
            num_epochs=int(cfg.get("num_epochs", 3)),
            batch_size=int(cfg.get("batch_size", 32)),
            learning_rate=float(cfg.get("learning_rate", 0.01)),
            num_samples=int(cfg.get("num_samples", getattr(dataset, "num_samples", 100) or 100)),
            input_dim=input_dim,
            hidden_dim=int(cfg.get("hidden_dim", 32)),
            output_dim=int(cfg.get("output_dim", 1)),
            data=data,
            global_weights=task.get("global_weights"),
        )
        try:
            payload = json.loads(weight_delta)
            loss = payload.get("final_loss")
            n = payload.get("training_config", {}).get("num_samples", 0)
        except (json.JSONDecodeError, TypeError):
            loss, n = None, 0
        return TrainResult(
            weight_delta=weight_delta,
            metrics={"final_loss": loss},
            num_samples=int(n or 0),
        )


register_trainer("simple_mlp", SimpleMLPTrainer)


# Example custom architecture users can copy

class TinyCNNTrainer(Trainer):
    """
    Minimal CNN trainer for class-folder image datasets.
    Demonstrates a second built-in architecture beyond MLP.
    """

    model_id = "tiny_cnn"

    def train(
        self,
        task: Dict[str, Any],
        dataset: Any,
        client_id: Optional[str] = None,
    ) -> TrainResult:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from trainer import _load_model_parameters, _model_parameters_to_list

        cfg = task.get("model_config") or {}
        channels = int(cfg.get("channels", 1))
        size = int(cfg.get("image_size", 8))
        num_classes = int(cfg.get("num_classes", 2))
        epochs = int(cfg.get("num_epochs", 2))
        lr = float(cfg.get("learning_rate", 0.01))

        class TinyCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(channels, 8, 3, padding=1)
                self.fc = nn.Linear(8 * size * size, num_classes)

            def forward(self, x):
                x = torch.relu(self.conv(x))
                x = x.view(x.size(0), -1)
                return self.fc(x)

        torch.manual_seed(int(task.get("model_seed", 0)))
        model = TinyCNN()
        if task.get("global_weights") is not None:
            _load_model_parameters(model, task["global_weights"])
        initial = TinyCNN()
        initial.load_state_dict(model.state_dict())
        base_weights = _model_parameters_to_list(initial)

        if dataset is None or not hasattr(dataset, "as_tensors"):
            raise ValueError("tiny_cnn requires a configured image dataset")
        X, y = dataset.as_tensors(kind="image", channels=channels, size=size)
        if int(y.min()) < 0 or int(y.max()) >= num_classes:
            raise ValueError(
                f"Image labels must be in [0, {num_classes - 1}]; "
                f"found [{int(y.min())}, {int(y.max())}]"
            )

        opt = optim.Adam(model.parameters(), lr=lr)
        crit = nn.CrossEntropyLoss()
        model.train()
        loss_val = 0.0
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_val = float(loss.item())

        deltas = []
        for p0, p1 in zip(initial.parameters(), model.parameters()):
            deltas.append((p1.data - p0.data).cpu().numpy().flatten().tolist())

        payload = {
            "client_id": client_id or "unknown",
            "round_id": task.get("round_id"),
            "model_version": task.get("model_version"),
            "model_id": self.model_id,
            "model_config": cfg,
            "base_weights": base_weights,
            "weight_delta": deltas,
            "final_loss": loss_val,
            "training_config": {"num_epochs": epochs, "num_samples": int(X.size(0))},
        }
        return TrainResult(
            weight_delta=json.dumps(payload, sort_keys=True),
            metrics={"final_loss": loss_val},
            num_samples=int(X.size(0)),
        )


register_trainer("tiny_cnn", TinyCNNTrainer)
