"""
Trainer module for local PyTorch-based model training.

Implements real neural network training using PyTorch, replacing the
previous fake training simulation. This module performs actual gradient
descent and returns weight deltas compatible with federated learning.
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class SimpleMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron (MLP) for federated learning.
    
    A fully connected neural network with configurable input, hidden,
    and output dimensions.
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, output_dim: int = 1):
        """
        Initialize the MLP model.
        
        Args:
            input_dim: Size of input features
            hidden_dim: Size of hidden layer
            output_dim: Size of output layer
        """
        super(SimpleMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights with small random values
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights with small random values."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def _generate_fake_data(
    num_samples: int,
    input_dim: int,
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate fake training data for local training.
    
    Creates random input-output pairs for demonstration purposes.
    In production, this would load real local data.
    
    Args:
        num_samples: Number of training samples to generate
        input_dim: Dimension of input features
        seed: Random seed for reproducibility (uses client_id hash if None)
        
    Returns:
        Tuple of (input_tensor, target_tensor)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random input features
    X = torch.randn(num_samples, input_dim)
    
    # Generate targets as a simple function of inputs (for demonstration)
    # In real scenarios, this would be actual labels
    y = torch.sum(X, dim=1, keepdim=True) + 0.1 * torch.randn(num_samples, 1)
    
    return X, y


def _model_parameters_to_list(model: nn.Module) -> List[List[float]]:
    """
    Convert PyTorch model parameters to a list of lists (JSON-serializable).
    
    Args:
        model: PyTorch model
        
    Returns:
        List of parameter tensors, each as a list of floats
    """
    params = []
    for param in model.parameters():
        # Convert tensor to numpy, then to list
        params.append(param.data.cpu().numpy().flatten().tolist())
    return params


def _compute_weight_delta(
    initial_model: nn.Module,
    trained_model: nn.Module
) -> List[List[float]]:
    """
    Compute weight delta as the difference between trained and initial model.
    
    Args:
        initial_model: Model state before training
        trained_model: Model state after training
        
    Returns:
        List of parameter deltas, each as a list of floats
    """
    deltas = []
    for initial_param, trained_param in zip(initial_model.parameters(), trained_model.parameters()):
        delta = trained_param.data - initial_param.data
        deltas.append(delta.cpu().numpy().flatten().tolist())
    return deltas


def train_local_model(
    task: Dict[str, Any],
    client_id: Optional[str] = None,
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    num_samples: int = 100,
    input_dim: int = 10,
    hidden_dim: int = 32,
    output_dim: int = 1,
    seed: Optional[int] = None
) -> str:
    """
    Perform local PyTorch-based model training and return weight delta.
    
    This function:
    1. Creates a simple MLP model
    2. Generates fake local training data
    3. Trains the model using SGD/Adam
    4. Computes weight delta (difference from initial weights)
    5. Serializes delta to JSON string for coordinator
    
    Args:
        task: Task dictionary containing round_id, model_version, task, description
        client_id: Optional client identifier for reproducibility
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size for training (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.01)
        num_samples: Number of fake training samples (default: 100)
        input_dim: Input dimension for MLP (default: 10)
        hidden_dim: Hidden layer dimension (default: 32)
        output_dim: Output dimension (default: 1)
        seed: Random seed for reproducibility (uses hash of client_id + round_id if None)
        
    Returns:
        Weight delta as JSON-serialized string
    """
    round_id = task.get("round_id", 0)
    model_version = task.get("model_version", "v1")
    
    # Set random seed for reproducibility
    if seed is None and client_id is not None:
        # Use hash of client_id and round_id for deterministic seed
        seed_str = f"{client_id}_{round_id}"
        seed = hash(seed_str) % (2**31)  # Convert to positive int
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Create model
    model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    
    # Save initial model state for delta computation
    initial_model = SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    initial_model.load_state_dict(model.state_dict())
    
    # Generate fake training data
    X_train, y_train = _generate_fake_data(num_samples, input_dim, seed)
    
    # Setup loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        # Simple batch training (for MVP, we use all data)
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # Compute weight delta
    weight_delta = _compute_weight_delta(initial_model, model)
    
    # Create serializable update object
    update_data = {
        "client_id": client_id or "unknown",
        "round_id": round_id,
        "model_version": model_version,
        "weight_delta": weight_delta,
        "model_config": {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim
        },
        "training_config": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_samples": num_samples
        },
        "final_loss": float(loss.item())
    }
    
    # Serialize to JSON string
    weight_delta_str = json.dumps(update_data, sort_keys=True)
    
    return weight_delta_str


def train_local_model_with_client_id(task: Dict[str, Any], client_id: str) -> str:
    """
    Perform local training with client ID for reproducibility.
    
    This is the main entry point called by client.py. It wraps
    train_local_model() with client_id for deterministic behavior.
    
    Args:
        task: Task dictionary containing round_id, model_version, task, description
        client_id: Identifier of the client performing training
        
    Returns:
        Weight delta as JSON-serialized string
    """
    return train_local_model(task, client_id=client_id)
