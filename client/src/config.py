"""
Configuration module for the federated learning client.

Stores all configuration values used by the client.
"""

import os
from typing import Optional


class Config:
    """Configuration class for client settings."""
    
    # Coordinator URL
    COORDINATOR_URL: str = os.getenv(
        "COORDINATOR_URL",
        "http://127.0.0.1:8000"
    )
    
    # Client name (can be overridden via environment variable)
    CLIENT_NAME: Optional[str] = os.getenv("CLIENT_NAME", None)
    
    # Retry configuration
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    
    # Sleep between rounds (in seconds)
    SLEEP_BETWEEN_ROUNDS: float = float(os.getenv("SLEEP_BETWEEN_ROUNDS", "5.0"))
    
    # Request timeout (in seconds)
    REQUEST_TIMEOUT: float = float(os.getenv("REQUEST_TIMEOUT", "30.0"))
    
    # Retry delay (in seconds)
    RETRY_DELAY: float = float(os.getenv("RETRY_DELAY", "2.0"))

    # Pluggable model architecture (simple_mlp | tiny_cnn | custom | module:Class)
    MODEL_ID: str = os.getenv("MODEL_ID", "simple_mlp")

    # Private local dataset
    DATASET_PATH: Optional[str] = os.getenv("DATASET_PATH", None)
    DATASET_FORMAT: str = os.getenv("DATASET_FORMAT", "auto")

    # General job worker: comma-separated types, or "train" only / "all"
    # Examples: "train", "inference,compute", "all"
    WORK_MODES: str = os.getenv("WORK_MODES", "train")


# Global config instance
config = Config()

