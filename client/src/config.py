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


# Global config instance
config = Config()

