"""
Structured JSON Logging Utility

Provides JSON-formatted logging for the federated learning coordinator.
Logs are both human-readable and machine-parseable.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class JSONFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as JSON.
    
    Ensures all logs are structured and parseable.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": getattr(record, "component", "unknown"),
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Add event type if present
        if hasattr(record, "event"):
            log_data["event"] = record.event
        
        # Add round_id if present
        if hasattr(record, "round_id"):
            log_data["round_id"] = record.round_id
        
        # Add client_id if present
        if hasattr(record, "client_id"):
            log_data["client_id"] = record.client_id
        
        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, sort_keys=True)


def setup_coordinator_logger(
    log_dir: Optional[str] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up JSON logging for the coordinator.
    
    Args:
        log_dir: Directory for log files (defaults to coordinator/logs/)
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        # Get coordinator directory (parent of src)
        current_file = Path(__file__)
        coordinator_dir = current_file.parent.parent.parent
        log_dir = str(coordinator_dir / "logs")
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("coordinator")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # File handler (JSON logs)
    file_handler = logging.FileHandler(log_path / "coordinator.json.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger
        
    Returns:
        Logger instance configured for the coordinator
    """
    logger = logging.getLogger("coordinator")
    if not logger.handlers:
        # If logger not set up, set it up with defaults
        logger = setup_coordinator_logger()
    
    # Create a child logger for the specific module
    module_logger = logger.getChild(module_name)
    return module_logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: str = "INFO",
    round_id: Optional[int] = None,
    client_id: Optional[str] = None,
    **kwargs
) -> None:
    """
    Log a structured event.
    
    Args:
        logger: Logger instance
        event: Event type (e.g., "round_started", "update_received")
        level: Log level (default: INFO)
        round_id: Optional round identifier
        client_id: Optional client identifier
        **kwargs: Additional fields to include in log
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create a log record with extra attributes
    extra = {
        "component": "coordinator",
        "event": event,
    }
    
    if round_id is not None:
        extra["round_id"] = round_id
    
    if client_id is not None:
        extra["client_id"] = client_id
    
    if kwargs:
        extra["extra_fields"] = kwargs
    
    logger.log(log_level, f"Event: {event}", extra=extra)

