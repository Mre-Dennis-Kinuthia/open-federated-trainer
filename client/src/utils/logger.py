"""
Structured JSON Logging Utility for Clients

Provides JSON-formatted logging for federated learning clients.
Logs are both human-readable and machine-parseable.
"""

import json
import logging
import sys
from datetime import datetime
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
            "component": getattr(record, "component", "client"),
            "module": record.module,
            "message": record.getMessage(),
        }
        
        # Add event type if present
        if hasattr(record, "event"):
            log_data["event"] = record.event
        
        # Add client_id if present
        if hasattr(record, "client_id"):
            log_data["client_id"] = record.client_id
        
        # Add round_id if present
        if hasattr(record, "round_id"):
            log_data["round_id"] = record.round_id
        
        # Add any extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, sort_keys=True)


def setup_client_logger(log_level: str = "INFO") -> logging.Logger:
    """
    Set up JSON logging for the client.
    
    Args:
        log_level: Logging level (default: INFO)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("client")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler (stdout) - JSON formatted
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module requesting the logger
        
    Returns:
        Logger instance configured for the client
    """
    logger = logging.getLogger("client")
    if not logger.handlers:
        # If logger not set up, set it up with defaults
        logger = setup_client_logger()
    
    # Create a child logger for the specific module
    module_logger = logger.getChild(module_name)
    return module_logger


def log_event(
    logger: logging.Logger,
    event: str,
    level: str = "INFO",
    client_id: Optional[str] = None,
    round_id: Optional[int] = None,
    **kwargs
) -> None:
    """
    Log a structured event.
    
    Args:
        logger: Logger instance
        event: Event type (e.g., "client_started", "training_completed")
        level: Log level (default: INFO)
        client_id: Optional client identifier
        round_id: Optional round identifier
        **kwargs: Additional fields to include in log
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create a log record with extra attributes
    extra = {
        "component": "client",
        "event": event,
    }
    
    if client_id is not None:
        extra["client_id"] = client_id
    
    if round_id is not None:
        extra["round_id"] = round_id
    
    if kwargs:
        extra["extra_fields"] = kwargs
    
    logger.log(log_level, f"Event: {event}", extra=extra)

