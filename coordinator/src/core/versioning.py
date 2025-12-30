"""
Versioning Module

Centralizes model version logic for federated learning.
"""

from typing import Optional
import re


def initial_version() -> str:
    """
    Get the initial model version.
    
    Returns:
        Initial version string: "v1"
    """
    return "v1"


def next_version(current_version: str) -> str:
    """
    Get the next sequential model version.
    
    Args:
        current_version: Current version string (e.g., "v1", "v2")
        
    Returns:
        Next version string (e.g., "v2", "v3")
        
    Raises:
        ValueError: If current_version is not in the expected format
    """
    # Validate format: must be "v" followed by digits
    pattern = r"^v(\d+)$"
    match = re.match(pattern, current_version)
    
    if not match:
        raise ValueError(
            f"Invalid version format: {current_version}. Expected format: v1, v2, v3, ..."
        )
    
    # Extract version number
    version_num = int(match.group(1))
    
    # Increment and return
    next_version_num = version_num + 1
    return f"v{next_version_num}"


def parse_version_number(version: str) -> Optional[int]:
    """
    Parse the numeric part from a version string.
    
    Args:
        version: Version string (e.g., "v1", "v2")
        
    Returns:
        Version number as integer, or None if invalid format
    """
    pattern = r"^v(\d+)$"
    match = re.match(pattern, version)
    
    if not match:
        return None
    
    return int(match.group(1))


def is_valid_version(version: str) -> bool:
    """
    Check if a version string is in the correct format.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid, False otherwise
    """
    pattern = r"^v\d+$"
    return bool(re.match(pattern, version))

