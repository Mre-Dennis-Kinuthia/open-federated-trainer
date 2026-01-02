"""
Security Module for Client

Handles API key management and security-related client operations.
"""

import os
from typing import Optional


class ClientSecurity:
    """
    Manages client-side security operations.
    
    Handles API key retrieval and validation.
    """
    
    def __init__(self):
        """Initialize the client security manager."""
        self.api_key: Optional[str] = self._load_api_key()
    
    def _load_api_key(self) -> Optional[str]:
        """
        Load API key from environment variable.
        
        Returns:
            API key string if found, None otherwise
        """
        api_key = os.getenv("CLIENT_API_KEY")
        if api_key:
            # Strip whitespace
            api_key = api_key.strip()
            if api_key:
                return api_key
        return None
    
    def get_api_key(self) -> Optional[str]:
        """
        Get the client's API key.
        
        Returns:
            API key string if available, None otherwise
        """
        return self.api_key
    
    def has_api_key(self) -> bool:
        """
        Check if client has an API key.
        
        Returns:
            True if API key is available, False otherwise
        """
        return self.api_key is not None
    
    def require_api_key(self) -> str:
        """
        Get API key, raising an error if not available.
        
        Returns:
            API key string
            
        Raises:
            ValueError: If API key is not set
        """
        if not self.api_key:
            raise ValueError(
                "CLIENT_API_KEY environment variable is required. "
                "Please set it before running the client."
            )
        return self.api_key


# Global security instance
_security = ClientSecurity()


def get_api_key() -> Optional[str]:
    """
    Get the client's API key.
    
    Returns:
        API key string if available, None otherwise
    """
    return _security.get_api_key()


def require_api_key() -> str:
    """
    Get API key, raising an error if not available.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not set
    """
    return _security.require_api_key()


def has_api_key() -> bool:
    """
    Check if client has an API key.
    
    Returns:
        True if API key is available, False otherwise
    """
    return _security.has_api_key()

