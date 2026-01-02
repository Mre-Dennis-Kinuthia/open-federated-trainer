"""
Authentication Module

Provides API key-based authentication for federated learning clients.
Simple, lightweight authentication suitable for research environments.
"""

import os
import secrets
from typing import Dict, Optional, Set


class AuthManager:
    """
    Manages API key authentication for clients.
    
    Stores client API keys in memory and validates incoming requests.
    """
    
    def __init__(self):
        """Initialize the authentication manager."""
        # Map: client_id -> api_key
        self.client_keys: Dict[str, str] = {}
        # Map: api_key -> client_id (for reverse lookup)
        self.key_to_client: Dict[str, str] = {}
        # Set of registered client IDs
        self.registered_clients: Set[str] = set()
    
    def generate_api_key(self) -> str:
        """
        Generate a new API key.
        
        Returns:
            A random API key string (32 bytes, hex-encoded)
        """
        return secrets.token_hex(16)  # 32 hex characters
    
    def register_client(self, client_id: str, api_key: Optional[str] = None) -> str:
        """
        Register a client and generate/assign an API key.
        
        Args:
            client_id: Identifier of the client
            api_key: Optional API key to assign (if None, generates one)
            
        Returns:
            The API key assigned to the client
            
        Raises:
            ValueError: If client is already registered
        """
        if client_id in self.registered_clients:
            raise ValueError(f"Client {client_id} is already registered")
        
        if api_key is None:
            api_key = self.generate_api_key()
        
        # Check for key collision (unlikely but possible)
        if api_key in self.key_to_client:
            # Regenerate if collision
            api_key = self.generate_api_key()
        
        self.client_keys[client_id] = api_key
        self.key_to_client[api_key] = client_id
        self.registered_clients.add(client_id)
        
        return api_key
    
    def validate_api_key(self, api_key: Optional[str], client_id: Optional[str] = None) -> bool:
        """
        Validate an API key.
        
        Args:
            api_key: The API key to validate
            client_id: Optional client ID to verify key belongs to this client
            
        Returns:
            True if API key is valid, False otherwise
        """
        if api_key is None or api_key == "":
            return False
        
        # Check if key exists
        if api_key not in self.key_to_client:
            return False
        
        # If client_id provided, verify it matches
        if client_id is not None:
            expected_client = self.key_to_client.get(api_key)
            if expected_client != client_id:
                return False
        
        return True
    
    def get_client_id_from_key(self, api_key: str) -> Optional[str]:
        """
        Get the client ID associated with an API key.
        
        Args:
            api_key: The API key
            
        Returns:
            Client ID if key is valid, None otherwise
        """
        return self.key_to_client.get(api_key)
    
    def revoke_client(self, client_id: str) -> bool:
        """
        Revoke a client's API key.
        
        Args:
            client_id: Identifier of the client to revoke
            
        Returns:
            True if client was revoked, False if not found
        """
        if client_id not in self.registered_clients:
            return False
        
        api_key = self.client_keys.get(client_id)
        if api_key:
            del self.key_to_client[api_key]
            del self.client_keys[client_id]
        
        self.registered_clients.discard(client_id)
        return True
    
    def is_registered(self, client_id: str) -> bool:
        """
        Check if a client is registered.
        
        Args:
            client_id: Identifier of the client
            
        Returns:
            True if client is registered, False otherwise
        """
        return client_id in self.registered_clients

