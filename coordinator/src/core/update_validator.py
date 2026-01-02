"""
Update Validator Module

Validates client updates before aggregation.
Includes security checks: authentication, rate limiting, and value validation.
"""

import json
import math
from typing import Optional, Tuple
from .round_manager import RoundManager
from .auth import AuthManager
from .rate_limiter import RateLimiter
from .privacy import PrivacyProtector
from utils.logger import get_logger

logger = get_logger("update_validator")


class UpdateValidator:
    """
    Validates client updates for federated learning.
    
    Performs security and validation checks on weight deltas.
    """
    
    def __init__(
        self,
        round_manager: RoundManager,
        auth_manager: Optional[AuthManager] = None,
        rate_limiter: Optional[RateLimiter] = None,
        privacy_protector: Optional[PrivacyProtector] = None
    ):
        """
        Initialize the update validator.
        
        Args:
            round_manager: Round manager instance to coordinate with
            auth_manager: Optional authentication manager
            rate_limiter: Optional rate limiter
            privacy_protector: Optional privacy protector
        """
        self.round_manager = round_manager
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.privacy_protector = privacy_protector or PrivacyProtector()
    
    def validate(
        self,
        client_id: str,
        round_id: int,
        weight_delta: str,
        api_key: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a client update with security checks.
        
        Args:
            client_id: Identifier of the client submitting the update
            round_id: Identifier of the round
            weight_delta: The weight delta update (as string in MVP)
            api_key: Optional API key for authentication
            
        Returns:
            Tuple of (is_valid: bool, reason: Optional[str])
        """
        # 1. Authentication check
        if self.auth_manager:
            if not self.auth_manager.validate_api_key(api_key, client_id):
                reason = "authentication_failed"
                logger.warning(f"Update rejected: authentication failed", extra={
                    "component": "coordinator",
                    "event": "update_rejected",
                    "round_id": round_id,
                    "client_id": client_id,
                    "reason": reason
                })
                return False, reason
        
        # 2. Check if client is registered
        if client_id not in self.round_manager.clients:
            reason = "client_not_registered"
            logger.warning(f"Update rejected: client {client_id} not registered", extra={
                "component": "coordinator",
                "event": "update_rejected",
                "round_id": round_id,
                "client_id": client_id,
                "reason": reason
            })
            return False, reason
        
        # 3. Check if round exists and client is assigned to it
        if not self.round_manager.validate_update(client_id, round_id):
            reason = "invalid_round_or_assignment"
            logger.warning(f"Update rejected: invalid round or assignment", extra={
                "component": "coordinator",
                "event": "update_rejected",
                "round_id": round_id,
                "client_id": client_id,
                "reason": reason
            })
            return False, reason
        
        # 4. Rate limiting check
        if self.rate_limiter:
            allowed, rate_reason = self.rate_limiter.check_update_rate(client_id, round_id)
            if not allowed:
                logger.warning(f"Update rejected: rate limit exceeded", extra={
                    "component": "coordinator",
                    "event": "update_rejected",
                    "round_id": round_id,
                    "client_id": client_id,
                    "reason": "rate_limit_exceeded",
                    "details": rate_reason
                })
                return False, "rate_limit_exceeded"
        
        # 5. Basic validation: weight_delta should not be empty
        if not weight_delta or not isinstance(weight_delta, str):
            reason = "invalid_weight_delta_format"
            logger.warning(f"Update rejected: invalid weight_delta format", extra={
                "component": "coordinator",
                "event": "update_rejected",
                "round_id": round_id,
                "client_id": client_id,
                "reason": reason
            })
            return False, reason
        
        # 6. Validate update values (check for NaN/Inf)
        try:
            update_data = json.loads(weight_delta)
            weight_delta_list = update_data.get("weight_delta", [])
            
            if isinstance(weight_delta_list, list) and len(weight_delta_list) > 0:
                is_valid, error_msg = self.privacy_protector.validate_update_values(weight_delta_list)
                if not is_valid:
                    reason = "non_finite_values"
                    logger.warning(f"Update rejected: {error_msg}", extra={
                        "component": "coordinator",
                        "event": "update_rejected",
                        "round_id": round_id,
                        "client_id": client_id,
                        "reason": reason,
                        "details": error_msg
                    })
                    return False, reason
        except (json.JSONDecodeError, KeyError, TypeError):
            # If parsing fails, we'll let it through to basic validation
            # The aggregator will handle it
            pass
        
        return True, None

