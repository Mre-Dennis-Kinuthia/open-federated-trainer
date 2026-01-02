"""
API module for communicating with the federated learning coordinator.

Handles all HTTP communication with the coordinator server.
"""

import time
import requests
from typing import Dict, Optional, Any
from requests.exceptions import RequestException, Timeout, ConnectionError

from config import config
from security import get_api_key


class CoordinatorAPIError(Exception):
    """Exception raised for coordinator API errors."""
    pass


class CoordinatorConnectionError(Exception):
    """Exception raised for coordinator connection errors."""
    pass


def _make_request(
    method: str,
    url: str,
    max_retries: int = None,
    retry_delay: float = None,
    **kwargs
) -> requests.Response:
    """
    Make an HTTP request with retry logic.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retries (defaults to config.MAX_RETRIES)
        retry_delay: Delay between retries in seconds (defaults to config.RETRY_DELAY)
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        Response object
        
    Raises:
        CoordinatorConnectionError: If connection fails after all retries
        CoordinatorAPIError: If API returns an error status code
    """
    if max_retries is None:
        max_retries = config.MAX_RETRIES
    if retry_delay is None:
        retry_delay = config.RETRY_DELAY
    
    if "timeout" not in kwargs:
        kwargs["timeout"] = config.REQUEST_TIMEOUT
    
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = requests.request(method, url, **kwargs)
            
            # Check for HTTP errors
            if response.status_code >= 400:
                error_msg = f"API error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", "")
                    if error_detail:
                        error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text}"
                
                raise CoordinatorAPIError(error_msg)
            
            return response
            
        except (ConnectionError, Timeout) as e:
            last_exception = e
            if attempt < max_retries:
                print(f"Connection error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(retry_delay)
            else:
                raise CoordinatorConnectionError(
                    f"Failed to connect to coordinator after {max_retries + 1} attempts: {e}"
                )
        
        except RequestException as e:
            last_exception = e
            if attempt < max_retries:
                print(f"Request error (attempt {attempt + 1}/{max_retries + 1}): {e}")
                time.sleep(retry_delay)
            else:
                raise CoordinatorConnectionError(
                    f"Request failed after {max_retries + 1} attempts: {e}"
                )
    
    # Should not reach here, but just in case
    raise CoordinatorConnectionError(f"Request failed: {last_exception}")


def register_client(client_name: str) -> tuple[str, str]:
    """
    Register a client with the coordinator and receive an API key.
    
    Args:
        client_name: Name/identifier for the client
        
    Returns:
        Tuple of (client_id, api_key)
        
    Raises:
        CoordinatorAPIError: If registration fails
        CoordinatorConnectionError: If connection fails
    """
    url = f"{config.COORDINATOR_URL}/client/register"
    payload = {"client_name": client_name}
    
    response = _make_request("POST", url, json=payload)
    data = response.json()
    
    if data.get("success"):
        client_id = data.get("client_id", client_name)
        api_key = data.get("api_key", "")
        return client_id, api_key
    else:
        raise CoordinatorAPIError(f"Registration failed: {data.get('message', 'Unknown error')}")


def fetch_task(client_id: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch a training task from the coordinator.
    
    Args:
        client_id: Identifier of the client
        api_key: Optional API key (uses security.get_api_key() if not provided)
        
    Returns:
        Task dictionary with round_id, model_version, task, and description
        
    Raises:
        CoordinatorAPIError: If task fetch fails
        CoordinatorConnectionError: If connection fails
    """
    if api_key is None:
        api_key = get_api_key()
    
    url = f"{config.COORDINATOR_URL}/task/{client_id}"
    params = {}
    if api_key:
        params["api_key"] = api_key
    
    response = _make_request("GET", url, params=params)
    return response.json()


def submit_update(
    client_id: str,
    round_id: int,
    weight_delta: str,
    api_key: Optional[str] = None
) -> bool:
    """
    Submit a model update to the coordinator.
    
    Args:
        client_id: Identifier of the client
        round_id: Identifier of the round
        weight_delta: Weight delta update (as string in MVP)
        api_key: Optional API key (uses security.get_api_key() if not provided)
        
    Returns:
        True if submission was successful
        
    Raises:
        CoordinatorAPIError: If submission fails
        CoordinatorConnectionError: If connection fails
    """
    if api_key is None:
        api_key = get_api_key()
    
    url = f"{config.COORDINATOR_URL}/update"
    payload = {
        "client_id": client_id,
        "round_id": round_id,
        "weight_delta": weight_delta
    }
    if api_key:
        payload["api_key"] = api_key
    
    response = _make_request("POST", url, json=payload)
    data = response.json()
    
    if data.get("success"):
        return True
    else:
        raise CoordinatorAPIError(f"Update submission failed: {data.get('message', 'Unknown error')}")


def get_round_status(round_id: int) -> Dict[str, Any]:
    """
    Get the status of a training round.
    
    Args:
        round_id: Identifier of the round
        
    Returns:
        Round status dictionary
        
    Raises:
        CoordinatorAPIError: If status fetch fails
        CoordinatorConnectionError: If connection fails
    """
    url = f"{config.COORDINATOR_URL}/status/{round_id}"
    
    response = _make_request("GET", url)
    return response.json()

