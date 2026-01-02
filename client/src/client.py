"""
Main client module for federated learning.

Orchestrates the client-side federated learning workflow:
1. Register with coordinator
2. Fetch training tasks
3. Perform local training
4. Submit updates
5. Repeat
"""

import time
import sys
import uuid
import os
from typing import Optional

from config import config
from api import (
    register_client,
    fetch_task,
    submit_update,
    get_round_status,
    CoordinatorAPIError,
    CoordinatorConnectionError
)
from trainer import train_local_model_with_client_id
from utils.logger import setup_client_logger, log_event
from behavior import (
    simulate_startup_delay,
    should_dropout,
    get_training_speed_multiplier,
    simulate_coordinator_delay,
    apply_training_delay
)
from security import get_api_key, require_api_key, has_api_key

logger = setup_client_logger()


def generate_client_name() -> str:
    """
    Generate a unique client name if not provided in config.
    
    In Docker containers, uses hostname (container name) for unique identification.
    Otherwise generates a UUID-based name.
    
    Returns:
        Client name string
    """
    if config.CLIENT_NAME:
        return config.CLIENT_NAME
    
    # Try to use hostname from environment (Docker Compose sets HOSTNAME)
    hostname = os.getenv("HOSTNAME")
    if hostname and hostname != "localhost":
        # Clean up hostname - remove common prefixes/suffixes
        # Docker Compose creates names like "open-federated-trainer-client-1"
        # Extract just the meaningful part
        if "client" in hostname.lower():
            # Extract instance number or use full hostname
            parts = hostname.split("-")
            if len(parts) > 1:
                # Try to find a number in the hostname
                for part in reversed(parts):
                    if part.isdigit():
                        return f"client-{part}"
            return hostname.replace("open-federated-trainer-", "").replace("-", "_")
        return hostname
    
    # Try to use hostname from socket (fallback)
    try:
        import socket
        hostname = socket.gethostname()
        if hostname and hostname not in ["localhost", "localhost.localdomain"]:
            return hostname.replace("-", "_")
    except Exception:
        pass
    
    # Fallback: Generate a unique client name
    unique_id = str(uuid.uuid4())[:8]
    return f"client_{unique_id}"


def run_client_loop(client_id: str, api_key: Optional[str] = None) -> None:
    """
    Main client execution loop.
    
    Continuously:
    1. Fetch training task
    2. Perform local training
    3. Submit update
    4. Sleep and repeat
    
    Args:
        client_id: Identifier of the client
        api_key: API key for authentication
    """
    print(f"[Client {client_id}] Starting federated learning client loop...")
    
    round_count = 0
    
    while True:
        try:
            round_count += 1
            print(f"\n[Client {client_id}] === Round {round_count} ===")
            
            # Step 1: Fetch training task
            print(f"[Client {client_id}] Fetching training task...")
            try:
                # Simulate coordinator delay (if enabled)
                coordinator_delay = simulate_coordinator_delay()
                if coordinator_delay > 0:
                    print(f"[Client {client_id}] Behavior simulation: Coordinator delay {coordinator_delay:.2f}s")
                    time.sleep(coordinator_delay)
                
                task = fetch_task(client_id, api_key=api_key)
                round_id = task["round_id"]
                print(f"[Client {client_id}] Task received: Round {round_id}, "
                      f"Model {task['model_version']}, Task: {task['task']}")
                log_event(logger, "task_received", client_id=client_id, round_id=round_id, extra_fields={
                    "model_version": task["model_version"],
                    "task": task["task"]
                })
            except CoordinatorConnectionError as e:
                print(f"[Client {client_id}] Coordinator unavailable: {e}")
                print(f"[Client {client_id}] Retrying in {config.RETRY_DELAY} seconds...")
                time.sleep(config.RETRY_DELAY)
                continue
            except CoordinatorAPIError as e:
                error_msg = str(e).lower()
                # Check if client is not registered (404 or similar)
                if "404" in error_msg or "not found" in error_msg or "not registered" in error_msg or "401" in error_msg or "authentication" in error_msg.lower():
                    print(f"[Client {client_id}] Client not registered or authentication failed, attempting to re-register...")
                    try:
                        new_client_id, new_api_key = register_client(client_id)
                        print(f"[Client {client_id}] Re-registered successfully as '{new_client_id}'")
                        print(f"[Client {client_id}] New API Key: {new_api_key}")
                        client_id = new_client_id  # Update client_id in case it changed
                        api_key = new_api_key  # Update API key
                        continue  # Retry fetching task
                    except Exception as reg_error:
                        print(f"[Client {client_id}] Re-registration failed: {reg_error}")
                        print(f"[Client {client_id}] Waiting {config.SLEEP_BETWEEN_ROUNDS} seconds before retry...")
                        time.sleep(config.SLEEP_BETWEEN_ROUNDS)
                        continue
                else:
                    print(f"[Client {client_id}] Failed to fetch task: {e}")
                    print(f"[Client {client_id}] Waiting {config.SLEEP_BETWEEN_ROUNDS} seconds before retry...")
                    time.sleep(config.SLEEP_BETWEEN_ROUNDS)
                    continue
            
            round_id = task["round_id"]
            
            # Step 2: Perform local training
            print(f"[Client {client_id}] Starting local training for round {round_id}...")
            training_start_time = time.time()
            log_event(logger, "training_started", client_id=client_id, round_id=round_id)
            
            try:
                weight_delta = train_local_model_with_client_id(task, client_id)
                training_duration = time.time() - training_start_time
                update_size_bytes = len(weight_delta.encode('utf-8'))
                print(f"[Client {client_id}] Training completed. Weight delta: {weight_delta[:50]}...")
                log_event(logger, "training_completed", client_id=client_id, round_id=round_id, extra_fields={
                    "training_duration_seconds": training_duration,
                    "update_size_parameters": len(weight_delta),  # Approximate
                    "update_size_bytes": update_size_bytes
                })
            except Exception as e:
                print(f"[Client {client_id}] Training failed: {e}")
                print(f"[Client {client_id}] Skipping this round...")
                time.sleep(config.SLEEP_BETWEEN_ROUNDS)
                continue
            
            # Step 3: Submit update to coordinator
            print(f"[Client {client_id}] Submitting update for round {round_id}...")
            try:
                success = submit_update(client_id, round_id, weight_delta, api_key=api_key)
                if success:
                    print(f"[Client {client_id}] Update submitted successfully for round {round_id}")
                    log_event(logger, "update_sent", client_id=client_id, round_id=round_id, extra_fields={
                        "update_size_bytes": len(weight_delta.encode('utf-8'))
                    })
                else:
                    print(f"[Client {client_id}] Update submission returned False")
                    log_event(logger, "update_failed", level="WARNING", client_id=client_id, round_id=round_id, extra_fields={
                        "reason": "submission_returned_false"
                    })
            except CoordinatorConnectionError as e:
                print(f"[Client {client_id}] Coordinator unavailable during update: {e}")
                print(f"[Client {client_id}] Update may be lost. Continuing...")
                log_event(logger, "update_failed", level="WARNING", client_id=client_id, round_id=round_id, extra_fields={
                    "reason": "coordinator_unavailable",
                    "error": str(e)
                })
            except CoordinatorAPIError as e:
                print(f"[Client {client_id}] Failed to submit update: {e}")
                print(f"[Client {client_id}] Update rejected by coordinator")
                log_event(logger, "update_failed", level="WARNING", client_id=client_id, round_id=round_id, extra_fields={
                    "reason": "coordinator_rejected",
                    "error": str(e)
                })
            
            # Optional: Check round status
            try:
                status = get_round_status(round_id)
                print(f"[Client {client_id}] Round {round_id} status: {status['state']}, "
                      f"{status['total_updates']}/{status['total_clients']} updates received")
            except Exception as e:
                # Non-critical, just log
                print(f"[Client {client_id}] Could not fetch round status: {e}")
            
            # Step 4: Sleep before next round
            print(f"[Client {client_id}] Waiting {config.SLEEP_BETWEEN_ROUNDS} seconds before next round...")
            time.sleep(config.SLEEP_BETWEEN_ROUNDS)
            
        except KeyboardInterrupt:
            print(f"\n[Client {client_id}] Shutting down gracefully...")
            break
        except Exception as e:
            print(f"[Client {client_id}] Unexpected error: {e}")
            print(f"[Client {client_id}] Waiting {config.SLEEP_BETWEEN_ROUNDS} seconds before retry...")
            time.sleep(config.SLEEP_BETWEEN_ROUNDS)


def main() -> None:
    """
    Main entry point for the federated learning client.
    """
    print("=" * 60)
    print("Federated Learning Client")
    print("=" * 60)
    print(f"Coordinator URL: {config.COORDINATOR_URL}")
    print(f"Max Retries: {config.MAX_RETRIES}")
    print(f"Sleep Between Rounds: {config.SLEEP_BETWEEN_ROUNDS}s")
    print("=" * 60)
    
    # Generate or use configured client name
    client_name = generate_client_name()
    print(f"Client Name: {client_name}")
    
    # Step 1: Register with coordinator
    print(f"\n[Registration] Registering client '{client_name}' with coordinator...")
    
    # Check if API key already exists
    api_key = get_api_key()
    if api_key:
        print(f"[Registration] Using existing API key from CLIENT_API_KEY")
        client_id = client_name
    else:
        try:
            client_id, api_key = register_client(client_name)
            print(f"[Registration] Successfully registered as '{client_id}'")
            print(f"[Registration] API Key: {api_key}")
            print(f"[Registration] ⚠️  IMPORTANT: Save this API key!")
            print(f"[Registration] Set it as: export CLIENT_API_KEY='{api_key}'")
        except CoordinatorConnectionError as e:
            print(f"[Registration] ERROR: Cannot connect to coordinator: {e}")
            print(f"[Registration] Please ensure the coordinator is running at {config.COORDINATOR_URL}")
            sys.exit(1)
        except CoordinatorAPIError as e:
            # If client already exists, try to continue with the same name
            if "already registered" in str(e).lower():
                print(f"[Registration] Client already registered, but no API key found!")
                print(f"[Registration] ERROR: CLIENT_API_KEY environment variable is required.")
                print(f"[Registration] Please set CLIENT_API_KEY or re-register the client.")
                sys.exit(1)
            else:
                print(f"[Registration] ERROR: Registration failed: {e}")
                sys.exit(1)
    
    log_event(logger, "client_started", client_id=client_id, extra_fields={
        "coordinator_url": config.COORDINATOR_URL,
        "has_api_key": api_key is not None
    })
    
    # Step 2: Start the main client loop
    try:
        run_client_loop(client_id, api_key=api_key)
    except KeyboardInterrupt:
        print("\n[Client] Shutdown requested by user")
    except Exception as e:
        print(f"[Client] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

