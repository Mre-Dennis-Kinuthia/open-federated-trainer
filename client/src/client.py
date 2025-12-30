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
    
    # Try to use hostname (useful in Docker containers where hostname = container name)
    try:
        import socket
        hostname = socket.gethostname()
        if hostname and hostname != "localhost":
            return hostname
    except Exception:
        pass
    
    # Fallback: Generate a unique client name
    unique_id = str(uuid.uuid4())[:8]
    return f"client_{unique_id}"


def run_client_loop(client_id: str) -> None:
    """
    Main client execution loop.
    
    Continuously:
    1. Fetch training task
    2. Perform local training
    3. Submit update
    4. Sleep and repeat
    
    Args:
        client_id: Identifier of the client
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
                task = fetch_task(client_id)
                print(f"[Client {client_id}] Task received: Round {task['round_id']}, "
                      f"Model v{task['model_version']}, Task: {task['task']}")
            except CoordinatorConnectionError as e:
                print(f"[Client {client_id}] Coordinator unavailable: {e}")
                print(f"[Client {client_id}] Retrying in {config.RETRY_DELAY} seconds...")
                time.sleep(config.RETRY_DELAY)
                continue
            except CoordinatorAPIError as e:
                error_msg = str(e).lower()
                # Check if client is not registered (404 or similar)
                if "404" in error_msg or "not found" in error_msg or "not registered" in error_msg:
                    print(f"[Client {client_id}] Client not registered, attempting to re-register...")
                    try:
                        new_client_id = register_client(client_id)
                        print(f"[Client {client_id}] Re-registered successfully as '{new_client_id}'")
                        client_id = new_client_id  # Update client_id in case it changed
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
            try:
                weight_delta = train_local_model_with_client_id(task, client_id)
                print(f"[Client {client_id}] Training completed. Weight delta: {weight_delta[:50]}...")
            except Exception as e:
                print(f"[Client {client_id}] Training failed: {e}")
                print(f"[Client {client_id}] Skipping this round...")
                time.sleep(config.SLEEP_BETWEEN_ROUNDS)
                continue
            
            # Step 3: Submit update to coordinator
            print(f"[Client {client_id}] Submitting update for round {round_id}...")
            try:
                success = submit_update(client_id, round_id, weight_delta)
                if success:
                    print(f"[Client {client_id}] Update submitted successfully for round {round_id}")
                else:
                    print(f"[Client {client_id}] Update submission returned False")
            except CoordinatorConnectionError as e:
                print(f"[Client {client_id}] Coordinator unavailable during update: {e}")
                print(f"[Client {client_id}] Update may be lost. Continuing...")
            except CoordinatorAPIError as e:
                print(f"[Client {client_id}] Failed to submit update: {e}")
                print(f"[Client {client_id}] Update rejected by coordinator")
            
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
    try:
        client_id = register_client(client_name)
        print(f"[Registration] Successfully registered as '{client_id}'")
    except CoordinatorConnectionError as e:
        print(f"[Registration] ERROR: Cannot connect to coordinator: {e}")
        print(f"[Registration] Please ensure the coordinator is running at {config.COORDINATOR_URL}")
        sys.exit(1)
    except CoordinatorAPIError as e:
        # If client already exists, try to continue with the same name
        if "already registered" in str(e).lower():
            print(f"[Registration] Client already registered, continuing as '{client_name}'")
            client_id = client_name
        else:
            print(f"[Registration] ERROR: Registration failed: {e}")
            sys.exit(1)
    
    # Step 2: Start the main client loop
    try:
        run_client_loop(client_id)
    except KeyboardInterrupt:
        print("\n[Client] Shutdown requested by user")
    except Exception as e:
        print(f"[Client] Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

