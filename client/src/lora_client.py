"""
LoRA Client Module

Main client loop for federated LoRA fine-tuning.
Extends the base client to support LoRA training rounds.
"""

import time
import sys
import os
from typing import Optional

from config import config
from api import (
    download_lora_adapter,
    register_client,
    get_lora_round,
    CoordinatorAPIError,
    CoordinatorConnectionError
)
from training import train_lora_adapter, load_local_dataset
from submit import upload_adapter
from utils.logger import setup_client_logger, log_event
from security import get_api_key, save_api_key

logger = setup_client_logger()

# Mirrors coordinator model_registry HuggingFace names
_BASE_MODEL_NAMES = {
    "llama-7b": "meta-llama/Llama-2-7b-hf",
    "mistral-7b": "mistralai/Mistral-7B-v0.1",
    "phi-2": "microsoft/phi-2",
    "tiny-llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def get_base_model_name(base_model_id: str) -> str:
    """Resolve registry id to HuggingFace model name."""
    return _BASE_MODEL_NAMES.get(base_model_id, base_model_id)


def run_lora_client_loop(
    client_id: str,
    round_id: int,
    api_key: Optional[str] = None
) -> None:
    """
    Run a single LoRA training round.
    
    Args:
        client_id: Client identifier
        round_id: Round identifier to participate in
        api_key: API key for authentication
    """
    print(f"[Client {client_id}] Starting LoRA training for round {round_id}...")
    
    try:
        # Step 1: Fetch round configuration
        print(f"[Client {client_id}] Fetching round configuration...")
        round_config = get_lora_round(round_id, api_key=api_key)
        
        base_model_id = round_config["base_model_id"]
        base_model_name = get_base_model_name(base_model_id)
        print(f"[Client {client_id}] Round config received: base_model={base_model_name}, "
              f"max_steps={round_config.get('max_steps', 100)}")
        
        # Step 2: Load local dataset
        print(f"[Client {client_id}] Loading local dataset...")
        texts, num_samples = load_local_dataset()
        print(f"[Client {client_id}] Loaded {num_samples} training samples")
        
        # Step 3: Train LoRA adapter
        print(f"[Client {client_id}] Starting LoRA training...")
        training_start_time = time.time()
        
        try:
            previous_adapter_state = None
            previous_version = round_config.get("adapter_version")
            if previous_version:
                previous = download_lora_adapter(
                    previous_version,
                    client_id,
                    api_key=api_key,
                )
                previous_adapter_state = previous.get("adapter_state_dict")
                if not previous_adapter_state:
                    raise ValueError(
                        f"Adapter {previous_version} has no state dictionary"
                    )
            
            adapter_state_dict, metrics = train_lora_adapter(
                base_model_name=base_model_name,
                texts=texts,
                round_config=round_config,
                previous_adapter_state=previous_adapter_state,
            )
            
            training_duration = time.time() - training_start_time
            print(f"[Client {client_id}] Training completed in {training_duration:.2f}s")
            print(f"[Client {client_id}] Final loss: {metrics.final_loss:.4f}")
            
        except Exception as e:
            print(f"[Client {client_id}] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Step 4: Submit adapter
        print(f"[Client {client_id}] Submitting adapter to coordinator...")
        try:
            success = upload_adapter(
                client_id=client_id,
                round_id=round_id,
                adapter_state_dict=adapter_state_dict,
                num_samples=metrics.num_samples,
                training_loss=metrics.final_loss,
                api_key=api_key
            )
            
            if success:
                print(f"[Client {client_id}] Adapter submitted successfully")
                log_event(logger, "lora_adapter_submitted", client_id=client_id, round_id=round_id, extra_fields={
                    "num_samples": metrics.num_samples,
                    "training_loss": metrics.final_loss,
                    "training_duration": training_duration
                })
            else:
                print(f"[Client {client_id}] Adapter submission returned False")
        except CoordinatorConnectionError as e:
            print(f"[Client {client_id}] Coordinator unavailable during submission: {e}")
        except CoordinatorAPIError as e:
            print(f"[Client {client_id}] Failed to submit adapter: {e}")
    
    except CoordinatorConnectionError as e:
        print(f"[Client {client_id}] Coordinator unavailable: {e}")
    except CoordinatorAPIError as e:
        print(f"[Client {client_id}] API error: {e}")
    except Exception as e:
        print(f"[Client {client_id}] Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """
    Main entry point for LoRA client.
    
    Usage:
        python lora_client.py <round_id> [client_name]
    """
    print("=" * 60)
    print("Federated LoRA Fine-Tuning Client")
    print("=" * 60)
    
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python lora_client.py <round_id> [client_name]")
        sys.exit(1)
    
    try:
        round_id = int(sys.argv[1])
    except ValueError:
        print(f"ERROR: Invalid round_id: {sys.argv[1]}")
        sys.exit(1)
    
    client_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Generate client name if not provided
    if not client_name:
        import socket
        hostname = socket.gethostname()
        client_name = f"lora-client-{hostname}"
    
    print(f"Coordinator URL: {config.COORDINATOR_URL}")
    print(f"Round ID: {round_id}")
    print(f"Client Name: {client_name}")
    print()
    
    # Register client
    print(f"[Registration] Registering client '{client_name}' with coordinator...")
    api_key = get_api_key()
    
    if not api_key:
        try:
            client_id, api_key = register_client(client_name)
            save_api_key(api_key)
            print(f"[Registration] Successfully registered as '{client_id}'")
            print(f"[Registration] API key persisted for reconnect")
        except CoordinatorConnectionError as e:
            print(f"[Registration] ERROR: Cannot connect to coordinator: {e}")
            sys.exit(1)
        except CoordinatorAPIError as e:
            print(f"[Registration] ERROR: Registration failed: {e}")
            sys.exit(1)
    else:
        print(f"[Registration] Using existing API key")
        client_id = client_name
        # Refresh registration so coordinator knows this client after restart
        try:
            client_id, api_key = register_client(client_name)
            save_api_key(api_key)
        except Exception:
            pass
    
    # Run LoRA training
    run_lora_client_loop(client_id, round_id, api_key)


if __name__ == "__main__":
    main()

