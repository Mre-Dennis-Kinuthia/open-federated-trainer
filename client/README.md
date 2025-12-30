# Federated Learning Client

The Client is the participant component of the federated learning platform. It connects to a coordinator server, receives training tasks, performs local model training, and submits updates back to the coordinator without sharing raw data.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Reference](#module-reference)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [Development](#development)
- [Testing](#testing)

## Overview

The Client implements a participant in a federated learning system where:

1. **Client registers** with the coordinator
2. **Receives training tasks** for specific rounds
3. **Performs local training** on its data (simulated in MVP)
4. **Submits weight updates** to the coordinator
5. **Repeats the cycle** for continuous learning

The client is designed to be:
- **Resilient**: Handles network failures and coordinator downtime gracefully
- **Modular**: Each component can be tested and replaced independently
- **Extensible**: Easy to replace simulated training with real ML frameworks
- **Autonomous**: Automatically re-registers if connection is lost

## Architecture

```
client/
└── src/
    ├── config.py      # Configuration management
    ├── api.py         # HTTP communication with coordinator
    ├── trainer.py     # Local model training (simulated)
    └── client.py      # Main execution loop
```

### Component Interaction

```
┌─────────────┐
│  client.py  │
│  (Main)     │
└──────┬──────┘
       │
       ├──► config.py (Configuration)
       │
       ├──► api.py (HTTP Requests)
       │    ├── register_client()
       │    ├── fetch_task()
       │    ├── submit_update()
       │    └── get_round_status()
       │
       └──► trainer.py (Local Training)
            └── train_local_model_with_client_id()
```

### Workflow

```
1. Start Client
   │
   ├──► Register with Coordinator
   │
   └──► Main Loop:
        │
        ├──► Fetch Task
        │    │
        │    └──► (If 404: Auto re-register)
        │
        ├──► Train Locally
        │    │
        │    └──► Generate Weight Delta
        │
        ├──► Submit Update
        │
        ├──► Check Round Status
        │
        └──► Sleep & Repeat
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (or pip3)
- Access to a running coordinator server

### Setup

1. **Navigate to the client directory:**
   ```bash
   cd client
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   cd src
   python -c "import client; print('Client imports successful!')"
   ```

## Quick Start

### Basic Usage

1. **Ensure the coordinator is running** (see coordinator README)

2. **Run the client:**
   ```bash
   cd client/src
   python client.py
   ```

3. **The client will:**
   - Generate a unique client name (or use `CLIENT_NAME` env var)
   - Register with the coordinator
   - Start participating in training rounds

### With Custom Configuration

```bash
COORDINATOR_URL=http://localhost:8000 \
CLIENT_NAME=my_client \
MAX_RETRIES=5 \
SLEEP_BETWEEN_ROUNDS=10 \
python client.py
```

### Multiple Clients

Run multiple clients simultaneously to test federated learning:

**Terminal 1:**
```bash
cd client/src
CLIENT_NAME=client1 python client.py
```

**Terminal 2:**
```bash
cd client/src
CLIENT_NAME=client2 python client.py
```

**Terminal 3:**
```bash
cd client/src
CLIENT_NAME=client3 python client.py
```

All clients will participate in the same rounds and submit updates.

## Module Reference

### `config.py`

Configuration management module that loads settings from environment variables.

**Configuration Options:**

| Variable | Environment Variable | Default | Description |
|----------|---------------------|---------|-------------|
| `COORDINATOR_URL` | `COORDINATOR_URL` | `http://127.0.0.1:8000` | Coordinator server URL |
| `CLIENT_NAME` | `CLIENT_NAME` | `None` (auto-generated) | Client identifier |
| `MAX_RETRIES` | `MAX_RETRIES` | `3` | Maximum retry attempts for requests |
| `SLEEP_BETWEEN_ROUNDS` | `SLEEP_BETWEEN_ROUNDS` | `5.0` | Seconds to wait between rounds |
| `REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | `30.0` | Request timeout in seconds |
| `RETRY_DELAY` | `RETRY_DELAY` | `2.0` | Delay between retries in seconds |

**Usage:**
```python
from config import config

print(config.COORDINATOR_URL)
print(config.CLIENT_NAME)
```

### `api.py`

HTTP communication module for interacting with the coordinator.

**Functions:**

#### `register_client(client_name: str) -> str`
Register a client with the coordinator.

**Parameters:**
- `client_name`: Name/identifier for the client

**Returns:**
- Client ID (same as client_name)

**Raises:**
- `CoordinatorAPIError`: If registration fails
- `CoordinatorConnectionError`: If connection fails

**Example:**
```python
from api import register_client

client_id = register_client("my_client")
```

#### `fetch_task(client_id: str) -> Dict[str, Any]`
Fetch a training task from the coordinator.

**Parameters:**
- `client_id`: Identifier of the client

**Returns:**
- Task dictionary with `round_id`, `model_version`, `task`, and `description`

**Raises:**
- `CoordinatorAPIError`: If task fetch fails (e.g., client not registered)
- `CoordinatorConnectionError`: If connection fails

**Example:**
```python
from api import fetch_task

task = fetch_task("my_client")
print(task["round_id"])  # e.g., 1
print(task["model_version"])  # e.g., 1
```

#### `submit_update(client_id: str, round_id: int, weight_delta: str) -> bool`
Submit a model update to the coordinator.

**Parameters:**
- `client_id`: Identifier of the client
- `round_id`: Identifier of the round
- `weight_delta`: Weight delta update (as string in MVP)

**Returns:**
- `True` if submission was successful

**Raises:**
- `CoordinatorAPIError`: If submission fails
- `CoordinatorConnectionError`: If connection fails

**Example:**
```python
from api import submit_update

success = submit_update("my_client", 1, "delta_abc123")
```

#### `get_round_status(round_id: int) -> Dict[str, Any]`
Get the status of a training round.

**Parameters:**
- `round_id`: Identifier of the round

**Returns:**
- Round status dictionary with `state`, `total_updates`, `total_clients`, etc.

**Raises:**
- `CoordinatorAPIError`: If status fetch fails
- `CoordinatorConnectionError`: If connection fails

**Example:**
```python
from api import get_round_status

status = get_round_status(1)
print(status["state"])  # e.g., "COLLECTING"
print(status["total_updates"])  # e.g., 2
```

**Exceptions:**

- `CoordinatorAPIError`: Raised for API errors (4xx, 5xx responses)
- `CoordinatorConnectionError`: Raised for connection/network errors

**Retry Logic:**

All API functions automatically retry on connection failures:
- Maximum retries: `config.MAX_RETRIES` (default: 3)
- Retry delay: `config.RETRY_DELAY` (default: 2.0 seconds)
- Request timeout: `config.REQUEST_TIMEOUT` (default: 30.0 seconds)

### `trainer.py`

Local model training module. Currently simulates training for the MVP.

**Functions:**

#### `train_local_model_with_client_id(task: Dict[str, Any], client_id: str) -> str`
Simulate local model training and return a deterministic weight delta.

**Parameters:**
- `task`: Task dictionary containing `round_id`, `model_version`, `task`, `description`
- `client_id`: Identifier of the client performing training

**Returns:**
- Weight delta as a string (deterministic based on inputs)

**Note:** This is a placeholder implementation. In production, replace this with actual PyTorch/TensorFlow training code.

**Example:**
```python
from trainer import train_local_model_with_client_id

task = {"round_id": 1, "model_version": 1, "task": "train", "description": "..."}
weight_delta = train_local_model_with_client_id(task, "my_client")
# Returns: "delta_my_client_r1_v1_a1b2c3d4"
```

**Replacing with Real Training:**

To use real ML frameworks, replace the function body:

```python
def train_local_model_with_client_id(task: Dict[str, Any], client_id: str) -> str:
    import torch
    
    # Load model
    model = load_model(task["model_version"])
    
    # Load local data
    train_loader = get_local_data()
    
    # Train
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(10):
        for batch in train_loader:
            # Training loop...
            pass
    
    # Extract weight delta
    weight_delta = serialize_weights(model.state_dict())
    return weight_delta
```

### `client.py`

Main execution module that orchestrates the client workflow.

**Functions:**

#### `main() -> None`
Main entry point for the federated learning client.

**Workflow:**
1. Load configuration
2. Generate or use configured client name
3. Register with coordinator
4. Start main client loop

#### `run_client_loop(client_id: str) -> None`
Main client execution loop.

**Parameters:**
- `client_id`: Identifier of the client

**Loop Steps:**
1. Fetch training task
2. Perform local training
3. Submit update to coordinator
4. Check round status (optional)
5. Sleep and repeat

**Error Handling:**
- Automatically re-registers if client receives 404 (not registered)
- Handles coordinator downtime gracefully
- Retries on connection errors
- Continues on training failures (skips round)

**Example Output:**
```
============================================================
Federated Learning Client
============================================================
Coordinator URL: http://127.0.0.1:8000
Max Retries: 3
Sleep Between Rounds: 5.0s
============================================================
Client Name: client_abc12345

[Registration] Registering client 'client_abc12345' with coordinator...
[Registration] Successfully registered as 'client_abc12345'

[Client client_abc12345] Starting federated learning client loop...

[Client client_abc12345] === Round 1 ===
[Client client_abc12345] Fetching training task...
[Client client_abc12345] Task received: Round 1, Model v1, Task: train
[Client client_abc12345] Starting local training for round 1...
[Client client_abc12345] Training completed. Weight delta: delta_client_abc12345_r1_v1_a1b2c3d4...
[Client client_abc12345] Submitting update for round 1...
[Client client_abc12345] Update submitted successfully for round 1
[Client client_abc12345] Round 1 status: COLLECTING, 1/2 updates received
[Client client_abc12345] Waiting 5.0 seconds before next round...
```

## Configuration

### Environment Variables

All configuration can be set via environment variables:

```bash
# Coordinator URL
export COORDINATOR_URL="http://localhost:8000"

# Client name (auto-generated if not set)
export CLIENT_NAME="my_client"

# Retry configuration
export MAX_RETRIES=5
export RETRY_DELAY=2.0

# Timing configuration
export SLEEP_BETWEEN_ROUNDS=10.0
export REQUEST_TIMEOUT=30.0
```

### Configuration File

You can also modify `config.py` directly, but environment variables take precedence.

## Usage Examples

### Example 1: Basic Single Client

```bash
cd client/src
python client.py
```

### Example 2: Custom Client Name

```bash
cd client/src
CLIENT_NAME=experiment_1 python client.py
```

### Example 3: Multiple Clients with Different Names

**Terminal 1:**
```bash
cd client/src
CLIENT_NAME=client_alpha python client.py
```

**Terminal 2:**
```bash
cd client/src
CLIENT_NAME=client_beta python client.py
```

**Terminal 3:**
```bash
cd client/src
CLIENT_NAME=client_gamma python client.py
```

### Example 4: Custom Configuration

```bash
cd client/src
COORDINATOR_URL=http://192.168.1.100:8000 \
CLIENT_NAME=remote_client \
MAX_RETRIES=10 \
SLEEP_BETWEEN_ROUNDS=15.0 \
python client.py
```

### Example 5: Using as a Module

```python
from client import main

# Run the client programmatically
main()
```

## Error Handling

The client implements comprehensive error handling:

### Connection Errors

**Symptom:** `CoordinatorConnectionError: Failed to connect to coordinator`

**Causes:**
- Coordinator server is down
- Network connectivity issues
- Firewall blocking connections

**Behavior:**
- Client retries up to `MAX_RETRIES` times
- Waits `RETRY_DELAY` seconds between retries
- Continues retrying in the main loop

**Solution:**
1. Verify coordinator is running: `curl http://localhost:8000/`
2. Check `COORDINATOR_URL` configuration
3. Verify network connectivity

### Registration Errors

**Symptom:** `CoordinatorAPIError: Client already registered`

**Causes:**
- Client name already exists
- Coordinator was restarted but client wasn't

**Behavior:**
- Client attempts to continue with the same name
- If registration fails for other reasons, client exits

**Solution:**
- Use a different `CLIENT_NAME` or let client auto-generate one
- Restart client after coordinator restart

### Task Fetch Errors

**Symptom:** `CoordinatorAPIError: 404 - Client may not be registered`

**Causes:**
- Client lost registration (coordinator restart)
- Client not properly registered

**Behavior:**
- Client automatically detects 404 errors
- Automatically attempts re-registration
- Retries task fetch after successful re-registration

**Solution:**
- Usually self-resolving (automatic re-registration)
- If persistent, restart the client

### Update Submission Errors

**Symptom:** `CoordinatorAPIError: Update submission failed`

**Causes:**
- Invalid update format
- Round already closed
- Client not assigned to round

**Behavior:**
- Client logs the error
- Continues to next round (update is lost)

**Solution:**
- Check coordinator logs for details
- Verify round is still open
- Ensure client is properly registered

### Training Errors

**Symptom:** `Training failed: <error>`

**Causes:**
- Error in training code
- Resource constraints (memory, CPU)

**Behavior:**
- Client logs the error
- Skips the current round
- Continues to next round

**Solution:**
- Fix training code
- Check system resources
- Review error logs

## Development

### Project Structure

```
client/
├── src/
│   ├── __init__.py
│   ├── config.py      # Configuration
│   ├── api.py         # HTTP communication
│   ├── trainer.py     # Training simulation
│   └── client.py      # Main entry point
├── requirements.txt   # Dependencies
├── README.md          # This file
└── TESTING.md         # Testing guide
```

### Adding Real Training

To replace simulated training with real ML frameworks:

1. **Install ML framework:**
   ```bash
   pip install torch  # or tensorflow
   ```

2. **Modify `trainer.py`:**
   ```python
   def train_local_model_with_client_id(task: Dict[str, Any], client_id: str) -> str:
       # Load model
       model = load_model_for_version(task["model_version"])
       
       # Load local data
       train_loader = get_local_training_data()
       
       # Train model
       optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
       criterion = torch.nn.CrossEntropyLoss()
       
       model.train()
       for epoch in range(10):
           for batch in train_loader:
               inputs, labels = batch
               optimizer.zero_grad()
               outputs = model(inputs)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()
       
       # Serialize weight delta
       weight_delta = serialize_model_weights(model.state_dict())
       return weight_delta
   ```

3. **Update `requirements.txt`:**
   ```
   requests>=2.28.1
   torch>=1.12.0  # or tensorflow>=2.8.0
   ```

### Extending API Functions

To add new coordinator endpoints:

1. **Add function to `api.py`:**
   ```python
   def get_model_weights(model_version: int) -> Dict[str, Any]:
       """Fetch model weights from coordinator."""
       url = f"{config.COORDINATOR_URL}/model/{model_version}"
       response = _make_request("GET", url)
       return response.json()
   ```

2. **Use in `client.py`:**
   ```python
   from api import get_model_weights
   
   weights = get_model_weights(task["model_version"])
   ```

### Testing

See `TESTING.md` for comprehensive testing instructions.

**Quick Test:**
```bash
# Terminal 1: Start coordinator
cd coordinator/src
python main.py

# Terminal 2: Start client
cd client/src
python client.py
```

## Testing

For detailed testing instructions, see [TESTING.md](./TESTING.md).

### Quick Verification

1. **Start coordinator:**
   ```bash
   cd coordinator/src
   python main.py
   ```

2. **Start client:**
   ```bash
   cd client/src
   python client.py
   ```

3. **Verify:**
   - Client registers successfully
   - Client receives tasks
   - Client submits updates
   - Multiple clients can run simultaneously

### Test Scripts

Use the provided test script:

```bash
cd client
bash test_client.sh
```

This script:
- Starts the coordinator (if not running)
- Runs multiple clients
- Monitors the coordinator status

## Troubleshooting

### Client can't connect to coordinator

**Error:** `CoordinatorConnectionError: Failed to connect to coordinator`

**Solutions:**
1. Verify coordinator is running: `curl http://localhost:8000/`
2. Check `COORDINATOR_URL` in config or environment
3. Check firewall/network settings
4. Verify coordinator is listening on the correct interface

### Client registration fails

**Error:** `Client already registered`

**Solutions:**
- Use a different `CLIENT_NAME` or let client auto-generate one
- Restart client after coordinator restart
- Check coordinator logs for details

### No tasks assigned

**Error:** `Could not assign task to client`

**Solutions:**
- Client may already have an active assignment
- Wait for the current round to complete
- Check coordinator logs for details
- Verify client is properly registered

### Client stuck in loop

**Symptom:** Client repeatedly fails to fetch tasks

**Solutions:**
1. Check coordinator is running and responsive
2. Verify client registration status
3. Check network connectivity
4. Review client logs for specific error messages
5. Restart both coordinator and client if needed

### Update submission fails

**Error:** `Update submission failed`

**Solutions:**
- Verify round is still open
- Check update format is correct
- Ensure client is assigned to the round
- Review coordinator logs for validation errors

## API Compatibility

The client is compatible with the coordinator API version 1.0:

- `POST /client/register` - Client registration
- `GET /task/{client_id}` - Fetch training task
- `POST /update` - Submit model update
- `GET /status/{round_id}` - Get round status

See the coordinator README for full API documentation.

## License

See the main project LICENSE file.

## Contributing

See the main project CONTRIBUTING guidelines.
