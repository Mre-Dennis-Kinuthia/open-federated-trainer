# Federated Learning Coordinator

The Coordinator is the central server component of the federated learning platform. It manages training rounds, assigns tasks to clients, validates and aggregates model updates, and coordinates the federated learning process without centralizing data.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Core Modules](#core-modules)
- [Usage Examples](#usage-examples)
- [Development](#development)

## Overview

The Coordinator implements a round-based federated learning system where:

1. **Clients register** with the coordinator
2. **Tasks are assigned** to clients for training rounds
3. **Clients submit updates** after local training
4. **Updates are aggregated** into a global model
5. **Rounds progress** through states: OPEN → COLLECTING → AGGREGATING → CLOSED

The system is designed to be:
- **Modular**: Each component can be tested independently
- **Stateless**: Round state is managed in-memory (MVP)
- **Extensible**: Easy to add new aggregation strategies, validators, etc.

## Architecture

```
coordinator/
└── src/
    ├── core/
    │   ├── round_manager.py      # Manages rounds and client tracking
    │   ├── task_assigner.py      # Assigns tasks to clients
    │   ├── update_validator.py  # Validates client updates
    │   └── aggregator.py         # Aggregates client updates
    └── main.py                   # FastAPI server and endpoints
```

### Component Interaction

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       │ 1. Register
       ▼
┌─────────────────┐
│  Round Manager  │◄──┐
└────────┬────────┘   │
         │             │
         │ 2. Assign   │
         ▼             │
┌─────────────────┐   │
│  Task Assigner  │   │
└────────┬────────┘   │
         │             │
         │ 3. Get Task │
         ▼             │
┌─────────────┐       │
│   Client     │       │
└──────┬───────┘       │
       │                │
       │ 4. Submit      │
       │    Update      │
       ▼                │
┌─────────────────┐     │
│ Update Validator│     │
└────────┬────────┘     │
         │               │
         │ 5. Validate   │
         ▼               │
┌─────────────────┐     │
│   Aggregator    │─────┘
└─────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (or pip3)

### Setup

1. **Clone or navigate to the coordinator directory:**
   ```bash
   cd coordinator
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

### Dependencies

- `fastapi>=0.104.0` - Web framework for the API
- `uvicorn[standard]>=0.24.0` - ASGI server
- `pydantic>=2.0.0` - Data validation

## Quick Start

1. **Start the server:**
   ```bash
   cd src
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   cd src
   uvicorn main:app --reload
   ```

2. **The server will start on:** `http://0.0.0.0:8000`

3. **Access interactive API documentation:**
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## API Reference

### Base URL

All endpoints are relative to: `http://localhost:8000`

### Endpoints

#### 1. Register Client

Register a new client with the coordinator.

**Endpoint:** `POST /client/register`

**Request Body:**
```json
{
  "client_name": "client1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Client client1 registered successfully",
  "client_id": "client1"
}
```

**Status Codes:**
- `200 OK` - Client registered successfully
- `400 Bad Request` - Client already registered

---

#### 2. Get Task

Get a training task assignment for a client.

**Endpoint:** `GET /task/{client_id}`

**Path Parameters:**
- `client_id` (string) - Identifier of the client

**Response:**
```json
{
  "round_id": 1,
  "model_version": 1,
  "task": "train",
  "description": "Train model version 1 for round 1"
}
```

**Status Codes:**
- `200 OK` - Task assigned successfully
- `404 Not Found` - Client not registered or already has active assignment

---

#### 3. Submit Update

Submit a model update from a client.

**Endpoint:** `POST /update`

**Request Body:**
```json
{
  "client_id": "client1",
  "round_id": 1,
  "weight_delta": "delta_from_client1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Update from client client1 submitted successfully for round 1"
}
```

**Status Codes:**
- `200 OK` - Update submitted successfully
- `400 Bad Request` - Invalid update (client not registered, round not found, etc.)

---

#### 4. Get Round Status

Get the current status of a training round.

**Endpoint:** `GET /status/{round_id}`

**Path Parameters:**
- `round_id` (integer) - Identifier of the round

**Response:**
```json
{
  "round_id": 1,
  "state": "COLLECTING",
  "assigned_clients": ["client1", "client2"],
  "updates_received": ["client1"],
  "total_clients": 2,
  "total_updates": 1
}
```

**Round States:**
- `OPEN` - Round created but no clients assigned yet
- `COLLECTING` - Clients assigned, collecting updates
- `AGGREGATING` - Updates being aggregated
- `CLOSED` - Round completed

**Status Codes:**
- `200 OK` - Round status retrieved
- `404 Not Found` - Round not found

---

#### 5. Aggregate Round

Aggregate all client updates for a round.

**Endpoint:** `GET /aggregate/{round_id}`

**Path Parameters:**
- `round_id` (integer) - Identifier of the round

**Response:**
```json
{
  "round_id": 1,
  "status": "aggregated",
  "aggregated_model": {
    "weight_deltas": ["delta_from_client1", "delta_from_client2"],
    "num_updates": 2,
    "client_ids": ["client1", "client2"]
  },
  "num_updates": 2
}
```

**Status Codes:**
- `200 OK` - Aggregation completed
- `404 Not Found` - Round not found

---

#### 6. Root Endpoint

Get API information and available endpoints.

**Endpoint:** `GET /`

**Response:**
```json
{
  "message": "Federated Learning Coordinator API",
  "version": "1.0.0",
  "endpoints": {
    "register_client": "POST /client/register",
    "get_task": "GET /task/{client_id}",
    "submit_update": "POST /update",
    "aggregate_round": "GET /aggregate/{round_id}",
    "get_round_status": "GET /status/{round_id}"
  }
}
```

## Core Modules

### RoundManager

Manages federated learning rounds, tracks clients, and maintains round states.

**Key Methods:**
- `register_client(client_name: str) -> bool` - Register a new client
- `assign_client_to_round(client_id: str) -> Optional[int]` - Assign client to a round
- `validate_update(client_id: str, round_id: int) -> bool` - Validate if update is allowed
- `add_update(client_id: str, round_id: int, weight_delta: str) -> bool` - Record client update
- `get_round_status(round_id: int) -> Optional[Dict]` - Get round status information

**Round States:**
- `OPEN` - Round is open for client assignments
- `COLLECTING` - Collecting updates from clients
- `AGGREGATING` - Aggregating collected updates
- `CLOSED` - Round is complete

### TaskAssigner

Assigns training tasks to registered clients and prevents duplicate assignments.

**Key Methods:**
- `assign_task(client_id: str) -> Optional[Dict]` - Assign a task to a client
- `get_client_assignment(client_id: str) -> Optional[Dict]` - Get current assignment
- `increment_model_version() -> int` - Increment model version

**Features:**
- Prevents duplicate task assignments
- Tracks model versions
- Coordinates with RoundManager for round assignment

### UpdateValidator

Validates client updates before they are aggregated.

**Key Methods:**
- `validate(client_id: str, round_id: int, weight_delta: str) -> bool` - Validate an update

**Validation Checks:**
- Client is registered
- Client is assigned to the round
- Round exists and is in a valid state
- Weight delta is not empty

### Aggregator

Collects and aggregates client updates for federated learning rounds.

**Key Methods:**
- `submit_update(client_id: str, round_id: int, weight_delta: str) -> bool` - Submit a client update
- `aggregate(round_id: int) -> Optional[Dict]` - Aggregate all updates for a round
- `get_updates_for_round(round_id: int) -> List[ClientUpdate]` - Get all updates for a round

**Aggregation Strategy (MVP):**
- Simple collection of all weight deltas
- In production, this would implement federated averaging or other aggregation algorithms

## Usage Examples

### Example 1: Complete Workflow

```bash
# 1. Register a client
curl -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "client1"}'

# 2. Get a task
curl http://localhost:8000/task/client1

# 3. Submit an update (after local training)
curl -X POST http://localhost:8000/update \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "client1",
    "round_id": 1,
    "weight_delta": "delta_from_client1"
  }'

# 4. Check round status
curl http://localhost:8000/status/1

# 5. Aggregate the round
curl http://localhost:8000/aggregate/1
```

### Example 2: Multiple Clients

```bash
# Register multiple clients
curl -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "client1"}'

curl -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "client2"}'

# Assign tasks to both
curl http://localhost:8000/task/client1
curl http://localhost:8000/task/client2

# Both submit updates
curl -X POST http://localhost:8000/update \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client1", "round_id": 1, "weight_delta": "delta1"}'

curl -X POST http://localhost:8000/update \
  -H "Content-Type: application/json" \
  -d '{"client_id": "client2", "round_id": 1, "weight_delta": "delta2"}'

# Aggregate with both updates
curl http://localhost:8000/aggregate/1
```

### Example 3: Python Client

```python
import requests

BASE_URL = "http://localhost:8000"

# Register client
response = requests.post(
    f"{BASE_URL}/client/register",
    json={"client_name": "python_client"}
)
print(response.json())

# Get task
response = requests.get(f"{BASE_URL}/task/python_client")
task = response.json()
print(f"Assigned to round {task['round_id']}")

# Submit update
response = requests.post(
    f"{BASE_URL}/update",
    json={
        "client_id": "python_client",
        "round_id": task["round_id"],
        "weight_delta": "my_weight_delta"
    }
)
print(response.json())

# Check status
response = requests.get(f"{BASE_URL}/status/{task['round_id']}")
print(response.json())

# Aggregate
response = requests.get(f"{BASE_URL}/aggregate/{task['round_id']}")
print(response.json())
```

## Development

### Project Structure

```
coordinator/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── round_manager.py
│   │   ├── task_assigner.py
│   │   ├── update_validator.py
│   │   └── aggregator.py
│   └── main.py
├── requirements.txt
└── README.md
```

### Running in Development Mode

```bash
cd src
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes.

### Testing

The modules are designed to be independently testable. Example test structure:

```python
from core.round_manager import RoundManager

def test_register_client():
    rm = RoundManager()
    assert rm.register_client("client1") == True
    assert rm.register_client("client1") == False  # Duplicate
```

### Extending the Coordinator

#### Adding New Aggregation Strategies

Modify `aggregator.py` to implement different aggregation algorithms:

```python
def aggregate(self, round_id: int) -> Optional[Dict]:
    # Implement federated averaging, weighted averaging, etc.
    pass
```

#### Adding Validation Rules

Extend `update_validator.py` to add custom validation:

```python
def validate(self, client_id: str, round_id: int, weight_delta: str) -> bool:
    # Add custom validation logic
    if len(weight_delta) > MAX_SIZE:
        return False
    return True
```

#### Adding Persistence

Currently, all state is in-memory. To add persistence:

1. Add database models (SQLAlchemy, etc.)
2. Modify core modules to persist state
3. Add migration scripts

### Code Style

- Follow PEP 8
- Use type hints for all methods
- Include docstrings for classes and methods
- Keep modules focused and single-responsibility

## Limitations (MVP)

The current MVP implementation has the following limitations:

1. **In-Memory State**: All state is lost on server restart
2. **Simple Aggregation**: Weight deltas are collected as strings, not actual model weights
3. **No Authentication**: No client authentication or authorization
4. **No Persistence**: No database or file-based persistence
5. **Single Coordinator**: No distributed coordinator support

These are intentional for the MVP and can be addressed in future iterations.

## License

Part of the open-federated-trainer project.

## Support

For issues, questions, or contributions, please refer to the main project repository.

