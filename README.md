# Open Federated Trainer

**fed-compute** is an open MVP federated learning platform for volunteer and edge compute. It coordinates round-based local training across distributed clients, aggregates model updates into a global model, and tolerates partial participation, network instability, and heterogeneous hardware **without centralizing data**.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Security Features](#security-features)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Overview

Open Federated Trainer is a production-ready federated learning system that enables distributed machine learning across volunteer and edge computing resources. Inspired by volunteer computing projects like SETI@home and Folding@home, this platform allows multiple clients to collaboratively train machine learning models without sharing raw data.

### What is Federated Learning?

Federated Learning is a distributed machine learning approach where:
- **Data stays local**: Training data never leaves the client devices
- **Models travel**: Only model updates (weight deltas) are shared
- **Privacy preserved**: Raw data remains decentralized
- **Collaborative learning**: Multiple participants contribute to a global model

### How It Works

1. **Registration**: Clients register with the coordinator and receive API keys
2. **Task Assignment**: Coordinator assigns training tasks to clients for specific rounds
3. **Local Training**: Clients train models locally using PyTorch on their own data
4. **Update Submission**: Clients submit weight deltas (not raw data) to the coordinator
5. **Aggregation**: Coordinator aggregates updates using federated averaging
6. **Model Evolution**: New model versions are created and distributed for the next round
7. **Repeat**: The cycle continues, improving the global model iteratively

## Key Features

### 🚀 Core Capabilities

- **Real PyTorch Training**: Actual neural network training using PyTorch (not simulation)
- **Round-Based Coordination**: Structured training rounds with state management
- **Federated Averaging**: Standard aggregation algorithm for model updates
- **Model Versioning**: Automatic versioning of aggregated models
- **Partial Participation**: System tolerates clients joining/leaving dynamically
- **Network Resilience**: Handles connection failures and retries gracefully

### 🔒 Security & Privacy

- **API Key Authentication**: Secure client-coordinator communication
- **Rate Limiting**: Prevents abuse and DoS attacks
- **Privacy Protection**: Validates updates for NaN/Inf values
- **Secure Update Validation**: Multi-layer validation before aggregation

### 📊 Monitoring & Observability

- **Comprehensive Metrics**: Track rounds, clients, updates, and performance
- **Structured Logging**: JSON-formatted logs for easy parsing
- **Round Status Tracking**: Real-time visibility into training rounds
- **Client Reputation**: Track client reliability and participation

### 🎯 Advanced Features

- **Async Round Management**: Flexible aggregation timing with straggler detection
- **Reputation System**: Track and score client reliability
- **Incentive System**: Reward-based participation (tokens, bonuses)
- **Behavior Simulation**: Test scenarios with delays, dropouts, and speed variations
- **Docker Compose**: Easy deployment and scaling

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Coordinator (FastAPI)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Round Manager│  │  Aggregator  │  │Task Assigner│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Auth Mgr   │  │Rate Limiter  │  │   Validator  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Reputation  │  │  Incentives  │  │   Metrics   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ HTTP/REST API
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
┌───────▼──────┐  ┌───────▼──────┐  ┌───────▼──────┐
│   Client 1   │  │   Client 2   │  │   Client N   │
│              │  │              │  │              │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │ Trainer  │ │  │ │ Trainer  │ │  │ │ Trainer  │ │
│ │(PyTorch) │ │  │ │(PyTorch) │ │  │ │(PyTorch) │ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
│ ┌──────────┐ │  │ ┌──────────┐ │  │ ┌──────────┐ │
│ │   API    │ │  │ │   API    │ │  │ │   API    │ │
│ └──────────┘ │  │ └──────────┘ │  │ └──────────┘ │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Coordinator Components

- **RoundManager**: Manages training rounds, client assignments, and round states
- **TaskAssigner**: Assigns training tasks to clients based on model versions
- **UpdateValidator**: Validates client updates (auth, rate limits, value checks)
- **Aggregator**: Aggregates client updates using federated averaging
- **ModelStore**: Persists aggregated models to disk
- **AuthManager**: Manages API key generation and validation
- **RateLimiter**: Enforces rate limits on requests and updates
- **PrivacyProtector**: Validates updates for privacy-violating values
- **MetricsCollector**: Tracks system-wide metrics and performance
- **ReputationManager**: Tracks client reliability and participation history
- **IncentiveManager**: Manages reward-based participation incentives
- **AsyncRoundManager**: Handles async round aggregation with straggler detection

### Client Components

- **client.py**: Main execution loop orchestrating the federated learning workflow
- **trainer.py**: PyTorch-based neural network training (SimpleMLP model)
- **api.py**: HTTP client for coordinator communication
- **security.py**: API key management and authentication
- **behavior.py**: Simulation utilities for testing (delays, dropouts, etc.)
- **config.py**: Configuration management
- **utils/logger.py**: Structured logging utilities

### Round Lifecycle

```
OPEN → COLLECTING → AGGREGATING → CLOSED
  │         │            │           │
  │         │            │           └─ Round complete, model saved
  │         │            └─ Aggregating updates into new model
  │         └─ Clients submitting updates
  └─ Round created, waiting for clients
```

## Installation

### Prerequisites

- **Docker** 20.10+ and **Docker Compose** 2.0+
- **Python** 3.12+ (for local development)
- **Git** (for cloning the repository)

### Docker Installation (Recommended)

See [INSTALL_DOCKER.md](./INSTALL_DOCKER.md) for detailed Docker installation instructions.

### Local Development Setup

#### Coordinator Setup

```bash
cd coordinator
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Client Setup

```bash
cd client
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note**: PyTorch installation may take several minutes depending on your system.

## Quick Start

### Using Docker Compose (Easiest)

```bash
# Clone the repository
git clone <repository-url>
cd open-federated-trainer

# Start coordinator + 1 client
docker compose up --build

# Scale to multiple clients (in a new terminal)
docker compose up --scale client=5
```

### Manual Start (Local Development)

**Terminal 1 - Coordinator:**
```bash
cd coordinator/src
python main.py
# Coordinator runs on http://localhost:8000
```

**Terminal 2 - Client:**
```bash
cd client/src
export COORDINATOR_URL=http://localhost:8000
python client.py
```

### Verify It's Working

1. **Check Coordinator Health:**
   ```bash
   curl http://localhost:8000/
   ```

2. **View API Documentation:**
   Open http://localhost:8000/docs in your browser

3. **Monitor Logs:**
   ```bash
   docker compose logs -f
   ```

For detailed quick start instructions, see [QUICK_START.md](./QUICK_START.md).

## API Documentation

### Coordinator Endpoints

#### Client Registration
```http
POST /client/register
Content-Type: application/json

{
  "client_name": "client-1"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Client client-1 registered successfully. Save your API key!",
  "client_id": "client-1",
  "api_key": "sk_abc123..."
}
```

#### Get Training Task
```http
GET /task/{client_id}?api_key=sk_abc123...
```

**Response:**
```json
{
  "round_id": 1,
  "model_version": "v1",
  "task": "train",
  "description": "Train local model for round 1"
}
```

#### Submit Update
```http
POST /update
Content-Type: application/json

{
  "client_id": "client-1",
  "round_id": 1,
  "weight_delta": "{...}",
  "api_key": "sk_abc123..."
}
```

**Response:**
```json
{
  "success": true,
  "message": "Update from client client-1 submitted successfully for round 1"
}
```

#### Aggregate Round
```http
GET /aggregate/{round_id}
```

**Response:**
```json
{
  "round_id": 1,
  "model_version": "v2",
  "status": "aggregated",
  "aggregated_model": {...},
  "num_updates": 5
}
```

#### Get Round Status
```http
GET /status/{round_id}
```

**Response:**
```json
{
  "round_id": 1,
  "model_version": "v1",
  "state": "COLLECTING",
  "assigned_clients": ["client-1", "client-2"],
  "updates_received": ["client-1"],
  "total_clients": 2,
  "total_updates": 1
}
```

#### Get Model
```http
GET /model/{version}
```

#### Metrics Endpoints
- `GET /metrics` - All metrics
- `GET /metrics/latest` - Latest round metrics
- `GET /metrics/round/{round_id}` - Specific round metrics

#### Reputation Endpoints
- `GET /reputation` - All client reputations
- `GET /reputation/{client_id}` - Specific client reputation

#### Incentive Endpoints
- `GET /incentives` - All client incentives
- `GET /incentives/{client_id}` - Specific client incentives

#### Async Round Stats
- `GET /async/round/{round_id}/stats` - Async round statistics

### Interactive API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Security Features

### Authentication

- **API Key Generation**: Unique API keys generated during client registration
- **Key Validation**: All API requests validated against registered keys
- **Secure Storage**: Clients store API keys in environment variables

### Rate Limiting

- **Request Rate Limits**: Prevents excessive API calls
- **Update Rate Limits**: Limits update submission frequency
- **Per-Client Tracking**: Individual rate limits per client
- **Configurable Thresholds**: Adjustable limits via environment variables

### Privacy Protection

- **Value Validation**: Checks for NaN, Inf, and other invalid values
- **Update Sanitization**: Ensures updates don't leak information
- **Secure Aggregation**: Validates updates before aggregation

### Update Validation Pipeline

1. **Authentication Check**: Verify API key matches client_id
2. **Registration Check**: Ensure client is registered
3. **Round Validation**: Verify client is assigned to the round
4. **Rate Limiting**: Check if client exceeds rate limits
5. **Format Validation**: Ensure weight_delta is valid
6. **Value Validation**: Check for non-finite values (NaN/Inf)

## Advanced Features

### Async Round Management

Enable flexible aggregation timing with straggler detection:

```bash
export ENABLE_ASYNC_ROUNDS=true
export ASYNC_MIN_UPDATES=2
export ASYNC_MAX_DURATION=300.0
```

**Features:**
- Aggregates when minimum updates received OR max duration reached
- Detects and records stragglers (late updates)
- Tracks round start times and latencies

### Reputation System

Tracks client reliability metrics:
- **Update Submission Rate**: How often client submits updates
- **Update Acceptance Rate**: Percentage of updates accepted
- **Round Dropout Rate**: How often client drops out of rounds
- **Consistency Score**: Participation consistency over time

### Incentive System

Reward-based participation:
- **Base Rewards**: Tokens for each accepted update
- **Speed Bonuses**: Extra rewards for fast submissions
- **Consistency Bonuses**: Rewards for consistent participation
- **Dropout Penalties**: Reduced rewards for dropouts

Configure via environment variables:
```bash
export INCENTIVE_BASE_REWARD=10.0
export INCENTIVE_SPEED_THRESHOLD=30.0
export INCENTIVE_CONSISTENCY_THRESHOLD=5
```

### Behavior Simulation

Test various scenarios with client behavior simulation:
- **Startup Delays**: Simulate slow client startup
- **Dropouts**: Random client dropouts during training
- **Training Speed Variations**: Different training speeds per client
- **Coordinator Delays**: Simulate network latency

## Project Structure

```
open-federated-trainer/
├── coordinator/              # Coordinator service
│   ├── src/
│   │   ├── core/            # Core coordinator modules
│   │   │   ├── round_manager.py
│   │   │   ├── aggregator.py
│   │   │   ├── task_assigner.py
│   │   │   ├── update_validator.py
│   │   │   ├── model_store.py
│   │   │   ├── auth.py
│   │   │   ├── rate_limiter.py
│   │   │   ├── privacy.py
│   │   │   ├── metrics.py
│   │   │   ├── reputation.py
│   │   │   ├── incentives.py
│   │   │   └── async_round_manager.py
│   │   ├── utils/           # Utility modules
│   │   │   └── logger.py
│   │   └── main.py          # FastAPI application
│   ├── models/              # Saved model versions
│   ├── Dockerfile
│   └── requirements.txt
│
├── client/                  # Client service
│   ├── src/
│   │   ├── client.py        # Main client loop
│   │   ├── trainer.py       # PyTorch training
│   │   ├── api.py           # Coordinator API client
│   │   ├── security.py      # API key management
│   │   ├── behavior.py      # Behavior simulation
│   │   ├── config.py        # Configuration
│   │   └── utils/
│   │       └── logger.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/                   # Integration tests
│   ├── integration/
│   └── README.md
│
├── docker-compose.yml        # Docker Compose configuration
├── .gitignore
├── README.md                 # This file
├── QUICK_START.md           # Quick start guide
├── SIMULATION.md            # Simulation scenarios
└── INSTALL_DOCKER.md        # Docker installation guide
```

## Development

### Running Tests

```bash
# Integration tests
cd tests
./test_single_round.sh
./test_multi_client.sh
./test_client_dropout.sh
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where possible
- Document functions with docstrings
- Use meaningful variable names

### Adding New Features

1. **Coordinator Features**: Add modules in `coordinator/src/core/`
2. **Client Features**: Add modules in `client/src/`
3. **API Endpoints**: Add to `coordinator/src/main.py`
4. **Tests**: Add integration tests in `tests/`

### Environment Variables

#### Coordinator
- `ENABLE_ASYNC_ROUNDS`: Enable async round management (default: false)
- `ASYNC_MIN_UPDATES`: Minimum updates for async aggregation (default: 2)
- `ASYNC_MAX_DURATION`: Max round duration in seconds (default: 300.0)
- `INCENTIVE_BASE_REWARD`: Base reward per update (default: 10.0)
- `INCENTIVE_SPEED_THRESHOLD`: Speed bonus threshold (default: 30.0)
- `INCENTIVE_CONSISTENCY_THRESHOLD`: Consistency bonus threshold (default: 5)

#### Client
- `COORDINATOR_URL`: Coordinator API URL (default: http://localhost:8000)
- `CLIENT_NAME`: Client identifier (auto-generated if not set)
- `CLIENT_API_KEY`: API key for authentication
- `MAX_RETRIES`: Maximum retry attempts (default: 3)
- `SLEEP_BETWEEN_ROUNDS`: Sleep duration between rounds (default: 5.0)
- `RETRY_DELAY`: Delay between retries (default: 2.0)

## Testing

### Integration Tests

The `tests/` directory contains integration tests:

- `test_single_round.sh`: Test single round completion
- `test_multi_client.sh`: Test multiple clients
- `test_client_dropout.sh`: Test client dropout scenarios
- `test_invalid_update.sh`: Test update validation
- `test_metrics_endpoint.sh`: Test metrics collection

### Running Tests

```bash
cd tests
chmod +x *.sh
./test_single_round.sh
```

### Manual Testing

1. Start coordinator and clients
2. Monitor logs: `docker compose logs -f`
3. Check API: `curl http://localhost:8000/`
4. Verify models: `ls -lh coordinator/models/`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Contribution Guidelines

- Write clear commit messages
- Add tests for new features
- Update documentation
- Follow existing code style
- Ensure all tests pass

## License

[Add your license here]

## Acknowledgments

- Inspired by volunteer computing projects like SETI@home and Folding@home
- Built with [PyTorch](https://pytorch.org/), [FastAPI](https://fastapi.tiangolo.com/), and [Docker](https://www.docker.com/)

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Built with ❤️ for decentralized machine learning**
