# Quick Start Guide: Federated LoRA Fine-Tuning

This guide will help you get started with the federated LoRA fine-tuning system.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Starting the System](#starting-the-system)
- [Creating a Training Round](#creating-a-training-round)
- [Running Clients](#running-clients)
- [Aggregating Results](#aggregating-results)
- [API Reference](#api-reference)
- [Example Workflow](#example-workflow)
- [Troubleshooting](#troubleshooting)

## Overview

The federated LoRA fine-tuning system allows multiple clients to collaboratively fine-tune Large Language Models (LLMs) using LoRA adapters without sharing raw data. Only adapter weights are shared with the coordinator.

**Key Features:**
- Base model weights remain fixed (downloaded from HuggingFace)
- Only LoRA adapter parameters are trained and shared
- Federated averaging aggregates adapters from multiple clients
- Supports 4-bit quantization for memory efficiency
- Asynchronous client participation

## Prerequisites

- **Python 3.12+**
- **pip** (Python package manager)
- **Internet connection** (for downloading models from HuggingFace)
- **At least 4GB RAM** (8GB+ recommended)
- **Optional: CUDA GPU** (for faster training, but CPU works too)

## Installation

### Step 1: Clone/Navigate to Project

```bash
cd /home/nansi/Work/open-federated-trainer
```

### Step 2: Install Coordinator Dependencies

```bash
cd coordinator
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** This installs FastAPI, PyTorch, Transformers, and PEFT. May take 5-10 minutes.

### Step 3: Install Client Dependencies

```bash
cd ../client
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Note:** This installs PyTorch, Transformers, PEFT, and BitsAndBytes. May take 5-10 minutes.

### Step 4: Verify Installation

```bash
# Check coordinator
cd coordinator
source venv/bin/activate
python -c "import fastapi, torch, transformers, peft; print('✓ Coordinator dependencies OK')"

# Check client
cd ../client
source venv/bin/activate
python -c "import torch, transformers, peft, bitsandbytes; print('✓ Client dependencies OK')"
```

## Starting the System

### Start Coordinator

**Terminal 1:**
```bash
cd coordinator
./run.sh
```

Or manually:
```bash
cd coordinator
source venv/bin/activate
cd src
python main.py
```

**Expected Output:**
```
INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify it's running:**
```bash
curl http://localhost:8000/
```

You should see a JSON response with API information.

### Check Available Endpoints

Visit http://localhost:8000/docs in your browser for interactive API documentation.

## Creating a Training Round

### Using cURL

**Terminal 2:**
```bash
curl -X POST http://localhost:8000/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "lora_r": 8,
    "lora_alpha": 16,
    "max_steps": 50,
    "learning_rate": 2e-4,
    "batch_size": 4
  }'
```

**Response:**
```json
{
  "round_id": 1,
  "base_model_id": "tiny-llama",
  "adapter_version": null,
  "lora_r": 8,
  "lora_alpha": 16,
  "max_steps": 50,
  "state": "OPEN",
  "created_at": "2025-01-02T..."
}
```

**Save the `round_id`** - you'll need it for clients!

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/rounds/create",
    json={
        "base_model_id": "tiny-llama",
        "max_steps": 50,
        "batch_size": 4
    }
)

round_data = response.json()
round_id = round_data["round_id"]
print(f"Created round {round_id}")
```

### Round Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_id` | Required | Model ID: `tiny-llama`, `phi-2`, `mistral-7b`, `llama-7b` |
| `lora_r` | 8 | LoRA rank (lower = fewer parameters) |
| `lora_alpha` | 16 | LoRA alpha (scaling factor) |
| `lora_dropout` | 0.1 | Dropout rate for LoRA layers |
| `target_modules` | `["q_proj", "v_proj"]` | Modules to apply LoRA to |
| `max_steps` | 100 | Maximum training steps per client |
| `learning_rate` | 2e-4 | Learning rate for training |
| `batch_size` | 4 | Batch size |
| `gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `warmup_steps` | 10 | Number of warmup steps |
| `max_seq_length` | 512 | Maximum sequence length |

## Running Clients

### Step 1: Register Clients (Get API Keys)

**Terminal 2:**
```bash
# Register client 1
CLIENT1_RESPONSE=$(curl -s -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "lora-client-1"}')

CLIENT1_API_KEY=$(echo $CLIENT1_RESPONSE | jq -r '.api_key')
echo "Client 1 API Key: $CLIENT1_API_KEY"

# Register client 2
CLIENT2_RESPONSE=$(curl -s -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "lora-client-2"}')

CLIENT2_API_KEY=$(echo $CLIENT2_RESPONSE | jq -r '.api_key')
echo "Client 2 API Key: $CLIENT2_API_KEY"
```

### Step 2: Run Clients

**Terminal 3 (Client 1):**
```bash
cd client
export CLIENT_API_KEY="<paste_CLIENT1_API_KEY_here>"
./run_lora.sh 1 lora-client-1
```

**Terminal 4 (Client 2):**
```bash
cd client
export CLIENT_API_KEY="<paste_CLIENT2_API_KEY_here>"
./run_lora.sh 1 lora-client-2
```

**Note:** Replace `1` with your actual `round_id` from step 2.

### What Happens During Training

1. **Client fetches round configuration** from coordinator
2. **Downloads base model** from HuggingFace (first time only, ~5-10 min)
3. **Initializes LoRA adapters** (fresh or from previous round)
4. **Loads local dataset** (synthetic data for MVP)
5. **Trains adapter** for specified number of steps
6. **Submits adapter weights** to coordinator

**Expected Client Output:**
```
[Client lora-client-1] Starting LoRA training for round 1...
[Client lora-client-1] Fetching round configuration...
[Client lora-client-1] Round config received: base_model=TinyLlama/..., max_steps=50
[Client lora-client-1] Loading local dataset...
[Client lora-client-1] Loaded 100 training samples
[Client lora-client-1] Starting LoRA training...
[Client lora-client-1] Training completed in 45.23s
[Client lora-client-1] Final loss: 2.3456
[Client lora-client-1] Submitting adapter to coordinator...
[Client lora-client-1] Adapter submitted successfully
```

## Aggregating Results

### Check Round Status

```bash
curl http://localhost:8000/rounds/1 | jq
```

### Aggregate Round

**Terminal 2:**
```bash
curl -X POST http://localhost:8000/rounds/1/aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": 1,
    "weight_by_samples": true
  }' | jq
```

**Response:**
```json
{
  "round_id": 1,
  "adapter_version": "v1",
  "status": "aggregated",
  "num_adapters": 2,
  "evaluation_passed": true,
  "evaluation_loss": 2.1234
}
```

### Verify Adapter Saved

```bash
ls -lh coordinator/adapters/
cat coordinator/adapters/model_v1.json | jq '.version, .num_clients'
```

## API Reference

### Coordinator Endpoints

#### `POST /rounds/create`
Create a new LoRA training round.

**Request:**
```json
{
  "base_model_id": "tiny-llama",
  "max_steps": 50
}
```

**Response:**
```json
{
  "round_id": 1,
  "base_model_id": "tiny-llama",
  "state": "OPEN",
  ...
}
```

#### `GET /rounds/{round_id}`
Get round configuration.

**Response:**
```json
{
  "round_id": 1,
  "base_model_id": "tiny-llama",
  "max_steps": 50,
  "state": "COLLECTING",
  ...
}
```

#### `POST /rounds/{round_id}/submit`
Submit LoRA adapter (used by clients).

**Request:**
```json
{
  "client_id": "client-1",
  "round_id": 1,
  "adapter_state_dict": {...},
  "num_samples": 100,
  "training_loss": 2.3456,
  "api_key": "sk_..."
}
```

#### `POST /rounds/{round_id}/aggregate`
Aggregate adapters using FedAvg.

**Request:**
```json
{
  "round_id": 1,
  "weight_by_samples": true
}
```

**Response:**
```json
{
  "round_id": 1,
  "adapter_version": "v1",
  "status": "aggregated",
  "num_adapters": 2,
  "evaluation_passed": true
}
```

### Client Registration

#### `POST /client/register`
Register a new client and receive API key.

**Request:**
```json
{
  "client_name": "my-client"
}
```

**Response:**
```json
{
  "success": true,
  "client_id": "my-client",
  "api_key": "sk_abc123...",
  "message": "Client registered successfully"
}
```

## Example Workflow

### Complete Example: 2 Clients Training

**Terminal 1 - Coordinator:**
```bash
cd coordinator
./run.sh
```

**Terminal 2 - Setup:**
```bash
# Create round
ROUND_RESPONSE=$(curl -s -X POST http://localhost:8000/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "max_steps": 10,
    "batch_size": 2
  }')

ROUND_ID=$(echo $ROUND_RESPONSE | jq -r '.round_id')
echo "Round ID: $ROUND_ID"

# Register clients
CLIENT1_KEY=$(curl -s -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "client-1"}' | jq -r '.api_key')

CLIENT2_KEY=$(curl -s -X POST http://localhost:8000/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "client-2"}' | jq -r '.api_key')

echo "Client 1 Key: $CLIENT1_KEY"
echo "Client 2 Key: $CLIENT2_KEY"
```

**Terminal 3 - Client 1:**
```bash
cd client
export CLIENT_API_KEY=$CLIENT1_KEY
./run_lora.sh $ROUND_ID client-1
```

**Terminal 4 - Client 2:**
```bash
cd client
export CLIENT_API_KEY=$CLIENT2_KEY
./run_lora.sh $ROUND_ID client-2
```

**Terminal 2 - Aggregate:**
```bash
# Wait for both clients to finish, then:
curl -X POST http://localhost:8000/rounds/$ROUND_ID/aggregate \
  -H "Content-Type: application/json" \
  -d "{\"round_id\": $ROUND_ID}" | jq
```

## Using the Automated Test Script

For quick testing, use the automated script:

```bash
# Make sure coordinator is running in another terminal
cd /home/nansi/Work/open-federated-trainer
./test_lora.sh
```

This script:
- Creates a round
- Registers 2 clients
- Runs training
- Aggregates results
- Verifies everything worked

## Supported Base Models

| Model ID | HuggingFace Model | Size | Notes |
|----------|-------------------|------|-------|
| `tiny-llama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Fastest, good for testing |
| `phi-2` | microsoft/phi-2 | 2.7B | Good balance |
| `mistral-7b` | mistralai/Mistral-7B-v0.1 | 7B | Requires more memory |
| `llama-7b` | meta-llama/Llama-2-7b-hf | 7B | Requires HuggingFace access |

**Recommendation:** Start with `tiny-llama` for testing.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Solution:**
```bash
cd coordinator
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: "Model download fails"

**Solutions:**
- Check internet connection
- Verify model name is correct
- Try a different model (e.g., `tiny-llama`)
- Check HuggingFace access (some models require authentication)

### Issue: "Out of memory"

**Solutions:**
- Use smaller model (`tiny-llama`)
- Reduce `batch_size` (e.g., `batch_size: 1`)
- Reduce `max_steps` (e.g., `max_steps: 10`)
- Enable 4-bit quantization (automatic if bitsandbytes available)
- Use CPU mode (slower but uses less memory)

### Issue: "Client can't connect to coordinator"

**Solutions:**
- Verify coordinator is running: `curl http://localhost:8000/`
- Check `COORDINATOR_URL` environment variable
- Check firewall settings
- Verify network connectivity

### Issue: "Adapter validation fails"

**Solutions:**
- Check coordinator logs for specific error
- Verify adapter structure matches expected format
- Check for NaN/Inf values in adapter weights
- Ensure all clients use same LoRA configuration

### Issue: "Aggregation fails"

**Solutions:**
- Verify all adapters have same parameter keys
- Check that at least one adapter was submitted
- Review aggregation logs
- Ensure round is not already closed

### Issue: "bitsandbytes not available"

**Solution:**
- On Linux: `pip install bitsandbytes`
- On macOS/Windows: System will fall back to CPU/FP32 mode automatically
- Training will be slower but will work

## Next Steps

After successful testing:

1. **Increase training steps** for better convergence
2. **Add more clients** (3-5 clients)
3. **Try different models** (phi-2, mistral-7b)
4. **Use real datasets** (modify `dataset_loader.py`)
5. **Test incremental training** (set `adapter_version` in round creation)

## Additional Resources

- **Full Documentation:** See `README.md` for system overview
- **White Paper:** See `WHITEPAPER.md` for technical details
- **LoRA Guide:** See `LORA_FEDERATED_LEARNING.md` for LoRA-specific details
- **Testing Guide:** See `TESTING_LORA.md` for detailed testing procedures

## Getting Help

- Check logs: Coordinator and client logs show detailed error messages
- API Documentation: Visit http://localhost:8000/docs for interactive API docs
- Review code: All modules have docstrings explaining functionality

---

**Happy Federated Learning! 🚀**
