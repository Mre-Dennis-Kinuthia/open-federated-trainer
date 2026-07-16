# Federated LoRA Fine-Tuning Guide

This document describes how to use the federated LoRA fine-tuning extension.

## Overview

The system extends the base federated learning platform to support fine-tuning Large Language Models (LLMs) using LoRA (Low-Rank Adaptation) adapters. Only adapter weights are trained and shared - base model weights remain fixed.

## Architecture

### Coordinator Extensions

- **Model Registry** (`coordinator/src/model_registry/`): Manages base LLM models
- **Rounds Module** (`coordinator/src/rounds/`): Manages LoRA training rounds
- **Aggregation** (`coordinator/src/aggregation/`): Federated averaging for adapters
- **Evaluation** (`coordinator/src/evaluation/`): Evaluates aggregated adapters

### Client Extensions

- **Training** (`client/src/training/`): LoRA training using PyTorch and PEFT
- **Submit** (`client/src/submit/`): Adapter submission to coordinator
- **LoRA Client** (`client/src/lora_client.py`): Main client loop for LoRA training

## Quick Start

### 1. Start Coordinator

```bash
cd coordinator/src
python main.py
```

### 2. Create a LoRA Training Round

```bash
curl -X POST http://localhost:8000/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "lora_r": 8,
    "lora_alpha": 16,
    "max_steps": 50,
    "learning_rate": 2e-4
  }'
```

Response:
```json
{
  "round_id": 1,
  "base_model_id": "tiny-llama",
  "adapter_version": null,
  "lora_r": 8,
  "lora_alpha": 16,
  "max_steps": 50,
  "state": "OPEN",
  ...
}
```

### 3. Run Clients

**Terminal 1 - Client 1:**
```bash
cd client/src
export CLIENT_API_KEY="your_api_key_here"  # Get from registration
python lora_client.py 1 client-1
```

**Terminal 2 - Client 2:**
```bash
cd client/src
export CLIENT_API_KEY="your_api_key_here"
python lora_client.py 1 client-2
```

### 4. Aggregate Round

After clients submit adapters:

```bash
curl -X POST http://localhost:8000/rounds/1/aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": 1,
    "weight_by_samples": true
  }'
```

## API Endpoints

### POST /rounds/create

Create a new LoRA fine-tuning round.

**Request:**
```json
{
  "base_model_id": "tiny-llama",
  "adapter_version": null,
  "lora_r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.1,
  "target_modules": ["q_proj", "v_proj"],
  "max_steps": 100,
  "learning_rate": 2e-4,
  "batch_size": 4,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 10,
  "max_seq_length": 512
}
```

### GET /rounds/{round_id}

Get round configuration.

### POST /rounds/{round_id}/submit

Submit LoRA adapter.

**Request:**
```json
{
  "client_id": "client-1",
  "round_id": 1,
  "adapter_state_dict": {...},
  "num_samples": 100,
  "training_loss": 0.123,
  "api_key": "sk_..."
}
```

### POST /rounds/{round_id}/aggregate

Aggregate adapters using FedAvg.

**Request:**
```json
{
  "round_id": 1,
  "weight_by_samples": true
}
```

## Supported Base Models

- `tiny-llama`: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- `phi-2`: microsoft/phi-2
- `mistral-7b`: mistralai/Mistral-7B-v0.1
- `llama-7b`: meta-llama/Llama-2-7b-hf

## Example: Complete Training Round

```bash
# 1. Create round
ROUND_ID=$(curl -s -X POST http://localhost:8000/rounds/create \
  -H "Content-Type: application/json" \
  -d '{"base_model_id": "tiny-llama", "max_steps": 50}' \
  | jq -r '.round_id')

# 2. Run clients (in separate terminals)
python lora_client.py $ROUND_ID client-1 &
python lora_client.py $ROUND_ID client-2 &

# 3. Wait for submissions, then aggregate
curl -X POST http://localhost:8000/rounds/$ROUND_ID/aggregate \
  -H "Content-Type: application/json" \
  -d '{"round_id": '$ROUND_ID', "weight_by_samples": true}'
```

## Notes

- Base models are loaded in 4-bit mode when possible (requires bitsandbytes)
- Falls back to CPU/FP32 if 4-bit loading fails
- Only LoRA adapter parameters are trained and shared
- Adapters are validated for NaN/Inf before aggregation
- FedAvg weights by number of samples by default

## Future Enhancements

- Download previous adapters for incremental training
- Support for more model architectures
- Real dataset loading through `DATASET_PATH` (CSV/JSONL/JSON/folder/Hugging Face)
- Full evaluation on held-out data
- Secure aggregation with differential privacy

