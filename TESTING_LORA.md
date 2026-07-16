# Testing Federated LoRA Fine-Tuning

This guide walks you through testing the LoRA federated learning extension.

## Prerequisites

### 1. Install Dependencies

**Coordinator:**
```bash
cd coordinator
pip install -r requirements.txt
```

**Client:**
```bash
cd client
pip install -r requirements.txt
```

**Note:** Installing `bitsandbytes` and `transformers` may take several minutes.

### 2. Verify Installation

```bash
# Check coordinator dependencies
python -c "import fastapi, torch, transformers, peft; print('✓ All dependencies installed')"

# Check client dependencies
python -c "import torch, transformers, peft, bitsandbytes; print('✓ All dependencies installed')"
```

## Quick Test (2 Clients)

### Step 1: Start Coordinator

**Terminal 1:**
```bash
cd coordinator/src
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Verify it's running:
```bash
curl http://localhost:8000/
```

### Step 2: Create a LoRA Training Round

**Terminal 2:**
```bash
# Create a round with minimal steps for quick testing
curl -X POST http://localhost:8000/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "lora_r": 4,
    "lora_alpha": 8,
    "max_steps": 10,
    "learning_rate": 2e-4,
    "batch_size": 2
  }' | jq
```

**Expected Response:**
```json
{
  "round_id": 1,
  "base_model_id": "tiny-llama",
  "state": "OPEN",
  "max_steps": 10,
  ...
}
```

**Save the round_id** (e.g., `ROUND_ID=1`)

### Step 3: Register Clients and Get API Keys

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

### Step 4: Run Clients

**Terminal 3 (Client 1):**
```bash
cd client/src
export CLIENT_API_KEY="<CLIENT1_API_KEY>"
python lora_client.py 1 lora-client-1
```

**Terminal 4 (Client 2):**
```bash
cd client/src
export CLIENT_API_KEY="<CLIENT2_API_KEY>"
python lora_client.py 1 lora-client-2
```

**Expected Output (Client):**
```
[Client lora-client-1] Starting LoRA training for round 1...
[Client lora-client-1] Fetching round configuration...
[Client lora-client-1] Round config received: base_model=TinyLlama/TinyLlama-1.1B-Chat-v1.0, max_steps=10
[Client lora-client-1] Loading local dataset...
[Client lora-client-1] Loaded 100 training samples
[Client lora-client-1] Starting LoRA training...
[Client lora-client-1] Training completed in X.XXs
[Client lora-client-1] Final loss: X.XXXX
[Client lora-client-1] Submitting adapter to coordinator...
[Client lora-client-1] Adapter submitted successfully
```

### Step 5: Verify Submissions

**Terminal 2:**
```bash
# Check round status (should show 2 submissions)
curl http://localhost:8000/rounds/1 | jq '.state'
```

### Step 6: Aggregate Round

**Terminal 2:**
```bash
curl -X POST http://localhost:8000/rounds/1/aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "round_id": 1,
    "weight_by_samples": true
  }' | jq
```

**Expected Response:**
```json
{
  "round_id": 1,
  "adapter_version": "v1",
  "status": "aggregated",
  "num_adapters": 2,
  "evaluation_passed": true,
  "evaluation_loss": 0.XXX
}
```

### Step 7: Verify Adapter Saved

**Terminal 2:**
```bash
# Check if adapter was saved
ls -lh coordinator/adapters/
cat coordinator/adapters/model_v1.json | jq '.version, .num_clients'
```

## Automated Test Script

Create a test script for easier testing:

```bash
#!/bin/bash
# test_lora.sh

set -e

COORDINATOR_URL="http://localhost:8000"

echo "=== Testing Federated LoRA Fine-Tuning ==="
echo

# 1. Check coordinator is running
echo "1. Checking coordinator..."
if ! curl -s $COORDINATOR_URL/ > /dev/null; then
    echo "ERROR: Coordinator not running at $COORDINATOR_URL"
    exit 1
fi
echo "✓ Coordinator is running"
echo

# 2. Create round
echo "2. Creating LoRA training round..."
ROUND_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "lora_r": 4,
    "max_steps": 5,
    "batch_size": 2
  }')

ROUND_ID=$(echo $ROUND_RESPONSE | jq -r '.round_id')
echo "✓ Created round $ROUND_ID"
echo

# 3. Register clients
echo "3. Registering clients..."
CLIENT1_KEY=$(curl -s -X POST $COORDINATOR_URL/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "test-client-1"}' | jq -r '.api_key')

CLIENT2_KEY=$(curl -s -X POST $COORDINATOR_URL/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "test-client-2"}' | jq -r '.api_key')

echo "✓ Registered clients"
echo

# 4. Run clients (in background)
echo "4. Running clients..."
cd client/src
export CLIENT_API_KEY=$CLIENT1_KEY
python lora_client.py $ROUND_ID test-client-1 > /tmp/client1.log 2>&1 &
CLIENT1_PID=$!

export CLIENT_API_KEY=$CLIENT2_KEY
python lora_client.py $ROUND_ID test-client-2 > /tmp/client2.log 2>&1 &
CLIENT2_PID=$!

echo "✓ Clients started (PIDs: $CLIENT1_PID, $CLIENT2_PID)"
echo "  Waiting for training to complete..."
sleep 60  # Wait for training

# 5. Check submissions
echo "5. Checking submissions..."
SUBMISSIONS=$(curl -s $COORDINATOR_URL/rounds/$ROUND_ID | jq -r '.state')
echo "  Round state: $SUBMISSIONS"
echo

# 6. Aggregate
echo "6. Aggregating round..."
AGG_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/rounds/$ROUND_ID/aggregate \
  -H "Content-Type: application/json" \
  -d "{\"round_id\": $ROUND_ID}")

ADAPTER_VERSION=$(echo $AGG_RESPONSE | jq -r '.adapter_version')
NUM_ADAPTERS=$(echo $AGG_RESPONSE | jq -r '.num_adapters')

echo "✓ Aggregation complete"
echo "  Adapter version: $ADAPTER_VERSION"
echo "  Number of adapters: $NUM_ADAPTERS"
echo

# 7. Cleanup
echo "7. Cleaning up..."
kill $CLIENT1_PID $CLIENT2_PID 2>/dev/null || true
echo "✓ Test complete!"
```

Make it executable:
```bash
chmod +x test_lora.sh
./test_lora.sh
```

## Testing Checklist

- [ ] Coordinator starts successfully
- [ ] Can create LoRA round
- [ ] Can register clients
- [ ] Clients can fetch round config
- [ ] Clients can load base model
- [ ] Clients can train LoRA adapters
- [ ] Clients can submit adapters
- [ ] Coordinator validates adapters
- [ ] Aggregation works correctly
- [ ] Adapter is saved to disk
- [ ] Evaluation passes

## Troubleshooting

### Issue: "transformers not available"

**Solution:**
```bash
pip install transformers>=4.35.0 peft>=0.7.0
```

### Issue: "bitsandbytes not available"

**Solution:**
```bash
# On Linux
pip install bitsandbytes

# On macOS/Windows, may need to skip 4-bit mode
# Edit lora_trainer.py to set use_4bit=False
```

### Issue: "Model download fails"

**Solution:**
- Check internet connection
- Verify HuggingFace model name is correct
- Try a smaller model (tiny-llama) first

### Issue: "Out of memory"

**Solution:**
- Use smaller model (tiny-llama)
- Reduce batch_size and max_steps
- Enable 4-bit quantization
- Use CPU mode if GPU unavailable

### Issue: "Adapter validation fails"

**Solution:**
- Check for NaN/Inf in adapter weights
- Verify adapter structure matches expected format
- Check coordinator logs for specific error

### Issue: "Aggregation fails"

**Solution:**
- Verify all adapters have same parameter keys
- Check adapter submissions are valid
- Review aggregation logs

## Manual Testing Steps

1. **Test Model Registry:**
   ```bash
   curl http://localhost:8000/ | jq '.endpoints'
   ```

2. **Test Round Creation:**
   ```bash
   curl -X POST http://localhost:8000/rounds/create \
     -H "Content-Type: application/json" \
     -d '{"base_model_id": "tiny-llama"}' | jq
   ```

3. **Test Round Retrieval:**
   ```bash
   curl http://localhost:8000/rounds/1 | jq
   ```

4. **Test Client Training:**
   ```bash
   cd client/src
   python lora_client.py 1 test-client
   ```

5. **Test Aggregation:**
   ```bash
   curl -X POST http://localhost:8000/rounds/1/aggregate \
     -H "Content-Type: application/json" \
     -d '{"round_id": 1}' | jq
   ```

## Expected File Structure After Test

```
coordinator/
├── adapters/
│   └── model_v1.json          # Aggregated adapter
└── src/
    └── main.py

client/
└── src/
    ├── lora_client.py
    └── training/
        └── lora_trainer.py
```

## Performance Notes

- **First run:** Model download may take 5-10 minutes
- **Training time:** ~1-5 minutes per client (depends on hardware)
- **Memory:** ~2-4GB RAM for tiny-llama, more for larger models

## Next Steps

After successful testing:
1. Try with more clients (3-5)
2. Increase max_steps for better convergence
3. Test with different base models
4. Test incremental training (adapter_version set)

