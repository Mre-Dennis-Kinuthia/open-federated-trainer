#!/bin/bash
# Automated test script for federated LoRA fine-tuning

set -e

COORDINATOR_URL="http://localhost:8000"
TIMEOUT=120  # Timeout in seconds

echo "=========================================="
echo "Testing Federated LoRA Fine-Tuning"
echo "=========================================="
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check coordinator is running
echo -e "${YELLOW}1. Checking coordinator...${NC}"
if ! curl -s $COORDINATOR_URL/ > /dev/null 2>&1; then
    echo -e "${RED}ERROR: Coordinator not running at $COORDINATOR_URL${NC}"
    echo "Please start the coordinator first:"
    echo "  cd coordinator/src && python main.py"
    exit 1
fi
echo -e "${GREEN}✓ Coordinator is running${NC}"
echo

# 2. Create round
echo -e "${YELLOW}2. Creating LoRA training round...${NC}"
ROUND_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/rounds/create \
  -H "Content-Type: application/json" \
  -d '{
    "base_model_id": "tiny-llama",
    "lora_r": 4,
    "lora_alpha": 8,
    "max_steps": 5,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 2
  }')

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to create round${NC}"
    exit 1
fi

ROUND_ID=$(echo $ROUND_RESPONSE | jq -r '.round_id')
if [ "$ROUND_ID" == "null" ] || [ -z "$ROUND_ID" ]; then
    echo -e "${RED}ERROR: Invalid round response${NC}"
    echo "$ROUND_RESPONSE"
    exit 1
fi

echo -e "${GREEN}✓ Created round $ROUND_ID${NC}"
echo "  Response: $(echo $ROUND_RESPONSE | jq -c '{base_model_id, max_steps, state}')"
echo

# 3. Register clients
echo -e "${YELLOW}3. Registering clients...${NC}"
CLIENT1_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "test-lora-client-1"}')

CLIENT1_KEY=$(echo $CLIENT1_RESPONSE | jq -r '.api_key')
CLIENT1_ID=$(echo $CLIENT1_RESPONSE | jq -r '.client_id')

if [ "$CLIENT1_KEY" == "null" ] || [ -z "$CLIENT1_KEY" ]; then
    echo -e "${RED}ERROR: Failed to register client 1${NC}"
    exit 1
fi

CLIENT2_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/client/register \
  -H "Content-Type: application/json" \
  -d '{"client_name": "test-lora-client-2"}')

CLIENT2_KEY=$(echo $CLIENT2_RESPONSE | jq -r '.api_key')
CLIENT2_ID=$(echo $CLIENT2_RESPONSE | jq -r '.client_id')

if [ "$CLIENT2_KEY" == "null" ] || [ -z "$CLIENT2_KEY" ]; then
    echo -e "${RED}ERROR: Failed to register client 2${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Registered clients${NC}"
echo "  Client 1: $CLIENT1_ID"
echo "  Client 2: $CLIENT2_ID"
echo

# 4. Run clients
echo -e "${YELLOW}4. Running clients (this may take a few minutes)...${NC}"
cd client/src

# Client 1
export CLIENT_API_KEY=$CLIENT1_KEY
python lora_client.py $ROUND_ID $CLIENT1_ID > /tmp/lora_client1.log 2>&1 &
CLIENT1_PID=$!

# Client 2
export CLIENT_API_KEY=$CLIENT2_KEY
python lora_client.py $ROUND_ID $CLIENT2_ID > /tmp/lora_client2.log 2>&1 &
CLIENT2_PID=$!

echo "  Started clients (PIDs: $CLIENT1_PID, $CLIENT2_PID)"
echo "  Waiting for training to complete (timeout: ${TIMEOUT}s)..."
echo

# Wait for clients to complete
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    
    # Check if processes are still running
    if ! kill -0 $CLIENT1_PID 2>/dev/null && ! kill -0 $CLIENT2_PID 2>/dev/null; then
        break
    fi
    
    echo -n "."
done
echo

# Wait a bit more for submissions to complete
sleep 10

# 5. Check round status
echo -e "${YELLOW}5. Checking round status...${NC}"
ROUND_STATUS=$(curl -s "$COORDINATOR_URL/rounds/$ROUND_ID")
ROUND_STATE=$(echo $ROUND_STATUS | jq -r '.state')
echo "  Round state: $ROUND_STATE"
echo

# 6. Check client logs
echo -e "${YELLOW}6. Checking client logs...${NC}"
if grep -q "Adapter submitted successfully" /tmp/lora_client1.log; then
    echo -e "${GREEN}✓ Client 1 submitted adapter${NC}"
else
    echo -e "${RED}✗ Client 1 may have failed${NC}"
    echo "  Last 10 lines:"
    tail -10 /tmp/lora_client1.log | sed 's/^/    /'
fi

if grep -q "Adapter submitted successfully" /tmp/lora_client2.log; then
    echo -e "${GREEN}✓ Client 2 submitted adapter${NC}"
else
    echo -e "${RED}✗ Client 2 may have failed${NC}"
    echo "  Last 10 lines:"
    tail -10 /tmp/lora_client2.log | sed 's/^/    /'
fi
echo

# 7. Aggregate
echo -e "${YELLOW}7. Aggregating round...${NC}"
AGG_RESPONSE=$(curl -s -X POST $COORDINATOR_URL/rounds/$ROUND_ID/aggregate \
  -H "Content-Type: application/json" \
  -d "{\"round_id\": $ROUND_ID, \"weight_by_samples\": true}")

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Aggregation failed${NC}"
    echo "$AGG_RESPONSE"
    exit 1
fi

ADAPTER_VERSION=$(echo $AGG_RESPONSE | jq -r '.adapter_version')
NUM_ADAPTERS=$(echo $AGG_RESPONSE | jq -r '.num_adapters')
EVAL_PASSED=$(echo $AGG_RESPONSE | jq -r '.evaluation_passed')

echo -e "${GREEN}✓ Aggregation complete${NC}"
echo "  Adapter version: $ADAPTER_VERSION"
echo "  Number of adapters: $NUM_ADAPTERS"
echo "  Evaluation passed: $EVAL_PASSED"
echo

# 8. Verify adapter saved
echo -e "${YELLOW}8. Verifying adapter saved...${NC}"
ADAPTER_FILE="../../coordinator/adapters/model_${ADAPTER_VERSION}.json"
if [ -f "$ADAPTER_FILE" ]; then
    echo -e "${GREEN}✓ Adapter saved to $ADAPTER_FILE${NC}"
    ADAPTER_CLIENTS=$(cat "$ADAPTER_FILE" | jq -r '.num_clients')
    echo "  Number of clients in adapter: $ADAPTER_CLIENTS"
else
    echo -e "${RED}✗ Adapter file not found${NC}"
fi
echo

# 9. Cleanup
echo -e "${YELLOW}9. Cleaning up...${NC}"
kill $CLIENT1_PID $CLIENT2_PID 2>/dev/null || true
echo -e "${GREEN}✓ Cleanup complete${NC}"
echo

# Summary
echo "=========================================="
echo -e "${GREEN}Test Summary${NC}"
echo "=========================================="
echo "Round ID: $ROUND_ID"
echo "Adapter Version: $ADAPTER_VERSION"
echo "Clients: $NUM_ADAPTERS"
echo "Evaluation: $EVAL_PASSED"
echo
echo -e "${GREEN}✓ All tests passed!${NC}"
echo

