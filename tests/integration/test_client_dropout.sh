#!/bin/bash
# Integration test: Client dropout simulation
# Validates that the system handles client failures gracefully

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
COORDINATOR_DIR="${COORDINATOR_DIR:-../../coordinator}"
CLIENT_DIR="${CLIENT_DIR:-../../client}"
NUM_CLIENTS=3
TEST_CLIENT_PREFIX="test_client_dropout"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Arrays to track PIDs
CLIENT_PIDS=()

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $COORDINATOR_PID 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    wait $COORDINATOR_PID 2>/dev/null || true
    for pid in "${CLIENT_PIDS[@]}"; do
        wait $pid 2>/dev/null || true
    done
}

trap cleanup EXIT

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Test: Client Dropout Simulation                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Start coordinator
echo "Step 1: Starting coordinator..."
cd "$COORDINATOR_DIR"
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/coordinator_dropout_test.log 2>&1 &
COORDINATOR_PID=$!
cd - > /dev/null

# Wait for coordinator to be ready
echo "Waiting for coordinator to start..."
for i in {1..30}; do
    if curl -s "$COORDINATOR_URL/" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Coordinator is ready${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Coordinator failed to start${NC}"
        exit 1
    fi
    sleep 1
done

# Step 2: Register multiple clients
echo ""
echo "Step 2: Registering $NUM_CLIENTS clients..."
CLIENT_NAMES=()
for i in $(seq 1 $NUM_CLIENTS); do
    CLIENT_NAME="${TEST_CLIENT_PREFIX}_$i"
    CLIENT_NAMES+=("$CLIENT_NAME")
    
    REGISTER_RESPONSE=$(curl -s -X POST "$COORDINATOR_URL/client/register" \
        -H "Content-Type: application/json" \
        -d "{\"client_name\": \"$CLIENT_NAME\"}")
    
    if echo "$REGISTER_RESPONSE" | grep -q '"success":true'; then
        echo -e "${GREEN}✓ Client $i registered: $CLIENT_NAME${NC}"
    else
        echo -e "${RED}✗ Client $i registration failed: $REGISTER_RESPONSE${NC}"
        exit 1
    fi
done

# Step 3: Fetch tasks for all clients
echo ""
echo "Step 3: Fetching tasks for all clients..."
ROUND_IDS=()
for CLIENT_NAME in "${CLIENT_NAMES[@]}"; do
    TASK_RESPONSE=$(curl -s "$COORDINATOR_URL/task/$CLIENT_NAME")
    ROUND_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['round_id'])" 2>/dev/null || echo "")
    
    if [ -z "$ROUND_ID" ]; then
        echo -e "${RED}✗ Failed to get task for $CLIENT_NAME${NC}"
        exit 1
    fi
    
    ROUND_IDS+=("$ROUND_ID")
done

# Get common round ID
COMMON_ROUND_ID="${ROUND_IDS[0]}"
echo -e "${GREEN}✓ All clients assigned to Round $COMMON_ROUND_ID${NC}"

# Step 4: Start all clients
echo ""
echo "Step 4: Starting $NUM_CLIENTS clients..."
cd "$CLIENT_DIR"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENT_NAME="${CLIENT_NAMES[$i]}"
    CLIENT_NAME="$CLIENT_NAME" COORDINATOR_URL="$COORDINATOR_URL" \
        timeout 60 python3 src/client.py > "/tmp/client_dropout_${i}.log" 2>&1 &
    CLIENT_PIDS+=($!)
    echo -e "${GREEN}✓ Started client $((i+1)): $CLIENT_NAME${NC}"
done
cd - > /dev/null

# Step 5: Wait for some updates, then kill one client
echo ""
echo "Step 5: Waiting for initial updates, then simulating dropout..."
sleep 5

# Check current status
STATUS=$(curl -s "$COORDINATOR_URL/status/$COMMON_ROUND_ID" 2>/dev/null || echo "{}")
CURRENT_UPDATES=$(echo "$STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_updates', 0))" 2>/dev/null || echo "0")
echo "  Current updates: $CURRENT_UPDATES"

# Kill the first client (simulate dropout)
if [ ${#CLIENT_PIDS[@]} -gt 0 ]; then
    DROPOUT_CLIENT_NAME="${CLIENT_NAMES[0]}"
    kill "${CLIENT_PIDS[0]}" 2>/dev/null || true
    wait "${CLIENT_PIDS[0]}" 2>/dev/null || true
    echo -e "${YELLOW}⚠ Simulated dropout: $DROPOUT_CLIENT_NAME${NC}"
fi

# Step 6: Wait for remaining clients to submit updates
echo ""
echo "Step 6: Waiting for remaining clients to submit updates..."
EXPECTED_MIN_UPDATES=$((NUM_CLIENTS - 1))  # At least N-1 updates
ALL_REMAINING_UPDATES_RECEIVED=false

for i in {1..60}; do
    STATUS=$(curl -s "$COORDINATOR_URL/status/$COMMON_ROUND_ID" 2>/dev/null || echo "{}")
    TOTAL_UPDATES=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_updates', 0))" 2>/dev/null || echo "0")
    TOTAL_CLIENTS=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_clients', 0))" 2>/dev/null || echo "0")
    
    echo "  Round $COMMON_ROUND_ID: $TOTAL_UPDATES/$TOTAL_CLIENTS updates received"
    
    # Accept if we have at least N-1 updates (partial participation)
    if [ "$TOTAL_UPDATES" -ge "$EXPECTED_MIN_UPDATES" ]; then
        echo -e "${GREEN}✓ At least $EXPECTED_MIN_UPDATES updates received (partial participation OK)${NC}"
        ALL_REMAINING_UPDATES_RECEIVED=true
        break
    fi
    sleep 2
done

if [ "$ALL_REMAINING_UPDATES_RECEIVED" = false ]; then
    echo -e "${RED}✗ Not enough updates received after dropout${NC}"
    exit 1
fi

# Step 7: Trigger aggregation with partial participation
echo ""
echo "Step 7: Triggering aggregation with partial participation..."
AGGREGATE_RESPONSE=$(curl -s "$COORDINATOR_URL/aggregate/$COMMON_ROUND_ID")
AGGREGATE_STATUS=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")
NUM_UPDATES=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('num_updates', 0))" 2>/dev/null || echo "0")

if [ "$AGGREGATE_STATUS" = "aggregated" ] && [ "$NUM_UPDATES" -ge "$EXPECTED_MIN_UPDATES" ]; then
    echo -e "${GREEN}✓ Aggregation completed with $NUM_UPDATES updates (partial participation)${NC}"
else
    echo -e "${RED}✗ Aggregation failed: $AGGREGATE_RESPONSE${NC}"
    exit 1
fi

# Step 8: Verify system recovered (can start new round)
echo ""
echo "Step 8: Verifying system can handle new round after dropout..."
NEW_CLIENT_NAME="${TEST_CLIENT_PREFIX}_recovery"
REGISTER_RESPONSE=$(curl -s -X POST "$COORDINATOR_URL/client/register" \
    -H "Content-Type: application/json" \
    -d "{\"client_name\": \"$NEW_CLIENT_NAME\"}")

if echo "$REGISTER_RESPONSE" | grep -q '"success":true'; then
    TASK_RESPONSE=$(curl -s "$COORDINATOR_URL/task/$NEW_CLIENT_NAME")
    NEW_ROUND_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['round_id'])" 2>/dev/null || echo "")
    
    if [ -n "$NEW_ROUND_ID" ] && [ "$NEW_ROUND_ID" != "$COMMON_ROUND_ID" ]; then
        echo -e "${GREEN}✓ System recovered: New client assigned to Round $NEW_ROUND_ID${NC}"
    else
        echo -e "${YELLOW}⚠ New round assignment unclear${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Could not register recovery client${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ${GREEN}TEST PASSED: Client Dropout Simulation${NC}                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0

