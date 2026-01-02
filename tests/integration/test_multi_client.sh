#!/bin/bash
# Integration test: Multi-client round
# Validates that multiple clients can participate in a federated learning round

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
COORDINATOR_DIR="${COORDINATOR_DIR:-../../coordinator}"
CLIENT_DIR="${CLIENT_DIR:-../../client}"
NUM_CLIENTS=3
TEST_CLIENT_PREFIX="test_client_multi"

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
echo "║  Test: Multi-Client Round                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Start coordinator
echo "Step 1: Starting coordinator..."
cd "$COORDINATOR_DIR"
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/coordinator_multi_test.log 2>&1 &
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
        echo -e "${RED}✗ Failed to get task for $CLIENT_NAME: $TASK_RESPONSE${NC}"
        exit 1
    fi
    
    ROUND_IDS+=("$ROUND_ID")
    echo -e "${GREEN}✓ Task assigned to $CLIENT_NAME: Round $ROUND_ID${NC}"
done

# Verify all clients are in the same round
UNIQUE_ROUNDS=$(printf '%s\n' "${ROUND_IDS[@]}" | sort -u | wc -l)
if [ "$UNIQUE_ROUNDS" -eq 1 ]; then
    COMMON_ROUND_ID="${ROUND_IDS[0]}"
    echo -e "${GREEN}✓ All clients assigned to Round $COMMON_ROUND_ID${NC}"
else
    echo -e "${YELLOW}⚠ Clients assigned to different rounds (may be expected)${NC}"
    COMMON_ROUND_ID="${ROUND_IDS[0]}"
fi

# Step 4: Start all clients
echo ""
echo "Step 4: Starting $NUM_CLIENTS clients..."
cd "$CLIENT_DIR"
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    CLIENT_NAME="${CLIENT_NAMES[$i]}"
    CLIENT_NAME="$CLIENT_NAME" COORDINATOR_URL="$COORDINATOR_URL" \
        timeout 60 python3 src/client.py > "/tmp/client_multi_${i}.log" 2>&1 &
    CLIENT_PIDS+=($!)
    echo -e "${GREEN}✓ Started client $((i+1)): $CLIENT_NAME${NC}"
done
cd - > /dev/null

# Step 5: Wait for all clients to submit updates
echo ""
echo "Step 5: Waiting for all clients to submit updates..."
EXPECTED_UPDATES=$NUM_CLIENTS
ALL_UPDATES_RECEIVED=false

for i in {1..60}; do
    STATUS=$(curl -s "$COORDINATOR_URL/status/$COMMON_ROUND_ID" 2>/dev/null || echo "{}")
    TOTAL_UPDATES=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_updates', 0))" 2>/dev/null || echo "0")
    TOTAL_CLIENTS=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_clients', 0))" 2>/dev/null || echo "0")
    
    echo "  Round $COMMON_ROUND_ID: $TOTAL_UPDATES/$TOTAL_CLIENTS updates received"
    
    if [ "$TOTAL_UPDATES" -ge "$EXPECTED_UPDATES" ] && [ "$TOTAL_CLIENTS" -ge "$EXPECTED_UPDATES" ]; then
        echo -e "${GREEN}✓ All $EXPECTED_UPDATES updates received${NC}"
        ALL_UPDATES_RECEIVED=true
        break
    fi
    sleep 2
done

if [ "$ALL_UPDATES_RECEIVED" = false ]; then
    echo -e "${RED}✗ Not all clients submitted updates${NC}"
    echo "Coordinator log:"
    tail -30 /tmp/coordinator_multi_test.log
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        echo "Client $((i+1)) log:"
        tail -20 "/tmp/client_multi_${i}.log"
    done
    exit 1
fi

# Step 6: Trigger aggregation
echo ""
echo "Step 6: Triggering aggregation..."
AGGREGATE_RESPONSE=$(curl -s "$COORDINATOR_URL/aggregate/$COMMON_ROUND_ID")
AGGREGATE_STATUS=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")
NUM_UPDATES=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('num_updates', 0))" 2>/dev/null || echo "0")

if [ "$AGGREGATE_STATUS" = "aggregated" ] && [ "$NUM_UPDATES" -eq "$EXPECTED_UPDATES" ]; then
    echo -e "${GREEN}✓ Aggregation completed with $NUM_UPDATES updates${NC}"
else
    echo -e "${RED}✗ Aggregation failed or wrong update count: $AGGREGATE_RESPONSE${NC}"
    exit 1
fi

# Step 7: Verify metrics
echo ""
echo "Step 7: Verifying metrics..."
METRICS=$(curl -s "$COORDINATOR_URL/metrics/round/$COMMON_ROUND_ID")
if [ -n "$METRICS" ] && [ "$METRICS" != "{}" ]; then
    UPDATES_RECEIVED=$(echo "$METRICS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('updates_received', 0))" 2>/dev/null || echo "0")
    CLIENTS_ASSIGNED=$(echo "$METRICS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('clients_assigned', 0))" 2>/dev/null || echo "0")
    
    if [ "$UPDATES_RECEIVED" -ge "$EXPECTED_UPDATES" ] && [ "$CLIENTS_ASSIGNED" -ge "$EXPECTED_UPDATES" ]; then
        echo -e "${GREEN}✓ Metrics correct: $UPDATES_RECEIVED updates, $CLIENTS_ASSIGNED clients${NC}"
    else
        echo -e "${RED}✗ Metrics incorrect: expected $EXPECTED_UPDATES, got $UPDATES_RECEIVED updates${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ Metrics not found for round $COMMON_ROUND_ID${NC}"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ${GREEN}TEST PASSED: Multi-Client Round${NC}                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0

