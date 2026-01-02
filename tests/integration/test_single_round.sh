#!/bin/bash
# Integration test: Single client round
# Validates that one client can complete a full federated learning round

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
COORDINATOR_DIR="${COORDINATOR_DIR:-../../coordinator}"
CLIENT_DIR="${CLIENT_DIR:-../../client}"
TEST_CLIENT_NAME="test_client_single"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $COORDINATOR_PID 2>/dev/null || true
    kill $CLIENT_PID 2>/dev/null || true
    wait $COORDINATOR_PID 2>/dev/null || true
    wait $CLIENT_PID 2>/dev/null || true
}

trap cleanup EXIT

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Test: Single Client Round                                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Start coordinator
echo "Step 1: Starting coordinator..."
cd "$COORDINATOR_DIR"
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/coordinator_test.log 2>&1 &
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

# Step 2: Register client
echo ""
echo "Step 2: Registering client..."
REGISTER_RESPONSE=$(curl -s -X POST "$COORDINATOR_URL/client/register" \
    -H "Content-Type: application/json" \
    -d "{\"client_name\": \"$TEST_CLIENT_NAME\"}")

if echo "$REGISTER_RESPONSE" | grep -q '"success":true'; then
    echo -e "${GREEN}✓ Client registered successfully${NC}"
else
    echo -e "${RED}✗ Client registration failed: $REGISTER_RESPONSE${NC}"
    exit 1
fi

# Step 3: Fetch task
echo ""
echo "Step 3: Fetching task..."
TASK_RESPONSE=$(curl -s "$COORDINATOR_URL/task/$TEST_CLIENT_NAME")
ROUND_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['round_id'])" 2>/dev/null || echo "")

if [ -z "$ROUND_ID" ]; then
    echo -e "${RED}✗ Failed to get task: $TASK_RESPONSE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Task assigned: Round $ROUND_ID${NC}"

# Step 4: Start client training (simulate)
echo ""
echo "Step 4: Simulating client training..."
cd "$CLIENT_DIR"
CLIENT_NAME="$TEST_CLIENT_NAME" COORDINATOR_URL="$COORDINATOR_URL" \
    timeout 30 python3 src/client.py > /tmp/client_test.log 2>&1 &
CLIENT_PID=$!
cd - > /dev/null

# Wait for client to submit update
echo "Waiting for client to submit update..."
UPDATE_RECEIVED=false
for i in {1..30}; do
    STATUS=$(curl -s "$COORDINATOR_URL/status/$ROUND_ID" 2>/dev/null || echo "{}")
    TOTAL_UPDATES=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_updates', 0))" 2>/dev/null || echo "0")
    
    if [ "$TOTAL_UPDATES" -ge 1 ]; then
        echo -e "${GREEN}✓ Update received (total: $TOTAL_UPDATES)${NC}"
        UPDATE_RECEIVED=true
        break
    fi
    sleep 1
done

if [ "$UPDATE_RECEIVED" = false ]; then
    echo -e "${RED}✗ Client did not submit update${NC}"
    echo "Coordinator log:"
    tail -20 /tmp/coordinator_test.log
    echo "Client log:"
    tail -20 /tmp/client_test.log
    exit 1
fi

# Step 5: Trigger aggregation
echo ""
echo "Step 5: Triggering aggregation..."
AGGREGATE_RESPONSE=$(curl -s "$COORDINATOR_URL/aggregate/$ROUND_ID")
AGGREGATE_STATUS=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")

if [ "$AGGREGATE_STATUS" = "aggregated" ]; then
    echo -e "${GREEN}✓ Aggregation completed${NC}"
else
    echo -e "${RED}✗ Aggregation failed: $AGGREGATE_RESPONSE${NC}"
    exit 1
fi

# Step 6: Verify metrics
echo ""
echo "Step 6: Verifying metrics..."
METRICS=$(curl -s "$COORDINATOR_URL/metrics/round/$ROUND_ID")
if [ -n "$METRICS" ] && [ "$METRICS" != "{}" ]; then
    ROUND_METRICS=$(echo "$METRICS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(f\"Round {d.get('round_id')}: {d.get('updates_received', 0)} updates, {d.get('clients_assigned', 0)} clients\")" 2>/dev/null || echo "")
    echo -e "${GREEN}✓ Metrics available: $ROUND_METRICS${NC}"
else
    echo -e "${RED}✗ Metrics not found for round $ROUND_ID${NC}"
    exit 1
fi

# Step 7: Verify metrics file exists
echo ""
echo "Step 7: Verifying metrics persistence..."
if [ -f "$COORDINATOR_DIR/metrics/round_${ROUND_ID}.json" ]; then
    echo -e "${GREEN}✓ Metrics file persisted: metrics/round_${ROUND_ID}.json${NC}"
else
    echo -e "${YELLOW}⚠ Metrics file not found (may be in container)${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ${GREEN}TEST PASSED: Single Client Round${NC}                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0

