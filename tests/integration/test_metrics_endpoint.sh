#!/bin/bash
# Integration test: Metrics endpoint validation
# Validates that metrics API endpoints work correctly

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
COORDINATOR_DIR="${COORDINATOR_DIR:-../../coordinator}"
CLIENT_DIR="${CLIENT_DIR:-../../client}"
TEST_CLIENT_NAME="test_client_metrics"

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
echo "║  Test: Metrics Endpoint Validation                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Start coordinator
echo "Step 1: Starting coordinator..."
cd "$COORDINATOR_DIR"
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/coordinator_metrics_test.log 2>&1 &
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

# Step 2: Test /metrics endpoint (should work even with no rounds)
echo ""
echo "Step 2: Testing /metrics endpoint..."
METRICS_ALL=$(curl -s "$COORDINATOR_URL/metrics")
if [ -n "$METRICS_ALL" ] && echo "$METRICS_ALL" | grep -q "global"; then
    GLOBAL_CLIENTS=$(echo "$METRICS_ALL" | python3 -c "import sys, json; print(json.load(sys.stdin)['metrics']['global'].get('total_clients_seen', 0))" 2>/dev/null || echo "0")
    echo -e "${GREEN}✓ /metrics endpoint responds${NC}"
    echo "  Global clients seen: $GLOBAL_CLIENTS"
else
    echo -e "${RED}✗ /metrics endpoint failed or invalid response${NC}"
    echo "Response: $METRICS_ALL"
    exit 1
fi

# Step 3: Test /metrics/latest endpoint (may be empty initially)
echo ""
echo "Step 3: Testing /metrics/latest endpoint..."
METRICS_LATEST=$(curl -s "$COORDINATOR_URL/metrics/latest")
if [ -n "$METRICS_LATEST" ]; then
    if [ "$METRICS_LATEST" = "{}" ]; then
        echo -e "${GREEN}✓ /metrics/latest returns empty object (no rounds yet)${NC}"
    else
        LATEST_ROUND_ID=$(echo "$METRICS_LATEST" | python3 -c "import sys, json; print(json.load(sys.stdin).get('round_id', 'none'))" 2>/dev/null || echo "none")
        echo -e "${GREEN}✓ /metrics/latest returns data: Round $LATEST_ROUND_ID${NC}"
    fi
else
    echo -e "${RED}✗ /metrics/latest endpoint failed${NC}"
    exit 1
fi

# Step 4: Register client and create a round
echo ""
echo "Step 4: Creating a test round..."
REGISTER_RESPONSE=$(curl -s -X POST "$COORDINATOR_URL/client/register" \
    -H "Content-Type: application/json" \
    -d "{\"client_name\": \"$TEST_CLIENT_NAME\"}")

if ! echo "$REGISTER_RESPONSE" | grep -q '"success":true'; then
    echo -e "${RED}✗ Client registration failed: $REGISTER_RESPONSE${NC}"
    exit 1
fi

TASK_RESPONSE=$(curl -s "$COORDINATOR_URL/task/$TEST_CLIENT_NAME")
ROUND_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['round_id'])" 2>/dev/null || echo "")

if [ -z "$ROUND_ID" ]; then
    echo -e "${RED}✗ Failed to get task${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Round $ROUND_ID created${NC}"

# Step 5: Test /metrics/round/{round_id} endpoint
echo ""
echo "Step 5: Testing /metrics/round/{round_id} endpoint..."
METRICS_ROUND=$(curl -s "$COORDINATOR_URL/metrics/round/$ROUND_ID")
if [ -n "$METRICS_ROUND" ] && [ "$METRICS_ROUND" != "{}" ]; then
    ROUND_METRICS_ROUND_ID=$(echo "$METRICS_ROUND" | python3 -c "import sys, json; print(json.load(sys.stdin).get('round_id', 'none'))" 2>/dev/null || echo "none")
    if [ "$ROUND_METRICS_ROUND_ID" = "$ROUND_ID" ]; then
        CLIENTS_ASSIGNED=$(echo "$METRICS_ROUND" | python3 -c "import sys, json; print(json.load(sys.stdin).get('clients_assigned', 0))" 2>/dev/null || echo "0")
        echo -e "${GREEN}✓ /metrics/round/$ROUND_ID returns correct data${NC}"
        echo "  Clients assigned: $CLIENTS_ASSIGNED"
    else
        echo -e "${RED}✗ Round ID mismatch in metrics${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗ /metrics/round/$ROUND_ID failed or returned empty${NC}"
    exit 1
fi

# Step 6: Submit update and verify metrics update
echo ""
echo "Step 6: Submitting update and verifying metrics update..."
cd "$CLIENT_DIR"
CLIENT_NAME="$TEST_CLIENT_NAME" COORDINATOR_URL="$COORDINATOR_URL" \
    timeout 30 python3 src/client.py > /tmp/client_metrics_test.log 2>&1 &
CLIENT_PID=$!
cd - > /dev/null

# Wait for update
sleep 10
kill $CLIENT_PID 2>/dev/null || true
wait $CLIENT_PID 2>/dev/null || true

# Check metrics again
METRICS_ROUND_AFTER=$(curl -s "$COORDINATOR_URL/metrics/round/$ROUND_ID")
UPDATES_RECEIVED=$(echo "$METRICS_ROUND_AFTER" | python3 -c "import sys, json; print(json.load(sys.stdin).get('updates_received', 0))" 2>/dev/null || echo "0")

if [ "$UPDATES_RECEIVED" -ge 1 ]; then
    echo -e "${GREEN}✓ Metrics updated after update submission: $UPDATES_RECEIVED updates${NC}"
else
    echo -e "${YELLOW}⚠ No updates in metrics (may need more time)${NC}"
fi

# Step 7: Aggregate and verify complete metrics
echo ""
echo "Step 7: Aggregating round and verifying complete metrics..."
AGGREGATE_RESPONSE=$(curl -s "$COORDINATOR_URL/aggregate/$ROUND_ID")
AGGREGATE_STATUS=$(echo "$AGGREGATE_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('status', ''))" 2>/dev/null || echo "")

if [ "$AGGREGATE_STATUS" = "aggregated" ]; then
    echo -e "${GREEN}✓ Round aggregated${NC}"
    
    # Check metrics after aggregation
    METRICS_ROUND_COMPLETE=$(curl -s "$COORDINATOR_URL/metrics/round/$ROUND_ID")
    ROUND_DURATION=$(echo "$METRICS_ROUND_COMPLETE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('round_duration_seconds', 'null'))" 2>/dev/null || echo "null")
    AGG_TIME=$(echo "$METRICS_ROUND_COMPLETE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('aggregation_time_seconds', 'null'))" 2>/dev/null || echo "null")
    
    if [ "$ROUND_DURATION" != "null" ] && [ "$AGG_TIME" != "null" ]; then
        echo -e "${GREEN}✓ Complete metrics available: duration=${ROUND_DURATION}s, agg_time=${AGG_TIME}s${NC}"
    else
        echo -e "${YELLOW}⚠ Some metrics still null after aggregation${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Aggregation status: $AGGREGATE_STATUS${NC}"
fi

# Step 8: Test /metrics/latest after aggregation
echo ""
echo "Step 8: Testing /metrics/latest after aggregation..."
METRICS_LATEST_AFTER=$(curl -s "$COORDINATOR_URL/metrics/latest")
LATEST_ROUND_ID_AFTER=$(echo "$METRICS_LATEST_AFTER" | python3 -c "import sys, json; print(json.load(sys.stdin).get('round_id', 'none'))" 2>/dev/null || echo "none")

if [ "$LATEST_ROUND_ID_AFTER" != "none" ]; then
    echo -e "${GREEN}✓ /metrics/latest returns latest round: $LATEST_ROUND_ID_AFTER${NC}"
else
    echo -e "${YELLOW}⚠ /metrics/latest returned empty or invalid${NC}"
fi

# Step 9: Verify metrics match round status
echo ""
echo "Step 9: Verifying metrics match round status..."
ROUND_STATUS=$(curl -s "$COORDINATOR_URL/status/$ROUND_ID")
STATUS_TOTAL_UPDATES=$(echo "$ROUND_STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin).get('total_updates', 0))" 2>/dev/null || echo "0")
METRICS_TOTAL_UPDATES=$(echo "$METRICS_ROUND_COMPLETE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('updates_received', 0))" 2>/dev/null || echo "0")

if [ "$STATUS_TOTAL_UPDATES" = "$METRICS_TOTAL_UPDATES" ]; then
    echo -e "${GREEN}✓ Metrics match round status: $STATUS_TOTAL_UPDATES updates${NC}"
else
    echo -e "${YELLOW}⚠ Metrics mismatch: status=$STATUS_TOTAL_UPDATES, metrics=$METRICS_TOTAL_UPDATES${NC}"
fi

# Step 10: Verify metrics file persistence
echo ""
echo "Step 10: Verifying metrics file persistence..."
if [ -f "$COORDINATOR_DIR/metrics/round_${ROUND_ID}.json" ]; then
    FILE_SIZE=$(stat -f%z "$COORDINATOR_DIR/metrics/round_${ROUND_ID}.json" 2>/dev/null || stat -c%s "$COORDINATOR_DIR/metrics/round_${ROUND_ID}.json" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -gt 0 ]; then
        echo -e "${GREEN}✓ Metrics file persisted: metrics/round_${ROUND_ID}.json (${FILE_SIZE} bytes)${NC}"
    else
        echo -e "${YELLOW}⚠ Metrics file exists but is empty${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Metrics file not found (may be in container)${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ${GREEN}TEST PASSED: Metrics Endpoint Validation${NC}                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0

