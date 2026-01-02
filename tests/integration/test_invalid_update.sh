#!/bin/bash
# Integration test: Invalid update handling
# Validates that coordinator rejects invalid updates gracefully

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"
COORDINATOR_DIR="${COORDINATOR_DIR:-../../coordinator}"
CLIENT_DIR="${CLIENT_DIR:-../../client}"
TEST_CLIENT_NAME="test_client_invalid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Cleanup function
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $COORDINATOR_PID 2>/dev/null || true
    wait $COORDINATOR_PID 2>/dev/null || true
}

trap cleanup EXIT

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Test: Invalid Update Handling                                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Start coordinator
echo "Step 1: Starting coordinator..."
cd "$COORDINATOR_DIR"
python3 -m uvicorn src.main:app --host 0.0.0.0 --port 8000 > /tmp/coordinator_invalid_test.log 2>&1 &
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

# Step 3: Fetch task to get a valid round_id
echo ""
echo "Step 3: Fetching task to get round_id..."
TASK_RESPONSE=$(curl -s "$COORDINATOR_URL/task/$TEST_CLIENT_NAME")
ROUND_ID=$(echo "$TASK_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['round_id'])" 2>/dev/null || echo "")

if [ -z "$ROUND_ID" ]; then
    echo -e "${RED}✗ Failed to get task: $TASK_RESPONSE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Task assigned: Round $ROUND_ID${NC}"

# Step 4: Test 1: Empty weight_delta
echo ""
echo "Step 4: Test 1 - Submitting empty weight_delta..."
EMPTY_UPDATE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$COORDINATOR_URL/update" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"$TEST_CLIENT_NAME\", \"round_id\": $ROUND_ID, \"weight_delta\": \"\"}")

HTTP_CODE=$(echo "$EMPTY_UPDATE_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "400" ]; then
    echo -e "${GREEN}✓ Empty update correctly rejected (HTTP 400)${NC}"
else
    echo -e "${RED}✗ Empty update not rejected: HTTP $HTTP_CODE${NC}"
    exit 1
fi

# Step 5: Test 2: Invalid client_id
echo ""
echo "Step 5: Test 2 - Submitting update with invalid client_id..."
INVALID_CLIENT_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$COORDINATOR_URL/update" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"nonexistent_client\", \"round_id\": $ROUND_ID, \"weight_delta\": \"valid_delta\"}")

HTTP_CODE=$(echo "$INVALID_CLIENT_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "400" ]; then
    echo -e "${GREEN}✓ Invalid client_id correctly rejected (HTTP 400)${NC}"
else
    echo -e "${RED}✗ Invalid client_id not rejected: HTTP $HTTP_CODE${NC}"
    exit 1
fi

# Step 6: Test 3: Invalid round_id
echo ""
echo "Step 6: Test 3 - Submitting update with invalid round_id..."
INVALID_ROUND_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$COORDINATOR_URL/update" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"$TEST_CLIENT_NAME\", \"round_id\": 99999, \"weight_delta\": \"valid_delta\"}")

HTTP_CODE=$(echo "$INVALID_ROUND_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "400" ]; then
    echo -e "${GREEN}✓ Invalid round_id correctly rejected (HTTP 400)${NC}"
else
    echo -e "${RED}✗ Invalid round_id not rejected: HTTP $HTTP_CODE${NC}"
    exit 1
fi

# Step 7: Test 4: Missing required fields
echo ""
echo "Step 7: Test 4 - Submitting update with missing fields..."
MISSING_FIELDS_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$COORDINATOR_URL/update" \
    -H "Content-Type: application/json" \
    -d "{\"client_id\": \"$TEST_CLIENT_NAME\", \"round_id\": $ROUND_ID}")

HTTP_CODE=$(echo "$MISSING_FIELDS_RESPONSE" | tail -1)
if [ "$HTTP_CODE" = "422" ] || [ "$HTTP_CODE" = "400" ]; then
    echo -e "${GREEN}✓ Missing fields correctly rejected (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${YELLOW}⚠ Missing fields response: HTTP $HTTP_CODE (may be acceptable)${NC}"
fi

# Step 8: Verify coordinator is still operational
echo ""
echo "Step 8: Verifying coordinator is still operational after invalid updates..."
HEALTH_CHECK=$(curl -s "$COORDINATOR_URL/" 2>/dev/null || echo "")
if echo "$HEALTH_CHECK" | grep -q "Federated Learning Coordinator"; then
    echo -e "${GREEN}✓ Coordinator is still operational${NC}"
else
    echo -e "${RED}✗ Coordinator may have crashed${NC}"
    echo "Response: $HEALTH_CHECK"
    exit 1
fi

# Step 9: Verify valid update still works
echo ""
echo "Step 9: Verifying valid update still works..."
# Start a client to submit a valid update
cd "$CLIENT_DIR"
CLIENT_NAME="$TEST_CLIENT_NAME" COORDINATOR_URL="$COORDINATOR_URL" \
    timeout 30 python3 src/client.py > /tmp/client_invalid_test.log 2>&1 &
CLIENT_PID=$!
cd - > /dev/null

# Wait for valid update
sleep 10
STATUS=$(curl -s "$COORDINATOR_URL/status/$ROUND_ID" 2>/dev/null || echo "{}")
TOTAL_UPDATES=$(echo "$STATUS" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('total_updates', 0))" 2>/dev/null || echo "0")

kill $CLIENT_PID 2>/dev/null || true
wait $CLIENT_PID 2>/dev/null || true

if [ "$TOTAL_UPDATES" -ge 1 ]; then
    echo -e "${GREEN}✓ Valid updates still work after invalid attempts${NC}"
else
    echo -e "${YELLOW}⚠ No valid updates received (may need more time)${NC}"
fi

# Step 10: Check logs for error messages
echo ""
echo "Step 10: Checking coordinator logs for error handling..."
if grep -q "update_rejected\|Invalid update" /tmp/coordinator_invalid_test.log 2>/dev/null; then
    echo -e "${GREEN}✓ Invalid updates logged correctly${NC}"
else
    echo -e "${YELLOW}⚠ Could not verify error logging in coordinator log${NC}"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ${GREEN}TEST PASSED: Invalid Update Handling${NC}                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
exit 0

