#!/bin/bash
# Quick test script for logging and metrics

set -e

COORDINATOR_URL="${COORDINATOR_URL:-http://localhost:8000}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Testing Logging & Metrics                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if coordinator is running
echo "1. Checking coordinator status..."
if curl -s "${COORDINATOR_URL}/" > /dev/null 2>&1; then
    echo "   ✅ Coordinator is running"
else
    echo "   ❌ Coordinator is not running!"
    echo "   Start it with: docker compose up"
    exit 1
fi

echo ""
echo "2. Testing Metrics API endpoints..."
echo ""

echo "   📊 All Metrics:"
curl -s "${COORDINATOR_URL}/metrics" | python3 -m json.tool | head -20
echo ""

echo "   📊 Latest Round Metrics:"
curl -s "${COORDINATOR_URL}/metrics/latest" | python3 -m json.tool
echo ""

# Check if there are any rounds
LATEST_ROUND=$(curl -s "${COORDINATOR_URL}/metrics/latest" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('round_id', 'none'))" 2>/dev/null || echo "none")

if [ "$LATEST_ROUND" != "none" ] && [ "$LATEST_ROUND" != "" ]; then
    echo "   📊 Round $LATEST_ROUND Metrics:"
    curl -s "${COORDINATOR_URL}/metrics/round/${LATEST_ROUND}" | python3 -m json.tool
    echo ""
fi

echo "3. Checking persisted metrics files..."
if [ -d "coordinator/metrics" ]; then
    METRIC_FILES=$(ls coordinator/metrics/round_*.json 2>/dev/null | wc -l)
    echo "   ✅ Found $METRIC_FILES metric file(s)"
    if [ "$METRIC_FILES" -gt 0 ]; then
        echo "   Latest metric file:"
        ls -lt coordinator/metrics/round_*.json 2>/dev/null | head -1 | awk '{print "   " $9}'
    fi
else
    echo "   ⚠️  Metrics directory not found (may be in Docker container)"
fi

echo ""
echo "4. Checking summary log..."
if [ -f "coordinator/logs/rounds.log" ]; then
    echo "   ✅ Summary log exists"
    echo "   Last 5 lines:"
    tail -5 coordinator/logs/rounds.log | sed 's/^/   /'
else
    echo "   ⚠️  Summary log not found (may be in Docker container)"
fi

echo ""
echo "5. Checking coordinator JSON logs..."
if [ -f "coordinator/logs/coordinator.json.log" ]; then
    LOG_LINES=$(wc -l < coordinator/logs/coordinator.json.log 2>/dev/null || echo "0")
    echo "   ✅ Coordinator log exists ($LOG_LINES lines)"
    echo "   Last log entry:"
    tail -1 coordinator/logs/coordinator.json.log | python3 -m json.tool 2>/dev/null | head -10 | sed 's/^/   /' || tail -1 coordinator/logs/coordinator.json.log | sed 's/^/   /'
else
    echo "   ⚠️  Coordinator log not found (may be in Docker container)"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Test Complete!                                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "To view live logs:"
echo "  docker compose logs -f coordinator"
echo "  docker compose logs -f client"
echo ""
echo "To view metrics in real-time:"
echo "  watch -n 2 'curl -s http://localhost:8000/metrics/latest | python3 -m json.tool'"

