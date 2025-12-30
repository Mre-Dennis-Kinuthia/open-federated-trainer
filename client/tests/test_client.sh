#!/bin/bash
# Quick test script for the federated learning client

echo "=========================================="
echo "Testing Federated Learning Client"
echo "=========================================="
echo ""

# Check if coordinator is running
echo "Step 1: Checking if coordinator is running..."
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo "✓ Coordinator is running"
else
    echo "✗ Coordinator is NOT running!"
    echo "  Please start the coordinator first:"
    echo "  cd ../coordinator/src && python main.py"
    exit 1
fi

# Check if dependencies are installed
echo ""
echo "Step 2: Checking dependencies..."
if python3 -c "import requests" 2>/dev/null; then
    echo "✓ requests library is installed"
else
    echo "✗ requests library is missing"
    echo "  Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the client
echo ""
echo "Step 3: Starting client..."
echo "  (Press Ctrl+C to stop)"
echo ""
cd src
python3 client.py

