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
MISSING_DEPS=()

if ! python3 -c "import requests" 2>/dev/null; then
    MISSING_DEPS+=("requests")
fi

if ! python3 -c "import torch" 2>/dev/null; then
    MISSING_DEPS+=("torch")
fi

if ! python3 -c "import numpy" 2>/dev/null; then
    MISSING_DEPS+=("numpy")
fi

if [ ${#MISSING_DEPS[@]} -eq 0 ]; then
    echo "✓ All dependencies are installed"
else
    echo "✗ Missing dependencies: ${MISSING_DEPS[*]}"
    echo ""
    echo "  Attempting to install dependencies..."
    echo "  (This may take a few minutes, especially for PyTorch...)"
    echo ""
    
    cd ..
    if [ -f "requirements.txt" ]; then
        # Check if we're in a virtual environment
        if [ -z "$VIRTUAL_ENV" ]; then
            # Not in venv - check for externally-managed-environment error
            echo "  Note: System Python detected (externally-managed-environment)"
            echo ""
            echo "  RECOMMENDED: Use Docker (dependencies already installed):"
            echo "    cd ../.. && docker compose up --scale client=1"
            echo ""
            echo "  ALTERNATIVE: Create a virtual environment:"
            echo "    cd .."
            echo "    python3 -m venv venv"
            echo "    source venv/bin/activate"
            echo "    pip install -r requirements.txt"
            echo "    cd tests && ./test_client.sh"
            echo ""
            echo "  Or use --break-system-packages (not recommended):"
            echo "    python3 -m pip install --break-system-packages -r requirements.txt"
            echo ""
            exit 1
        else
            # In venv - try to install
            INSTALLED=false
            
            # Try python3 -m pip
            if python3 -m pip --version > /dev/null 2>&1; then
                echo "  Using: python3 -m pip (in virtual environment)"
                python3 -m pip install -r requirements.txt && INSTALLED=true
            # Try pip3
            elif command -v pip3 > /dev/null 2>&1; then
                echo "  Using: pip3 (in virtual environment)"
                pip3 install -r requirements.txt && INSTALLED=true
            # Try pip
            elif command -v pip > /dev/null 2>&1; then
                echo "  Using: pip (in virtual environment)"
                pip install -r requirements.txt && INSTALLED=true
            fi
            
            if [ "$INSTALLED" = true ]; then
                echo "✓ Dependencies installed"
            else
                echo ""
                echo "✗ Failed to install dependencies"
                exit 1
            fi
        fi
    else
        echo "✗ requirements.txt not found"
        exit 1
    fi
    cd tests
fi

# Run the client
echo ""
echo "Step 3: Starting client..."
echo "  (Press Ctrl+C to stop)"
echo ""
cd ../src
python3 client.py

