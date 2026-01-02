#!/bin/bash

# Script to free up port 8000

echo "Checking what's using port 8000..."

# Check for Docker containers
echo ""
echo "1. Checking Docker containers..."
CONTAINERS=$(docker ps -a --format "{{.ID}} {{.Names}}" | grep -E "(coordinator|8000)" || true)
if [ -n "$CONTAINERS" ]; then
    echo "Found containers:"
    echo "$CONTAINERS"
    echo ""
    read -p "Stop these containers? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker ps -a --format "{{.ID}} {{.Names}}" | grep -E "(coordinator|8000)" | awk '{print $1}' | xargs -r docker stop
        echo "Containers stopped."
    fi
else
    echo "No coordinator containers found."
fi

# Check for Python processes
echo ""
echo "2. Checking Python processes..."
PYTHON_PROCS=$(ps aux | grep -E "(python.*main.py|uvicorn.*8000)" | grep -v grep || true)
if [ -n "$PYTHON_PROCS" ]; then
    echo "Found Python processes:"
    echo "$PYTHON_PROCS"
    echo ""
    read -p "Kill these processes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ps aux | grep -E "(python.*main.py|uvicorn.*8000)" | grep -v grep | awk '{print $2}' | xargs -r kill
        echo "Processes killed."
    fi
else
    echo "No Python coordinator processes found."
fi

# Check port directly
echo ""
echo "3. Checking port 8000 directly..."
if command -v lsof &> /dev/null; then
    PORT_USERS=$(lsof -i :8000 2>/dev/null || true)
    if [ -n "$PORT_USERS" ]; then
        echo "Processes using port 8000:"
        echo "$PORT_USERS"
    else
        echo "Port 8000 appears to be free."
    fi
fi

echo ""
echo "Done! Try running: docker compose up --scale client=5"

