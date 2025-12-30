#!/bin/bash

# Federated Learning Simulation Script
# Usage: ./simulate.sh [num_clients]

set -e

NUM_CLIENTS=${1:-3}

echo "=========================================="
echo "Federated Learning Simulation"
echo "=========================================="
echo "Starting coordinator + $NUM_CLIENTS clients"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    echo "Please install Docker and Docker Compose"
    exit 1
fi

if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo "ERROR: Docker Compose is not installed"
    echo "Please install Docker Compose"
    exit 1
fi

# Determine docker compose command
if command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo "Building images..."
$DOCKER_COMPOSE build

echo ""
echo "Starting services..."
echo "Press Ctrl+C to stop"
echo ""

# Start with specified number of clients
$DOCKER_COMPOSE up --scale client=$NUM_CLIENTS

