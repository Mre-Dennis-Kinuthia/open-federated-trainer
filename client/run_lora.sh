#!/bin/bash
# Run LoRA client with virtual environment

cd "$(dirname "$0")"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies (this may take a few minutes)..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

cd src

# Check if round_id provided
if [ $# -lt 1 ]; then
    echo "Usage: ./run_lora.sh <round_id> [client_name]"
    exit 1
fi

python lora_client.py "$@"

