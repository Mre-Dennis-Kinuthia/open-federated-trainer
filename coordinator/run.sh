#!/bin/bash
# Run coordinator with virtual environment (volunteer/edge defaults)

cd "$(dirname "$0")"
source venv/bin/activate

export ENABLE_ASYNC_ROUNDS="${ENABLE_ASYNC_ROUNDS:-true}"
export ASYNC_MIN_UPDATES="${ASYNC_MIN_UPDATES:-2}"
export ASYNC_MAX_DURATION="${ASYNC_MAX_DURATION:-300}"

cd src
python main.py
