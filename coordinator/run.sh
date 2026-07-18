#!/bin/bash
# Run coordinator with virtual environment (volunteer/edge defaults)

cd "$(dirname "$0")"
source venv/bin/activate

export ENABLE_ASYNC_ROUNDS="${ENABLE_ASYNC_ROUNDS:-true}"
export ASYNC_MIN_UPDATES="${ASYNC_MIN_UPDATES:-2}"
export ASYNC_MAX_DURATION="${ASYNC_MAX_DURATION:-300}"
export ENABLE_LOCAL_LAUNCHER="${ENABLE_LOCAL_LAUNCHER:-true}"
# Dev-only default. Override with a strong secret in any shared or production deploy.
# Unset OPERATOR_API_KEY historically left all operator routes open — that is unsafe.
export OPERATOR_API_KEY="${OPERATOR_API_KEY:-dev-operator-change-me}"

cd src
python main.py
