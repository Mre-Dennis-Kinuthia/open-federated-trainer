# Integration Tests

This directory contains integration tests for the Federated Learning MVP system. These tests validate the entire system end-to-end under realistic conditions.

## Overview

The integration tests verify:
- ✅ Coordinator and client end-to-end communication
- ✅ Multi-client federated learning rounds
- ✅ Client dropout and failure handling
- ✅ Invalid update rejection
- ✅ Metrics and logging functionality

## Prerequisites

- Python 3.9+
- `curl` command-line tool
- `python3` with `json` module (standard library)
- Coordinator and client code in `../../coordinator` and `../../client`

## Test Scripts

### 1. `test_single_round.sh`
**Purpose:** Validates a single client completing a full federated learning round.

**What it tests:**
- Coordinator startup
- Client registration
- Task assignment
- Update submission
- Aggregation
- Metrics persistence

**Run:**
```bash
cd tests/integration
./test_single_round.sh
```

### 2. `test_multi_client.sh`
**Purpose:** Validates multiple clients participating in a federated learning round.

**What it tests:**
- Multiple client registration
- Coordinated task assignment
- Parallel update submission
- Aggregation with multiple updates
- Metrics tracking for multi-client rounds

**Run:**
```bash
cd tests/integration
./test_multi_client.sh
```

### 3. `test_client_dropout.sh`
**Purpose:** Simulates client failures and validates graceful handling.

**What it tests:**
- Client dropout mid-round
- System continues with remaining clients
- Partial participation aggregation
- System recovery after failure

**Run:**
```bash
cd tests/integration
./test_client_dropout.sh
```

### 4. `test_invalid_update.sh`
**Purpose:** Validates that the coordinator rejects invalid updates gracefully.

**What it tests:**
- Empty weight_delta rejection
- Invalid client_id rejection
- Invalid round_id rejection
- Missing field handling
- Coordinator stability after errors

**Run:**
```bash
cd tests/integration
./test_invalid_update.sh
```

### 5. `test_metrics_endpoint.sh`
**Purpose:** Validates metrics API endpoints and data consistency.

**What it tests:**
- `/metrics` endpoint response
- `/metrics/latest` endpoint
- `/metrics/round/{round_id}` endpoint
- Metrics update after events
- Metrics persistence to disk
- Metrics consistency with round status

**Run:**
```bash
cd tests/integration
./test_metrics_endpoint.sh
```

## Running All Tests

To run all tests in sequence:

```bash
cd tests/integration
for test in test_*.sh; do
    echo "Running $test..."
    ./$test || exit 1
done
echo "All tests passed!"
```

Or use the test runner:

```bash
cd tests/integration
./run_all_tests.sh  # (if created)
```

## Test Output

Each test script:
- Prints colored output (green for pass, red for fail)
- Shows step-by-step progress
- Cleans up processes on exit
- Exits with code 0 on success, non-zero on failure

## Environment Variables

Tests respect these environment variables:

- `COORDINATOR_URL`: Coordinator API URL (default: `http://localhost:8000`)
- `COORDINATOR_DIR`: Path to coordinator directory (default: `../../coordinator`)
- `CLIENT_DIR`: Path to client directory (default: `../../client`)

Example:
```bash
COORDINATOR_URL=http://localhost:9000 ./test_single_round.sh
```

## Troubleshooting

### Tests fail with "Coordinator failed to start"
- Ensure port 8000 is not in use
- Check that coordinator dependencies are installed
- Verify Python path is correct

### Tests fail with "Client registration failed"
- Ensure coordinator is running
- Check coordinator logs in `/tmp/coordinator_*_test.log`
- Verify network connectivity

### Tests timeout waiting for updates
- Increase timeout values in test scripts
- Check client logs in `/tmp/client_*_test.log`
- Verify client can connect to coordinator

### Metrics tests fail
- Ensure metrics endpoints are implemented (Step 5)
- Check that metrics directory exists: `coordinator/metrics/`
- Verify aggregation was triggered

## Test Logs

Test scripts create temporary log files:
- Coordinator logs: `/tmp/coordinator_*_test.log`
- Client logs: `/tmp/client_*_test.log`

These are cleaned up automatically but can be inspected for debugging.

## Success Criteria

A test passes when:
1. ✅ All steps complete without errors
2. ✅ Expected HTTP status codes are returned
3. ✅ Metrics data is consistent
4. ✅ System remains stable after failures
5. ✅ Cleanup completes successfully

## Notes

- Tests use real processes (no mocking)
- Tests are designed for Ubuntu WSL but should work on Linux/Mac
- Tests may take 30-60 seconds each to complete
- Tests clean up processes automatically on exit (Ctrl+C safe)

## Contributing

When adding new tests:
1. Follow the existing test structure
2. Use colored output for clarity
3. Implement proper cleanup
4. Exit with appropriate codes
5. Document what the test validates

