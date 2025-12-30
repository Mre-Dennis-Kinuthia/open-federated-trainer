# Testing the Federated Learning Client

## Prerequisites

1. **Coordinator must be running** - The client needs the coordinator server to be active.

## Quick Start Testing

### Step 1: Start the Coordinator

In one terminal, start the coordinator:

```bash
cd /home/nansi/Work/open-federated-trainer/coordinator
source venv/bin/activate  # If using virtual environment
cd src
python main.py
```

Or with uvicorn:
```bash
cd coordinator/src
uvicorn main:app --reload
```

The coordinator should start on `http://0.0.0.0:8000`

### Step 2: Install Client Dependencies

In another terminal:

```bash
cd /home/nansi/Work/open-federated-trainer/client
pip install -r requirements.txt
```

Or if using a virtual environment:
```bash
cd client
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the Client

```bash
cd client/src
python client.py
```

## Testing Scenarios

### Test 1: Single Client

1. Start coordinator (see Step 1 above)
2. Run one client:
   ```bash
   cd client/src
   python client.py
   ```

Expected output:
- Client registers successfully
- Fetches tasks
- Performs training
- Submits updates
- Repeats the cycle

### Test 2: Multiple Clients

Run multiple clients simultaneously to test federated learning:

**Terminal 1:**
```bash
cd client/src
CLIENT_NAME=client1 python client.py
```

**Terminal 2:**
```bash
cd client/src
CLIENT_NAME=client2 python client.py
```

**Terminal 3:**
```bash
cd client/src
CLIENT_NAME=client3 python client.py
```

All clients will participate in the same rounds and submit updates.

### Test 3: Custom Configuration

You can override configuration via environment variables:

```bash
COORDINATOR_URL=http://localhost:8000 \
CLIENT_NAME=my_client \
MAX_RETRIES=5 \
SLEEP_BETWEEN_ROUNDS=10 \
python client.py
```

### Test 4: Test Coordinator API Directly

While the client is running, you can check the coordinator status:

```bash
# Check round status
curl http://localhost:8000/status/1

# Check API info
curl http://localhost:8000/
```

## Verification Checklist

- [ ] Coordinator starts without errors
- [ ] Client connects to coordinator
- [ ] Client registers successfully
- [ ] Client receives tasks
- [ ] Client submits updates
- [ ] Multiple clients can run simultaneously
- [ ] Updates are aggregated correctly

## Troubleshooting

### Client can't connect to coordinator

**Error:** `CoordinatorConnectionError: Failed to connect to coordinator`

**Solution:**
1. Verify coordinator is running: `curl http://localhost:8000/`
2. Check COORDINATOR_URL in config.py or set via environment variable
3. Check firewall/network settings

### Client registration fails

**Error:** `Client already registered`

**Solution:**
- Use a different CLIENT_NAME or let the client generate a unique name automatically

### No tasks assigned

**Error:** `Could not assign task to client`

**Solution:**
- Client may already have an active assignment
- Wait for the current round to complete
- Check coordinator logs for details

## Example Test Session

```bash
# Terminal 1: Start Coordinator
cd coordinator/src
python main.py

# Terminal 2: Start Client 1
cd client/src
CLIENT_NAME=test_client_1 python client.py

# Terminal 3: Start Client 2
cd client/src
CLIENT_NAME=test_client_2 python client.py

# Terminal 4: Monitor Coordinator
watch -n 2 'curl -s http://localhost:8000/status/1 | python3 -m json.tool'
```

## Expected Client Output

```
============================================================
Federated Learning Client
============================================================
Coordinator URL: http://127.0.0.1:8000
Max Retries: 3
Sleep Between Rounds: 5.0s
============================================================
Client Name: client_abc12345

[Registration] Registering client 'client_abc12345' with coordinator...
[Registration] Successfully registered as 'client_abc12345'

[Client client_abc12345] Starting federated learning client loop...

[Client client_abc12345] === Round 1 ===
[Client client_abc12345] Fetching training task...
[Client client_abc12345] Task received: Round 1, Model v1, Task: train
[Client client_abc12345] Starting local training for round 1...
[Client client_abc12345] Training completed. Weight delta: delta_client_abc12345_r1_v1_a1b2c3d4...
[Client client_abc12345] Submitting update for round 1...
[Client client_abc12345] Update submitted successfully for round 1
[Client client_abc12345] Round 1 status: COLLECTING, 1/1 updates received
[Client client_abc12345] Waiting 5.0 seconds before next round...
```

