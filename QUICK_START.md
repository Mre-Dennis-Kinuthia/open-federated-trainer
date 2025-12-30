# Quick Start: Federated Learning Simulation

## Step-by-Step Simulation

### Step 1: Build and Start (Basic)

```bash
cd /home/nansi/Work/open-federated-trainer

# Option A: Use the simulation script
./simulate.sh 5

# Option B: Use docker compose directly
docker compose up --scale client=5
```

### Step 2: Monitor in Separate Terminals

**Terminal 2: Watch Coordinator API**
```bash
watch -n 2 'curl -s http://localhost:8000/status/1 | python3 -m json.tool'
```

**Terminal 3: Watch Models Directory**
```bash
watch -n 5 'ls -lh coordinator/models/'
```

**Terminal 4: View Specific Logs**
```bash
# All logs
docker compose logs -f

# Coordinator only
docker compose logs -f coordinator

# Clients only
docker compose logs -f client
```

### Step 3: Test Different Scenarios

**Scale Up:**
```bash
docker compose up -d --scale client=10
```

**Scale Down:**
```bash
docker compose up -d --scale client=3
```

**Stop a Client (Simulate Failure):**
```bash
docker compose stop open-federated-trainer-client-1
```

**Restart a Client:**
```bash
docker compose start open-federated-trainer-client-1
```

**Restart Coordinator:**
```bash
docker compose restart coordinator
```

### Step 4: Stop Everything

```bash
docker compose down
```

## What You Should See

1. **Coordinator starts** → Healthcheck passes → Ready on port 8000
2. **Clients start** → Wait for coordinator → Register with unique names
3. **Tasks assigned** → Each client gets a round and model version
4. **Training happens** → Clients simulate local training
5. **Updates submitted** → Clients send weight deltas
6. **Aggregation** → Coordinator creates new model version
7. **Models saved** → Files appear in `coordinator/models/`
8. **Next round** → Process repeats with new model version

## Expected Output

### Coordinator Logs
```
INFO:     Started server process
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     127.0.0.1:xxxxx - "POST /client/register HTTP/1.1" 200
INFO:     127.0.0.1:xxxxx - "GET /task/client-1 HTTP/1.1" 200
INFO:     127.0.0.1:xxxxx - "POST /update HTTP/1.1" 200
```

### Client Logs
```
[Client open-federated-trainer-client-1] Starting federated learning client loop...
[Client open-federated-trainer-client-1] === Round 1 ===
[Client open-federated-trainer-client-1] Fetching training task...
[Client open-federated-trainer-client-1] Task received: Round 1, Model v1, Task: train
[Client open-federated-trainer-client-1] Starting local training...
[Client open-federated-trainer-client-1] Training completed
[Client open-federated-trainer-client-1] Submitting update...
[Client open-federated-trainer-client-1] Update submitted successfully
```

## Troubleshooting

**Problem: Clients can't connect**
```bash
# Check coordinator is healthy
docker compose ps coordinator

# Check network
docker compose exec client ping coordinator
```

**Problem: Port 8000 already in use**
```bash
# Find what's using port 8000
sudo lsof -i :8000

# Or change port in docker-compose.yml
```

**Problem: Too many containers**
```bash
# Clean up
docker compose down
docker container prune
```

## Next Steps

- Read [SIMULATION.md](./SIMULATION.md) for detailed scenarios
- Experiment with different client counts
- Test failure scenarios
- Monitor model version progression

