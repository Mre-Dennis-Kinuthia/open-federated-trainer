# Federated Learning Simulation Guide

This guide explains how to simulate a distributed federated learning system using Docker Compose, inspired by SETI@home / Folding@home volunteer computing.

## Prerequisites

- Docker installed and running
- Docker Compose installed (usually comes with Docker Desktop)
- At least 2GB free disk space

## Quick Start

### 1. Build and Start the System

```bash
# Navigate to project root
cd /home/nansi/Work/open-federated-trainer

# Build and start coordinator + 1 client
docker compose up --build
```

This will:
- Build the coordinator and client images
- Start the coordinator on port 8000
- Start 1 client that connects to the coordinator
- Show logs from all containers

### 2. Scale to Multiple Clients

In a **new terminal**, run:

```bash
cd /home/nansi/Work/open-federated-trainer

# Scale to 5 clients
docker compose up --scale client=5

# Or scale to 10 clients
docker compose up --scale client=10
```

Each client container will:
- Get a unique hostname (e.g., `open-federated-trainer-client-1`, `open-federated-trainer-client-2`)
- Automatically register with the coordinator
- Participate in training rounds independently

## Simulation Scenarios

### Scenario 1: Basic Multi-Client Simulation

```bash
# Terminal 1: Start coordinator + 3 clients
docker compose up --scale client=3
```

**What to observe:**
- Coordinator starts and becomes healthy
- 3 clients register with unique names
- Clients fetch tasks and submit updates
- Coordinator aggregates updates into new model versions
- Models are saved to `coordinator/models/`

### Scenario 2: Large-Scale Simulation

```bash
# Terminal 1: Start coordinator + 20 clients
docker compose up --scale client=20
```

**What to observe:**
- Many clients registering simultaneously
- Multiple rounds running concurrently
- Coordinator handling multiple updates
- Model versions incrementing (v1 → v2 → v3...)

### Scenario 3: Simulating Unreliable Nodes

```bash
# Terminal 1: Start system
docker compose up --scale client=5

# Terminal 2: Stop a random client
docker compose stop open-federated-trainer-client-3

# Wait 10 seconds, then restart it
docker compose start open-federated-trainer-client-3
```

**What to observe:**
- Client stops participating
- Other clients continue training
- Restarted client re-registers and continues
- System continues functioning despite node failures

### Scenario 4: Coordinator Restart (Persistence Test)

```bash
# Terminal 1: Start system
docker compose up --scale client=3

# Wait for a few rounds to complete (models saved)

# Terminal 2: Restart coordinator
docker compose restart coordinator
```

**What to observe:**
- Coordinator restarts and loads latest model version
- Clients automatically reconnect
- Training continues from where it left off
- Models persist across restarts

## Monitoring the System

### View All Logs

```bash
docker compose logs -f
```

### View Coordinator Logs Only

```bash
docker compose logs -f coordinator
```

### View Client Logs Only

```bash
docker compose logs -f client
```

### View Specific Client Logs

```bash
# List all client containers
docker compose ps client

# View logs for specific client
docker compose logs -f open-federated-trainer-client-1
```

### Monitor Coordinator API

```bash
# Check coordinator status
curl http://localhost:8000/

# Check round status
curl http://localhost:8000/status/1 | python3 -m json.tool

# List available models
ls -la coordinator/models/
```

## Useful Commands

### Check Running Containers

```bash
docker compose ps
```

### Stop All Containers

```bash
docker compose down
```

### Stop and Remove All (including volumes)

```bash
docker compose down -v
```

### Rebuild Images

```bash
docker compose build --no-cache
```

### Scale Clients Dynamically

```bash
# Scale up to 10 clients
docker compose up -d --scale client=10

# Scale down to 3 clients
docker compose up -d --scale client=3
```

### View Resource Usage

```bash
docker stats
```

## Expected Behavior

### Coordinator

- Starts on port 8000
- Healthcheck passes after ~10 seconds
- Logs show client registrations
- Logs show task assignments
- Logs show update submissions
- Logs show aggregation results

### Clients

- Each client gets unique name from hostname
- Clients wait for coordinator to be healthy
- Clients register automatically
- Clients fetch tasks continuously
- Clients submit updates after training
- Clients handle coordinator downtime gracefully

### Models

- Models saved to `coordinator/models/model_v1.json`
- New versions created after each aggregation
- Models persist across coordinator restarts
- Latest version loaded on coordinator startup

## Troubleshooting

### Coordinator Not Starting

```bash
# Check logs
docker compose logs coordinator

# Verify port 8000 is not in use
netstat -tuln | grep 8000
```

### Clients Can't Connect

```bash
# Verify coordinator is healthy
docker compose ps coordinator

# Check network connectivity
docker compose exec client ping coordinator

# Verify COORDINATOR_URL
docker compose exec client env | grep COORDINATOR_URL
```

### Models Not Persisting

```bash
# Check volume mount
docker compose exec coordinator ls -la /app/models

# Verify host directory exists
ls -la coordinator/models/
```

### Too Many Containers

```bash
# Stop all
docker compose down

# Remove unused containers
docker container prune

# Remove unused images
docker image prune
```

## Performance Tips

1. **Start Small**: Begin with 3-5 clients to verify everything works
2. **Monitor Resources**: Use `docker stats` to monitor CPU/memory
3. **Adjust Sleep Times**: Reduce `SLEEP_BETWEEN_ROUNDS` for faster simulation
4. **Limit Logs**: Use `--no-log-prefix` for cleaner output

## Example Session

```bash
# Terminal 1: Start system with 5 clients
$ docker compose up --scale client=5

# Output shows:
# - Coordinator starting
# - 5 clients registering
# - Tasks being assigned
# - Updates being submitted
# - Models being aggregated

# Terminal 2: Monitor coordinator API
$ watch -n 2 'curl -s http://localhost:8000/status/1 | python3 -m json.tool'

# Terminal 3: Check models
$ watch -n 5 'ls -lh coordinator/models/'

# After a few minutes, you should see:
# - Multiple rounds completed
# - Multiple model versions (v1, v2, v3...)
# - All clients participating
```

## Next Steps

- Experiment with different numbers of clients
- Test coordinator restart scenarios
- Monitor model version progression
- Analyze round completion times
- Test with different `SLEEP_BETWEEN_ROUNDS` values

