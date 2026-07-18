# Production readiness notes — volunteer / edge

This document tracks what was hardened toward production use and what remains.

## Implemented (this iteration)

1. **Durable auth** — API keys + client registry in `coordinator/data/state.json`
2. **Idempotent registration** — returning clients get the same key (no hard fail)
3. **Client key persistence** — `client/data/api_key` (or `CLIENT_API_KEY` / `CLIENT_API_KEY_FILE`)
4. **Real classic FedAvg** — clients load the shared global model, train from identical base weights, and the coordinator applies averaged deltas to produce the next global model
5. **Pending update checkpoint** — in-flight updates survive coordinator restart
6. **Async auto-aggregation** — `ENABLE_ASYNC_ROUNDS=true` by default; aggregates when min updates or timeout hit
7. **Operator auth** — set `OPERATOR_API_KEY` to lock create/aggregate endpoints
8. **Health** — `GET /health` for orchestration
9. **Docker volumes** — models, metrics, logs, data, adapters; client key volume; restart policies
10. **LoRA client** — fixed `get_base_model_name`; persists API key
11. **Pluggable models** — `simple_mlp` / `tiny_cnn` / `MODEL_MODULE` custom trainers (see EXTENSIBILITY.md)
12. **Private datasets** — CSV / JSONL / JSON / folder / HuggingFace via `DATASET_PATH` (data stays on client)
13. **Non-training jobs** — real Transformers inference/labeling + allowlisted compute plugins
14. **Durable job queue** — jobs, leases, attempts, and results survive coordinator restart
15. **Real LoRA evaluation** — optional base-model holdout loss; no proxy metric
16. **LoRA continuation** — authenticated clients download prior aggregated adapters
17. **Durable LoRA rounds** — round configuration, lifecycle, and adapter submissions survive coordinator restarts
18. **Reliable LoRA uploads** — transient failures use bounded exponential retries; upload size is capped by `MAX_ADAPTER_UPLOAD_BYTES`
19. **Safer local launcher** — disabled by default outside `coordinator/run.sh`

## Still required for full production fleet

| Area | Gap |
|------|-----|
| Transport | TLS reverse proxy (Caddy/nginx); never expose :8000 bare |
| Auth | Invite-gated registration; rotate keys; no keys in query strings long-term |
| Privacy | Optional DP noise accounting; secure aggregation |
| Scheduling | Reputation-weighted client selection / FedAvg weights |
| LoRA | Asynchronous evaluation and resumable/chunked adapter uploads for very large adapters |
| Observability | Prometheus metrics; structured log shipping |
| Scale | Compose/Helm multi-replica + PG/S3; residual in-memory state (geo/reputation) |
| Data | Optional DP; richer feature extractors (beyond CSV/JSONL/HF loaders) |
| Jobs | Sandboxed plugin containers; queue shared via SQL when `METADATA_BACKEND=postgres` |
| Deploy | See `docs/deploy/PRODUCTION_DEPLOY.md` + Helm chart |

## Recommended operator env

```bash
export ENABLE_ASYNC_ROUNDS=true
export ASYNC_MIN_UPDATES=2
export ASYNC_MAX_DURATION=300
export OPERATOR_API_KEY="$(openssl rand -hex 16)"
export CORS_ORIGINS="https://ops.example.com"
```

Aggregate with:

```bash
curl "http://localhost:8000/aggregate/1?operator_key=$OPERATOR_API_KEY"
```

## Verify

```bash
cd coordinator && ./run.sh
cd client && python src/client.py   # x2
curl -s http://localhost:8000/health
cd ui && ./run.sh
python tests/test_fedavg.py
```
