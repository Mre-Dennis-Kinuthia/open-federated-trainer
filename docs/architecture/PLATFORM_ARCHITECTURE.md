# fed-compute platform architecture

How the coordinator, TLS edge, local UI/launcher, and edge devices interact (HA production profile).

> **Vercel note:** The hosted UI at `fed-compute-ui.vercel.app` calls same-origin paths like `/launch/demo`. Vercel only serves the SPA, so POSTs return **405** and GETs return HTML. Rebuild with `VITE_API_BASE=https://your-public-coordinator` and allow that origin in `CORS_ORIGINS`. Local UI uses Vite `/api` → `:8443`.

---

## System map.

| | |
|---|---|
| Coordinator replicas | 2 (`coordinator-a`, `coordinator-b`) |
| Public TLS edge | `:8443` (HTTP `:8080` redirects) |
| Shared metadata | Postgres |
| Model artifacts | MinIO / S3 |

```
┌─────────────────┐   ┌─────────────────┐   ┌──────────────────────────────┐
│ Operator UI     │   │ Local launcher  │   │ Edge / volunteer devices     │
│ Vite :5173      │   │ /launch         │   │ client.py · lora · worker    │
│ Vercel SPA      │   │ train + worker  │   │                              │
└────────┬────────┘   └────────┬────────┘   └──────────────┬───────────────┘
         │                     │                           │
         └─────────────────────┼───────────────────────────┘
                               ▼
              ┌────────────────────────────────────┐
              │  TLS edge — nginx :8443            │
              │  ip_hash → coordinator-a / b       │
              └─────────────────┬──────────────────┘
                       ┌───────┴───────┐
                       ▼               ▼
         ┌──────────────────┐   ┌──────────────────┐
         │ coordinator-a    │   │ coordinator-b    │
         │ :8000            │   │ :8000            │
         │ rounds · jobs    │   │ HA replica       │
         │ launch · overview│   │ shared SQL state │
         └────────┬─────────┘   └────────┬─────────┘
                  │                      │
                  └──────────┬───────────┘
                             ▼
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐            ┌────────────────┐
     │ Postgres       │            │ MinIO / S3     │
     │ rounds · nodes │            │ models ·       │
     │ jobs · HA locks│            │ adapters       │
     └────────────────┘            └────────────────┘
```

Compose clients talk to `coordinator-a` on the Docker network; external devices and the browser hit nginx `:8443`. Overview reloads durable rounds so A/B do not flicker.

---

## Classic FL train loop

```
Register → Task → Model → Train local → Update → Aggregate
```

| Step | Who | Wire | Payload |
|------|-----|------|---------|
| 1. Register | Train client | `POST /client/register` | name → `client_id` + API key |
| 2. Task | Train client | `GET /task/{id}` | round, model version, config |
| 3. Model | Train client | `GET /model/{v}` | global weights download |
| 4. Train | Device only | local | `DATASET_PATH` — rows never uploaded |
| 5. Update | Train client | `POST /update` or `/v2/*` | deltas, hashes, metrics |
| 6. Aggregate | Coordinator | async or `GET /aggregate/{id}` | FedAvg → new version |

**Sources:** `client/src/client.py` · coordinator `RoundManager` · Protocol V2 optional binary path.

---

## Edge devices & jobs

### Edge train devices (FL nodes)

Same `client.py` / `lora_client.py` as lab clients. Join rounds, tolerate stragglers, drop out safely.

- **Compose demo:** `client-1` / `client-2` → `http://coordinator-a:8000`
- **Field devices:** HTTPS → edge `:8443` with `X-Api-Key`

### Job volunteers (workers)

`worker.py` claims inference / label / compute jobs from the durable queue.

| API | Role |
|-----|------|
| `GET /jobs/claim` | Lease a job |
| `POST /jobs/{id}/lease` | Heartbeat |
| `POST /jobs/{id}/result` | Return JSON result |

Compute plugins must be allowlisted on the worker (`COMPUTE_PLUGIN_ALLOWLIST`).

### Local vs remote launch

| Path | Where processes run | How UI starts them |
|------|---------------------|--------------------|
| Local launcher | Coordinator host / container with `CLIENT_ROOT` | `POST /launch` · `/launch/demo` |
| Compose clients | `federated-client-1/2` containers | `docker start` (not the Launch panel) |
| External edge | User devices / VMs | Manual `client.py` against `:8443` |

---

## What leaves the node

### Stays on the device

- Raw training / label datasets (`DATASET_PATH`)
- Local GPU/CPU compute
- Allowlisted plugin code on workers
- Client API key material on disk

### Leaves the device

- Weight deltas / LoRA adapters / sample counts / loss
- Job payloads & results (prompts if you put them in the job)
- Coarse city-level geo for the activity globe (optional)

FL alone ≠ privacy. No SecAgg / DP in this build — see [`PRIVACY_MODEL.md`](./PRIVACY_MODEL.md).

---

## Deploy modes

| Mode | Public API | Stack | UI proxy |
|------|------------|-------|----------|
| JSON demo | `:8000` | Single coordinator · local JSON/artifacts | `VITE_API_PROXY=http://127.0.0.1:8000` |
| HA lab (prod compose) | `https://localhost:8443` | nginx · A/B · Postgres · MinIO · client-1/2 | default Vite `/api` → `:8443` |
| Vercel SPA | Must set `VITE_API_BASE` | Static UI only — no FastAPI on Vercel | Absolute coordinator URL + CORS |

### Operator surfaces

- `GET /dashboard/overview` — console poll
- `GET /dashboard/activity` — landing globe feed
- Privileged routes (`/launch`, aggregate, jobs) honor `OPERATOR_API_KEY` when set

---

## Key API paths

| Path | Role |
|------|------|
| `POST /client/register` | Node identity + API key |
| `GET /task/{client_id}` | FL assignment |
| `GET /model/{version}` | Download global weights |
| `POST /update` | Submit FL update (legacy JSON) |
| `POST /v2/deltas`, `POST /v2/updates` | Binary/hash update path |
| `GET /aggregate/{round_id}` | Classic aggregate |
| `POST /rounds/{round_id}/aggregate` | LoRA / operator aggregate |
| `POST/GET /jobs`, `GET /jobs/claim`, … | Job fabric |
| `GET /dashboard/overview` | Operator console |
| `GET/POST /launch`, `/launch/demo` | Local launcher |
| `GET /health`, `GET /ready` | Liveness / readiness |

---

## Sources

- `docker-compose.prod.yml`
- `deploy/nginx/nginx.conf`
- `coordinator/src/main.py`
- `client/src/`
- `ui/src/api.ts`
- `docs/deploy/PRODUCTION_DEPLOY.md`
