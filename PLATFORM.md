# fed-compute — Platform Overview

**fed-compute** (open-federated-trainer) is an open-source platform for
**federated learning and distributed compute over private data**. Many
independent machines ("clients") collaborate on shared workloads — training a
global model, fine-tuning an LLM, serving inference, labeling data, or running
scientific simulations — while their raw datasets **never leave their own
machines**. A central **coordinator** orchestrates the work, aggregates
results, and publishes versioned artifacts.

> Private data. Shared progress.

---

## 1. What the platform does

| Workload | What happens | What travels over the network |
|---|---|---|
| **Classic federated training** | Clients train a shared PyTorch model (MLP, CNN, or custom) on their local datasets; the coordinator averages the updates (FedAvg) and publishes a new global model version | Weight *deltas* only — never data |
| **LoRA fine-tuning** | Clients fine-tune registry LLMs (e.g. tiny-llama) with PEFT/LoRA on private text; the coordinator merges adapters, evaluates them, and versions the result | Small adapter matrices only |
| **Inference serving** | Workers run real Hugging Face Transformers pipelines against queued inputs | Prompts/inputs and predictions |
| **Auto-labeling** | Workers label private datasets locally (including zero-shot with candidate labels); only labels/metadata return | Labels, not the underlying data |
| **Science compute** | Folding@home-style jobs: workers execute allowlisted, operator-installed Python plugins (e.g. a Lennard-Jones molecular dynamics kernel) on work units | Work-unit parameters and results |

All non-training workloads flow through a **durable job queue** (claim →
execute → submit result), so the platform is a general federated task fabric,
not just a trainer.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Coordinator                          │
│                    FastAPI · port 8000                      │
│                                                             │
│  Rounds & aggregation   Job queue    LoRA rounds & eval     │
│  Model store (v1…vN)    Reputation   Incentives (tokens)    │
│  Auth & rate limiting   Privacy      Geo presence (coarse)  │
│  Local launcher         Metrics      Durable JSON state     │
└──────────▲──────────────────▲──────────────────▲────────────┘
           │ weight deltas    │ job results      │ adapters
           │ (JSON)           │                  │
   ┌───────┴──────┐   ┌───────┴──────┐   ┌───────┴──────┐
   │ Train client │   │  Job worker  │   │ LoRA client  │
   │ client.py    │   │  worker.py   │   │lora_client.py│
   │ private data │   │ HF pipelines │   │ PEFT + HF    │
   └──────────────┘   └──────────────┘   └──────────────┘

   ┌──────────────────────────────────────────────┐
   │  Web UI (React + Vite, port 5173)            │
   │  • Landing page: live activity globe         │
   │  • Operator console: rounds, clients, LoRA,  │
   │    jobs, launch, settings                    │
   └──────────────────────────────────────────────┘
```

### 2.1 Coordinator (`coordinator/`)

A FastAPI service that owns all shared state. Key modules
(`coordinator/src/core/`):

- **`round_manager` / `async_round_manager`** — classic FL round lifecycle
  (`OPEN → COLLECTING → AGGREGATING → CLOSED`). Async mode auto-aggregates
  when a minimum number of updates arrives or a time window closes.
- **`aggregator`** — FedAvg over client weight deltas. Verifies that all
  clients trained from identical base weights, the same architecture, and
  compatible configs before applying the averaged delta to publish full
  global weights.
- **`model_store`** — versioned global models (`v1`, `v2`, …) on disk with
  weights, architecture id/config, and aggregation provenance.
- **`task_assigner`** — hands each client a training task for the active
  architecture, tracking the correct model version per architecture/config.
- **`update_validator` / `privacy` / `rate_limiter`** — update sanity checks,
  privacy protections, and per-client rate limits.
- **`auth`** — per-client API keys, plus a single **operator key**
  (`OPERATOR_API_KEY`) gating all privileged actions. Sent via the
  `X-Operator-Key` header (legacy `operator_key` query param still accepted).
- **`reputation` / `incentives`** — acceptance-rate and dropout scoring, and
  token rewards (base + speed and consistency bonuses) per contribution.
- **`geo_presence`** — coarse, privacy-preserving liveness map: client IPs
  are geolocated to city level, rounded, jittered per client, and exposed
  anonymously (no IDs, no IPs) at `/dashboard/activity` for the landing globe.
- **`local_launcher`** — optionally (`ENABLE_LOCAL_LAUNCHER=true`) spawns
  train clients and job workers on the coordinator host from the UI, with
  per-process logs under `coordinator/logs/launch/`.

LoRA-specific pieces live in `rounds/` (durable round configs and adapter
submissions in `data/lora_rounds.json`), `aggregation/` (sample-weighted
adapter FedAvg), `evaluation/` (real perplexity/loss evaluation of merged
adapters, with optional regression rejection via `LORA_REJECT_REGRESSION`),
and `model_registry/` (base LLM registry). Non-training jobs live in `jobs/`
with a JSON-persisted queue that survives restarts.

### 2.2 Client (`client/`)

One codebase, three entrypoints:

- **`client.py` — train client.** Registers, polls for a task, downloads the
  current global weights, trains locally on the private dataset, and submits
  a weight delta. Architecture is chosen by `MODEL_ID`:
  - `simple_mlp` — built-in MLP for tabular data
  - `tiny_cnn` — built-in CNN for image-like tensors
  - `custom` — any user model via `MODEL_MODULE=pkg.module:TrainerClass`
    implementing the `Trainer` interface (see `client/examples/custom_linear.py`)
- **`worker.py` — job worker.** Claims queued jobs matching its `JOB_TYPES`
  (`inference,label,compute`), runs real Transformers pipelines or allowlisted
  compute plugins (`COMPUTE_PLUGIN_ALLOWLIST`), and submits results.
- **`lora_client.py` — LoRA participant.** Joins a LoRA round, fine-tunes the
  base model with PEFT on the local dataset, and uploads the adapter (with
  retry/backoff).

**Private datasets** stay on the client. `DATASET_PATH` + `DATASET_FORMAT`
support CSV, JSONL, JSON, image folders, and Hugging Face datasets
(`client/src/private_datasets/`). Synthetic data is **opt-in only** via
`ALLOW_SYNTHETIC_DATA=true`; by default a missing dataset is a hard error.

> Exception: inference/label jobs may include `payload.inputs` on the
> coordinator job queue. Prefer local dataset aliases so text never leaves
> the worker. See `docs/architecture/PRIVACY_MODEL.md`.

### 2.3 Web UI (`ui/`)

React 19 + Vite + TypeScript, in two faces:

- **Landing page** (`/#/`) — Vercel-style light design with a **live 3D globe**
  (three.js / three-globe) showing where clients are running right now.
  Points come from the anonymized `/dashboard/activity` feed, refreshed every
  8 seconds, so new joiners appear moments after they connect. Locations are
  deliberately imprecise (city-level + jitter).
- **Operator console** (`/#/overview` …) — hash-routed dashboard with:
  - **Overview** — health, metrics, activity sparklines, active architecture
  - **Launch** — one-click demo, train-client and worker launchers with
    model/dataset compatibility checks and per-process logs
  - **Rounds** — classic rounds with truthful states and manual aggregation
    (confirmation dialogs; async mode is flagged)
  - **Clients** — presence (Online/Idle/Stale/Offline from real `last_seen`),
    reputation, acceptance rate, token balances
  - **LoRA** — round creation (with advanced LoRA hyperparameters), adapter
    continuation, aggregation
  - **Jobs** — enqueue inference/label/compute, inspect payloads and results,
    cancel; warns when no worker is running
  - **Settings** — operator key management (kept in session storage, sent as
    a header, never in the URL)

  Accessibility is first-class: semantic tables, ARIA live regions,
  focus-visible outlines, a skip link, a mobile drawer nav, and WCAG AA
  contrast.

---

## 3. Data flow (classic round)

1. Operator (or `run.sh` defaults) sets the active architecture; clients
   start with `MODEL_ID` and `DATASET_PATH`.
2. Client registers (`POST /client/register`) → gets an API key; the
   coordinator records coarse presence for the activity map.
3. Client polls `GET /task/{client_id}` → receives round id, model version,
   and architecture config; downloads global weights (`GET /model/{version}`).
4. Client trains locally for the configured epochs and submits
   `POST /submit_update` with a weight delta, its base weights hash,
   sample count, and final loss.
5. The round aggregates (async trigger or explicit `GET /aggregate/{round_id}`):
   deltas are validated, averaged, applied to the canonical base, and the
   result is published as the next model version.
6. Reputation and token incentives update; metrics and the UI reflect the
   new round; the next round begins from the improved model.

LoRA rounds follow the same shape with adapters instead of weight deltas,
plus a real evaluation gate before an aggregated adapter is accepted.

---

## 4. Security & privacy model

- **Raw data never leaves clients.** Only weight deltas, adapters, labels,
  and compute results travel.
- **Operator authentication** on every mutating control-plane action
  (aggregate, launch, jobs, model switching, LoRA rounds) via
  `X-Operator-Key`.
- **Per-client API keys** for task polling and update submission, plus rate
  limiting and update validation.
- **Payload size limits** on updates and adapter uploads (HTTP 413 beyond
  configurable byte budgets).
- **Compute allowlist** — workers only execute plugin entrypoints explicitly
  listed in `COMPUTE_PLUGIN_ALLOWLIST`; arbitrary code from the queue is
  refused.
- **Location privacy** — the activity globe uses city-level geolocation,
  rounded and jittered per client; IPs and client IDs are never exposed, and
  IPs are never written to disk. Lookups can be disabled entirely with
  `GEO_LOOKUP_DISABLED=true`.
- **Durable, inspectable state** — rounds, jobs, LoRA rounds, and presence
  persist as JSON under `coordinator/data/`, surviving restarts.

---

## 5. Running the platform

```bash
# 1. Coordinator (port 8000)
cd coordinator
OPERATOR_API_KEY=<secret> ./run.sh        # local launcher enabled by default

# 2. Web UI (port 5173)
cd ui
npm install && npm run dev

# 3. Open http://localhost:5173
#    → landing page with the live globe
#    → "Open console", paste the operator key under Settings
#    → Launch page: one-click demo (2 train clients + worker + sample job)
```

Clients can also be started by hand from `client/`:

```bash
# Train client with a private CSV
DATASET_PATH=data/private/train.csv MODEL_ID=simple_mlp venv/bin/python src/client.py

# Job worker for inference/label/compute
JOB_TYPES=inference,label,compute venv/bin/python src/worker.py

# LoRA participant
DATASET_PATH=data/private/corpus.jsonl venv/bin/python src/lora_client.py
```

Key environment variables:

| Variable | Component | Purpose |
|---|---|---|
| `OPERATOR_API_KEY` | coordinator | Enables operator auth for privileged endpoints |
| `ENABLE_ASYNC_ROUNDS`, `ASYNC_MIN_UPDATES`, `ASYNC_MAX_DURATION` | coordinator | Automatic aggregation policy |
| `ENABLE_LOCAL_LAUNCHER` | coordinator | Allow starting clients/workers from the UI |
| `DEFAULT_MODEL_ID` | coordinator | Active classic architecture at boot |
| `LORA_REJECT_REGRESSION` | coordinator | Reject aggregated adapters that evaluate worse |
| `GEO_LOOKUP_DISABLED` | coordinator | Turn off IP geolocation for the globe |
| `COORDINATOR_URL`, `CLIENT_NAME` | client | Where and who |
| `MODEL_ID`, `MODEL_MODULE` | client | Architecture selection / custom trainer |
| `DATASET_PATH`, `DATASET_FORMAT` | client | Private dataset location and loader |
| `JOB_TYPES`, `COMPUTE_PLUGIN_ALLOWLIST` | worker | Job kinds and permitted compute plugins |

---

## 6. Extensibility

- **Custom architectures** — implement the `Trainer` interface (train on a
  task + data, return weights/deltas and metadata) and point `MODEL_MODULE`
  at it; the coordinator versions it like any built-in.
  See `EXTENSIBILITY.md` and `client/examples/custom_linear.py`.
- **Custom science compute** — write a pure Python function taking a work
  unit dict, add it to `COMPUTE_PLUGIN_ALLOWLIST`, enqueue jobs with
  `{"entrypoint": "pkg.module:function", "work_unit": {...}}`.
  See `client/examples/science_plugin.py` (Lennard-Jones MD).
- **New dataset formats** — add a loader in `client/src/private_datasets/`.
- **New base LLMs for LoRA** — register in `coordinator/src/model_registry/`.

---

## 7. Repository layout

```
open-federated-trainer/
├── coordinator/          FastAPI control plane
│   ├── src/core/         rounds, aggregation, auth, store, launcher, geo
│   ├── src/rounds/       LoRA round management (durable)
│   ├── src/aggregation/  adapter FedAvg
│   ├── src/evaluation/   real adapter evaluation
│   ├── src/model_registry/  LoRA base model registry
│   ├── src/jobs/         durable job queue
│   ├── data/             persisted state (rounds, jobs, presence)
│   ├── models/           versioned global models
│   └── run.sh            configured startup
├── client/               training clients & job workers
│   ├── src/              client.py, worker.py, lora_client.py, trainer.py
│   ├── src/models/       pluggable Trainer registry (MLP, CNN, custom)
│   ├── src/datasets/     CSV/JSONL/JSON/image/HF loaders
│   └── examples/         custom trainer, science plugin, sample data
├── ui/                   React landing page + operator console
│   └── src/landing/      globe hero, live activity feed
├── tests/                unit + integration suites
├── QUICK_START.md        step-by-step classic FL walkthrough
├── LORA_FEDERATED_LEARNING.md   LoRA design & usage
├── EXTENSIBILITY.md      custom models, datasets, plugins
├── PRODUCTION.md         production-readiness status
└── WHITEPAPER.md         concept & incentive design
```

---

## 8. Testing

```bash
client/venv/bin/python -m pytest tests/ -q     # full suite
cd ui && npm run build                         # strict typecheck + bundle
```

The suite covers aggregation correctness (shared base-weight verification,
FedAvg math), durable LoRA rounds across restarts, dataset loading (including
real image decoding), job queue persistence, and legacy-format rejection.
