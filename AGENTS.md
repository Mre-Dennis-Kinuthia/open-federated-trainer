# AGENTS.md — fed-compute / open-federated-trainer

## Product

Open coordination layer for federated AI training and distributed compute over private data.

Promise: **Private data. Shared progress.**

Workloads: classic FedAvg, LoRA/PEFT, inference, labeling, allowlisted science compute, (future) federated evaluation and plugins.

## Source of truth

Code, tests, and runtime behavior beat docs. If docs disagree, fix docs or code explicitly — never leave present-tense roadmap as “implemented.”

Read before large changes:

- `docs/audit/CURRENT_STATE.md`
- `docs/architecture/TARGET_ARCHITECTURE.md`
- `docs/roadmap/IMPLEMENTATION_PLAN.md`
- `PRODUCTION.md`

## Non-negotiables

1. Raw **training** datasets stay on nodes unless an operator explicitly puts text in a job `payload.inputs` (disclose that exception).
2. Do not execute arbitrary network-supplied Python on the coordinator. Compute plugins require local `COMPUTE_PLUGIN_ALLOWLIST`.
3. Do not invent fake production statistics in the UI or landing page.
4. Do not claim federated learning alone guarantees privacy.
5. Do not implement crypto protocols or accountants from scratch; use mature libraries or interfaces only.
6. Do not add blockchain/cryptocurrency in early milestones.
7. Do not silently remove legacy demo paths; use feature flags and deprecation.
8. Prefer small vertical milestones; leave the repo passing tests after each batch.

## Architecture direction

- Control plane (metadata) separate from data plane (hashed artifacts).
- Default local demo: `METADATA_BACKEND=json`.
- Production: PostgreSQL + object store; stateless FastAPI replicas.
- Nested JSON tensors are transitional, not the production protocol.

## Commands

```bash
# Backend
client/venv/bin/python -m pytest tests/ -q

# UI
cd ui && npm run build

# Coordinator (local)
cd coordinator && ./run.sh
```

## Auth (current)

- Node: per-client API keys (`AuthManager`); prefer `X-Api-Key` / `Authorization: Bearer`. Query/body keys remain legacy adapters.
- Optional Ed25519 public key on register (Protocol V2 identity helpers).
- Operator: `OPERATOR_API_KEY` via `X-Operator-Key`. Unset key historically opened all operator routes — treat as unsafe; `run.sh` must set a **dev-only** default.
- Registration must not return an existing key without proof of possession.
- Protocol: clients may send `X-Protocol-Version: 2.0`; see `GET /protocol`.

## Documentation

Update docs with behavior changes. Label: Implemented / Experimental / Planned / Unsafe for production.
