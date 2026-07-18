# Documentation Discrepancies

Each row: **doc claim** vs **code reality**. Docs must be updated to match code or code must catch up; do not leave present-tense roadmap as “implemented.”

| Doc | Claim | Reality | Severity |
|-----|-------|---------|----------|
| `PLATFORM.md` | “There are no synthetic fallbacks” | `ALLOW_SYNTHETIC_DATA` enables synthetic in `private_datasets/` and `trainer.py` | P1 truthfulness |
| `PLATFORM.md` | Loaders in `client/src/datasets/` | Directory empty; loaders in `client/src/private_datasets/` | P2 DX |
| `PLATFORM.md` / older notes | `COMPUTE_ALLOWLIST` | Env is `COMPUTE_PLUGIN_ALLOWLIST` | P2 DX |
| `PLATFORM.md` | LoRA client without round_id | CLI requires `python lora_client.py <round_id>` | P2 |
| `PLATFORM.md` / `WHITEOBER.md` | “Raw data never leaves” | Job `payload.inputs` stored on coordinator and sent to workers | P0 privacy wording |
| `README.md` / `WHITEOBER.md` | “Production-ready” | `PRODUCTION.md` lists TLS, invite auth, DP, HA as still required | P0 truthfulness |
| `README.md` | Keys only in env | Also file `client/data/api_key` | P3 |
| `QUICK_START.md` | `./simulate.sh` | Script is `tests/simulate.sh` | P2 |
| `QUICK_START.md` | Clients “simulate” training | Real PyTorch training | P2 wording |
| `docker-compose.yml` + docs | Clients train out of the box | No `DATASET_PATH` → fail unless synthetic flag | P1 ops |
| `LORA_FEDERATED_LEARNING.md` | `cd coordinator/src; python main.py` | Prefer `coordinator/run.sh` | P3 |
| `PLATFORM.md` | Size limits on updates | Adapters/jobs limited; classic `/update` lacked limit (M0 fix) | P0 |

## UI truthfulness

| Surface | Issue |
|---------|-------|
| Landing “jobs processed” | `job_stats.total` is queue length (all states), not completed count |
| Console “healthy” | Means overview reachable AND `failed_updates==0`, not full system health |
| Globe arcs | Pairings are visualization, not measured network paths |
| Geo multi-node on one host | Private IPs share host public geo → can look like a fleet |

## Required doc policy going forward

- Label features: **Implemented** / **Experimental** / **Planned** / **Unsafe for production**
- Prefer: “Raw training data remains on participating nodes” over “training is completely private”
- Prefer: “Inference inputs are sent to the selected worker pool” when `payload.inputs` is used
- Link root README to `docs/audit/` and `PRODUCTION.md` for honesty
