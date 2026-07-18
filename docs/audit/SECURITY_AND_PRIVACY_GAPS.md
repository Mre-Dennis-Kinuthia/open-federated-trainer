# Security and Privacy Gaps

Findings ranked P0–P3. Format: evidence, impact, solution, migration, tests, blocks production.

## P0

### P0-1 Registration re-issues existing API keys
- **Evidence:** `AuthManager.register_client` returns existing key for known `client_id` without proof; `POST /client/register` is public
- **Impact:** Anyone who learns/guesses a client name obtains a working credential
- **Solution:** Require presenting current key for idempotent re-register; otherwise 409 Conflict; never return existing key unauthenticated
- **Migration:** Clients with saved keys continue; first reconnect after upgrade must use saved key
- **Tests:** Double register without key → 409; with correct key → same key; wrong key → 401
- **Blocks production:** Yes

### P0-2 Operator endpoints open when `OPERATOR_API_KEY` unset
- **Evidence:** `validate_operator_key` returns `True` if env empty
- **Impact:** Aggregate, launch subprocesses, create jobs, switch models without auth
- **Solution:** Local `run.sh` always sets a documented **dev-only** default key; production docs require key; optional `REQUIRE_OPERATOR_KEY=true` fail-closed at startup
- **Migration:** Dev workflows set key via run.sh; prod must set strong secret
- **Tests:** With key set, wrong key → 401; with REQUIRE_OPERATOR_KEY and unset → refuse start (later)
- **Blocks production:** Yes if deployed without key

### P0-3 Unauthenticated ops data plane
- **Evidence:** `GET /dashboard/overview`, `GET /jobs`, `GET /jobs/{id}` return clients, payloads, results
- **Impact:** Inference prompts/results and job payloads world-readable
- **Solution (M0 partial):** Redact `result`/`payload` on public job list; keep `/dashboard/activity` public; require operator for full job detail (or strip sensitive fields). Full RBAC in later milestones
- **Migration:** UI already has operator key for mutations; extend for job detail
- **Tests:** Unauthenticated job get does not include result body (or 401)
- **Blocks production:** Yes for multi-tenant / public bind

### P0-4 Classic update size unbounded
- **Evidence:** `_require_json_size` used for adapters/jobs; not `/update`
- **Impact:** Memory/disk DoS via nested float JSON into `state.json`
- **Solution:** `MAX_UPDATE_BYTES` (default e.g. 25MB) on update payload
- **Tests:** Oversized → 413
- **Blocks production:** Yes for internet-exposed coordinator

### P0-5 Privacy overclaim: “raw data never leaves”
- **Evidence:** Job create accepts `payload.inputs`; stored in `jobs.json`
- **Impact:** Operators and anyone reading `/jobs` see prompts/texts
- **Solution:** Document honestly; prefer dataset aliases; redact public APIs
- **Blocks production:** As a claim, yes; as a feature with disclosure, no

## P1

### P1-1 Public `GET /model/{version}` full weights
- Sensitive models downloadable without node auth
- Solution: node or federation-scoped download tokens (Protocol V2)

### P1-2 API keys in query strings
- `/task`, `/jobs/claim`, `/adapters` — leak via logs/proxies
- Solution: prefer headers; deprecate query

### P1-3 Local launcher RCE surface
- `MODEL_MODULE` env into spawned process; operator-controlled
- Solution: keep launcher off by default; pin allowlisted modules for train custom

### P1-4 Silent trainer fallback to simple_mlp
- Failed dynamic import swallowed in `get_trainer`
- Solution: hard-fail when custom/module path requested

### P1-5 Geo resolve applies location to all clients missing lat
- `_resolve` loop over all clients
- Solution: only clients whose pending IP key matches

### P1-6 No TLS, no invite-only registration, no RBAC
- Documented in PRODUCTION.md

## P2–P3

- LoRA JSON without lock under concurrent submits
- Rate limiter / reputation memory-only
- Cleartext ip-api.com lookup
- No audit log of operator actions
- No differential privacy / secure aggregation (planned; must not be claimed)

## Residual privacy model (accurate wording)

| Workload | What leaves the node |
|----------|----------------------|
| Classic FL | Weight deltas (and base weights in payload today) |
| LoRA | Adapter tensors |
| Inference with payload.inputs | Prompts to coordinator + worker |
| Label without inputs | Labels only if data local |
| Compute | Work-unit params + results |

Federated learning alone does **not** guarantee privacy of training data against reconstruction or membership inference from updates.
