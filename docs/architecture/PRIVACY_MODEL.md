# Privacy Model

## Principle

**“Raw data stays local” ≠ complete privacy.** Updates, adapters, metrics, and job inputs can leak information. Every experiment/job should expose an inspectable privacy summary.

## Capability flags (target)

`data_locality`, `tls_transport`, `authenticated_nodes`, `restricted_membership`, `secure_aggregation`, `central_differential_privacy`, `local_differential_privacy`, `trusted_coordinator`, `trusted_worker_pool`, `confidential_compute`, `result_release_approval`, `prompt_end_to_end_encryption`, `location_reporting_disabled`

## Workload privacy today

| Workload | Local | Leaves node | Who can see |
|----------|-------|-------------|-------------|
| Classic train | Dataset | Weight deltas (+ bases in payload) | Coordinator (and anyone with model download) |
| LoRA | Dataset | Adapters | Coordinator; nodes with adapter download key |
| Inference + inputs | — | Prompts in job queue | Coordinator; claiming workers; unauthenticated readers of `/jobs` until fixed |
| Label local | Dataset | Labels/metrics | Coordinator |
| Compute | — | Work unit + result | Same as jobs |

## Secure aggregation / DP

- Interfaces planned; **not implemented**
- Do not claim SecAgg or DP in UI until wired and tested
- Use mature libraries only; no home-grown accountants or crypto

## Location

- Optional coarse geo for landing globe
- City round + per-client jitter
- Disable with `GEO_LOOKUP_DISABLED=true`
- Never expose IP or client id on public API

## Accurate copy

Prefer:

> Raw training data remains on participating nodes. Model updates are visible to the coordinator unless secure aggregation is enabled.

Avoid:

> Your training is completely private.
