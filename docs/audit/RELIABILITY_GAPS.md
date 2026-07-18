# Reliability Gaps

## P0

### Classic rounds durability (mitigated in M1; reconcile in M3)
- **Was:** `RoundManager` restored only `clients` / `next_round_id`; open rounds were memory-only.
- **Now:** Rounds persist via `JsonRoundRepository` (`classic_rounds.json`) or SQL when `METADATA_BACKEND` is `sqlite`/`sql`/`postgres`, and restore on boot.
- **M3:** Mid-aggregate crash rolls `AGGREGATING` → `COLLECTING` with `resume_after_crash`; `Aggregator.reconcile_after_restart()` finishes the round. Duplicate aggregate returns the same `published_version`.
- **Remaining:** Async round closed-sets / timeouts and reputation still memory-only.

## P1

| Finding | Evidence | Impact | Direction |
|---------|----------|--------|-----------|
| Reputation/incentives lost | In-memory managers | Scoring reset after restart | Persist ledger |
| Async closed-set lost | `AsyncRoundManager` memory | Double aggregate / timeout confusion | Persist |
| Launcher process table lost | `_procs` memory | Orphan children; UI lies | Persist or probe PIDs |
| LoRA persist without lock | `LoRARoundManager._persist` | Corrupted `lora_rounds.json` | `threading.Lock` |
| ModelStore non-atomic write | `open('w')` without tmp+replace | Partial model files | Atomic replace like StateStore |
| Metrics incomplete after restart | Live map memory; disk only after `end_round` | Dashboard gaps | Flush periodically |
| No idempotency keys | Updates/jobs | Duplicate submissions | Protocol V2 |
| No lease extension heartbeats for jobs | Mitigated (M5) | `extend_lease` + worker heartbeat; expire→requeue |
| Aggregation in request path | Sync FedAvg / LoRA eval | Timeouts under load | Background workers |
| Single process globals | `main.py` singletons | Cannot horizontally scale | Stateless API + shared DB |

## P2

- No OpenTelemetry / Prometheus
- Health does not check disk/DB readiness separately
- Integration shell tests stale vs `DATASET_PATH`
- No fault tests for restart mid-round / mid-job
- UI polls every 3s without ETag / conditional requests

## Required tests (reliability)

- Restart coordinator with open classic round + pending updates → consistent recovery
- Concurrent LoRA submits → no corrupt JSON
- Job lease expire → requeue
- Duplicate aggregate → single published version
- Oversized update rejected without writing pending
