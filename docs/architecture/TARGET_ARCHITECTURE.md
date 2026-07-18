# Target Architecture

Evolutionary target for fed-compute. Preserve local JSON demo; introduce SQL + object store behind flags.

```mermaid
flowchart TB
  subgraph nodes [Node runtime]
    Daemon[Node daemon]
    Runner[Workload runners]
    Daemon --> Runner
  end
  subgraph api [Stateless coordinator replicas]
    FastAPI[FastAPI control plane]
  end
  subgraph meta [Metadata]
    PG[(PostgreSQL)]
  end
  subgraph data [Data plane]
    Obj[Object store S3 or local FS]
  end
  subgraph ui [UI]
    Console[Operator console]
    Landing[Public landing]
  end
  Daemon -->|Protocol V2 + identity| FastAPI
  FastAPI --> PG
  FastAPI --> Obj
  Runner -->|upload/download artifacts| Obj
  Console --> FastAPI
  Landing -->|public activity only| FastAPI
```

## Principles

1. **Control plane vs data plane** — metadata in SQL; large tensors as immutable hashed artifacts in object storage; typed HTTP for control.
2. **Stateless API** — no authoritative module globals; repositories + background reconcilers.
3. **Transitional compatibility** — `METADATA_BACKEND=json|postgres` (default `json`); legacy routes behind adapters; Protocol version negotiation.
4. **No network-supplied code execution** on coordinator; compute plugins allowlisted locally on nodes; later containers by digest.
5. **Truthful privacy** — privacy profiles on experiments/jobs; no unqualified “completely private.”

## Component boundaries

| Component | Responsibility |
|-----------|----------------|
| Coordinator API | AuthZ, state machines, scheduling, manifests |
| Repositories | Persist nodes, rounds, jobs, artifacts metadata |
| ArtifactStore | Put/get/verify by content hash |
| Aggregator workers | Off-request FedAvg / LoRA merge / eval |
| Node daemon | Identity, heartbeat, capabilities, local policy, launch runners |
| UI | Authoritative views; no fake production stats |

## Non-goals (near term)

- Blockchain / cryptocurrency rewards
- Custom unreviewed crypto protocols
- Rewriting working FedAvg math without tests
- Silent removal of JSON demo path
