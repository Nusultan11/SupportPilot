# ARCHITECTURE

Model: [workspace.dsl](architecture/workspace.dsl). Decisions: [DECISIONS](DECISIONS.md).

## 1. Current system (after reset)

A Python CLI (`supportpilot`) over local files. No online service yet.

| Package | Responsibility |
|---|---|
| `common/` | Settings, redacted logging, hashing, slugs, manifests, schemas, MLflow tracking |
| `data/acquire.py` | Download pinned source revisions, verify upstream hashes, publish atomically |
| `data/policy.py` | Licence/usage policy per pipeline step |
| `data/taxonomy.py` | Load and validate intent taxonomy v1 |
| `ops/` | Backups (tar + SHA256), environment report, MLflow health check |
| `cli.py` | Argument parsing and orchestration only |

Dependency direction: `cli` → `data`, `ops` → `common`. `common` imports nothing project-specific above it.

## 2. Storage

| Store | Content | Rule |
|---|---|---|
| `configs/` | Settings, source registry, taxonomy | In Git |
| `data/raw/<source>/<revision>/` | Original files + manifest | Immutable |
| `artifacts/`, `mlruns/` | Retired evidence, future models/reports | Ignored |
| `backups/` | Code bundles, data/model snapshots | Ignored, never delete without permission |

## 3. Target flow (to be rebuilt by stages)

Customer message → language detection → intent/action routing → one of:
answer from versioned knowledge with citations · call an authorized tool · clarify once ·
escalate · refuse out-of-scope. Facts never come from model weights (ADR-009).

## 4. Trust boundaries

- Data sources pass the licence policy before any use (ADR-018).
- Secrets only from the environment. Logs are redacted (ADR-016).
- MCP and agent tools are development-only (ADR-012).
