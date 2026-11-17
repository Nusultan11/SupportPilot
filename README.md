# SupportPilot

A customer-support assistant for a fictional Kazakhstan online store.
Languages: Russian, Kazakh, English and mixed RU–KK.

**Status:** reset on 2026-11-17. The Data and ML pipelines restart from scratch.
The repository holds the engineering foundation, licensed source registry and intent taxonomy.
See [PROGRESS](docs/PROGRESS.md).

## Start here

| Need | Document |
|---|---|
| Agent rules | [AGENTS.md](AGENTS.md) |
| State · stages | [PROGRESS](docs/PROGRESS.md) · [ROADMAP](docs/ROADMAP.md) |
| Requirements · design | [PROJECT](docs/PROJECT.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) |
| Contracts · evaluation | [DATA_CONTRACTS](docs/DATA_CONTRACTS.md) · [EVALUATION](docs/EVALUATION.md) |
| Decisions · commands | [DECISIONS](docs/DECISIONS.md) · [RUNBOOK](docs/RUNBOOK.md) |

## Quick start

```powershell
uv sync --locked
uv run pre-commit install
uv run supportpilot env --no-benchmark
uv run supportpilot data sources
```

## Data

Source licences and attribution: `configs/sources.yaml`. Git holds code, configs, tests and docs.
Datasets, models, MLflow runs and backups are local and ignored.
