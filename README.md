# SupportPilot

A reproducible customer-support research project for a fictional Kazakhstan online store:
Russian, Kazakh, English and mixed RU–KK. Store facts come from cited KB evidence;
live order, payment and stock state must come from authorized tools. Fine-tuning teaches behaviour.

Implemented: licensed data pipelines, versioned multilingual KB, hybrid retrieval,
LLM inference/evaluation/training and typed routing. End-to-end APIs, business tools and UI
are future work. See [current status and next action](docs/PROGRESS.md);
research measurements do not imply production or autonomous-answer readiness.

## Start here

| Need | Document |
|---|---|
| Agent rules | [AGENTS.md](AGENTS.md) |
| Current work / stages | [PROGRESS](docs/PROGRESS.md) · [ROADMAP](docs/ROADMAP.md) |
| Requirements / design | [PROJECT](docs/PROJECT.md) · [ARCHITECTURE](docs/ARCHITECTURE.md) |
| Contracts / evaluation | [DATA_CONTRACTS](docs/DATA_CONTRACTS.md) · [EVALUATION](docs/EVALUATION.md) |
| Decisions / commands | [DECISIONS](docs/DECISIONS.md) · [RUNBOOK](docs/RUNBOOK.md) |
| Measured results | [Retrieval](docs/reviews/stage6_scale_results.md) · [LLM](docs/reviews/stage7_results.md) |

## Local setup

Python 3.12 managed by uv; Docker Desktop for Qdrant. CUDA checks require a compatible
NVIDIA driver. Dependencies and model revisions are pinned in the lockfile/configuration.

```powershell
uv sync --locked
uv run pre-commit install
docker compose up -d qdrant
uv run supportpilot env
uv run supportpilot --help
```

Verification, pipeline commands, cloud execution, backup/restore and safe cleanup:
[RUNBOOK](docs/RUNBOOK.md). No end-to-end chat server is supplied yet.

## Data and storage

Source licences/attribution: `configs/sources.yaml`. All demo customers/orders are synthetic.
Authored datasets and reviews disclose LLM authorship and missing independent human review.
Bitext-derived published data retains CDLA-Sharing obligations; no unrestricted project-wide
dataset licence is implied.

Git contains source, configuration, tests and documentation. Local datasets, model weights,
experiment outputs, MLflow and backups are ignored. Their locations and retention rules are
defined in [AGENTS](AGENTS.md#7-artifact-lifecycle-and-cleanup).
