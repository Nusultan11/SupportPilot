# SupportPilot

A multilingual (Russian / Kazakh / English) customer-support assistant for an online store.
It combines retrieval-augmented generation over store documents, tool calls to order /
inventory / payment APIs, and a compact LoRA/QLoRA-tuned LLM. It is built to answer only
from evidence, cite its sources, and hand hard cases to a human operator.

> **Status:** Stage 0 (audit, research, documentation) is done. There is no runnable code yet.
> Current state: [docs/PROGRESS.md](docs/PROGRESS.md).

## What it will do

- Delivery and order tracking, returns and refunds, payment, cancellation and order changes,
  product details and availability, store policies and FAQ.
- Detects the user's language, including mixed RU–KK, and replies in it.
- Every policy statement carries a citation to a KB chunk. Dynamic facts come only from APIs.
- Abstains when evidence is missing. Escalates complaints, sensitive or unresolved cases.

## Planned stack

Python 3.12 · uv · Polars/PyArrow · PyTorch/Transformers/PEFT/TRL · Qdrant · FastAPI/Pydantic ·
Streamlit · MLflow · pytest/Ruff/mypy · Docker Compose (only where needed).
The rationale is in [docs/DECISIONS.md](docs/DECISIONS.md).

## Documentation

| Doc | Content |
|---|---|
| [AGENTS.md](AGENTS.md) | Rules for AI agents (canonical) |
| [PROJECT](docs/PROJECT.md) | Goal, scope, requirements |
| [ARCHITECTURE](docs/ARCHITECTURE.md) | Components and data flow |
| [ROADMAP](docs/ROADMAP.md) | Stages and definitions of done |
| [DATA_CONTRACTS](docs/DATA_CONTRACTS.md) | Schemas and versioning |
| [EVALUATION](docs/EVALUATION.md) | Metrics and experiment protocol |
| [DECISIONS](docs/DECISIONS.md) | Architecture decision records |
| [PROGRESS](docs/PROGRESS.md) | Current state |
| [RUNBOOK](docs/RUNBOOK.md) | Operations, backup and restore |

## Install and run

Added in Stage 1 (environment) and Stage 8 (end-to-end demo). See [RUNBOOK](docs/RUNBOOK.md).

## Data and licences

Training data comes from public datasets whose licences are recorded per source. For example,
Bitext datasets are CDLA-Sharing-1.0, so derived data shared publicly must keep that licence.
All customers, orders and store documents are synthetic.
