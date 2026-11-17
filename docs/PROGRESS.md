# PROGRESS — current state

Updated 2026-11-17. Short handoff: state, evidence, next action. Not a log.

## State

| Scope | Status |
|---|---|
| Project reset ([ADR-046](DECISIONS.md#adr-046--project-reset-2026-11-17)) | COMPLETED |
| Stage 1 — Data foundation | NOT_STARTED; needs owner authorization |
| Stages 2–5 | NOT_STARTED |

Earlier stages 0–9 are retired. Their code is in Git at `1561971`; their data and evidence
are in `backups/data/`, `artifacts/` and `mlruns/`. Do not resume them.

## What exists now

- CLI: `env`, `check mlflow`, `backup`, `data sources|acquire|verify|taxonomy`.
- Selected sources in `configs/sources.yaml`: `abcd`, `massive_en_ru`, `minds14_ru`, `xsid_kk`,
  `clinc150`, `doc2dial`, `mangi_kk_support_qa`. Raw copies are in `data/raw/` and verified.
- Taxonomy v1: `configs/taxonomy.yaml` (35 intents).
- Docker volume `supportpilot_qdrant_storage` still exists but is unused.

## Next action

Owner: define and authorize Stage 1 in [ROADMAP](ROADMAP.md#stage-1--data-foundation)
(target data, languages, labels, volume, budget).
