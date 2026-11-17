# PROGRESS — current state

Updated 2026-11-17. Short handoff: state, evidence, next action. Not a log.

## State

| Scope | Status |
|---|---|
| Project reset ([ADR-046](DECISIONS.md#adr-046--project-reset-2026-11-17)) | COMPLETED |
| Stage 1 — Data foundation ([ADR-047](DECISIONS.md#adr-047--stage-1-corpus-from-human-authored-sources-2026-11-17)) | PARTIALLY COMPLETED: pipeline done; KK / RU–KK and RU in-domain data missing; human review NOT_EVALUATED |
| Stages 2–5 | NOT_STARTED; Stage 2 needs owner authorization |

Earlier stages 0–9 are retired. Their code is in Git at `1561971`; their data and evidence
are in `backups/data/`, `artifacts/` and `mlruns/`. Do not resume them.

## What exists now

- CLI: `env`, `check mlflow`, `backup`, `data sources|acquire|verify|taxonomy|corpus`.
- Selected sources in `configs/sources.yaml`; raw copies in `data/raw/`, verified.
  `mangi_kk_support_qa` is registered but not admitted (authorship undocumented).
- Taxonomy v1: `configs/taxonomy.yaml` (35 intents). Corpus recipe: `configs/corpus.yaml`.
- Corpus `data/corpus/corpus-20261117-379aa455/` (backup `corpus`, pinned). Its `REPORT.md`
  has the full per-language report. Leakage and residual-PII gates: all 0.

| Intent corpus | EN | RU | KK | mixed RU–KK |
|---|---|---|---|---|
| core rows | 34,934 | 10,795 | 636 | 1 (auto-detected) |
| independent families | 30,896 | 10,238 | 612 | — |
| in-domain core rows | 6,729 | 39 | 0 | 0 |
| unverified (not targets) | 1,894 | 745 | 0 | 0 |
| v1 intents with no core row | 15 | 30 | 34 | 35 |

- RU core = human-localised MASSIVE (out-of-scope / greeting) + 260 MINDS-14 rows.
- KK core = 636 human-translated xSID rows, all `out_of_scope`, validation/test only.
- Grounded (doc2dial, EN): 19,086 turns, 383 document families, split 15,292 / 1,893 / 1,901.
- Label quality: agent spot-check only (EN 60 in-domain rows: 1 error fixed, 1 debatable;
  RU all 39: 1 noisy). Independent human review: NOT_EVALUATED (sheets in the version's `review/`).

## Gaps (blocking a real KK / RU–KK assistant)

- No licensed natural KK or code-switched RU–KK support text was found (HF search 2026-11-17).
- RU has almost no in-domain support requests; EN lacks 15 intents (e.g. cancel_order,
  human_agent, payment_methods, product_defect).
- Closing these needs new data: commissioned human writing or purchase — a cost decision.

## Next action

Owner: review Stage 1 (optionally fill `review/sample_*.csv`), decide on commissioned KK / RU–KK
data, and authorize Stage 2.
