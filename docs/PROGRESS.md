# PROGRESS — current state

Updated 2026-11-17. Short handoff: state, evidence, next action. Not a log.

## State

| Scope | Status |
|---|---|
| Project reset ([ADR-046](DECISIONS.md#adr-046--project-reset-2026-11-17)) | COMPLETED |
| Stage 1 — Data foundation ([ADR-047](DECISIONS.md#adr-047--stage-1-corpus-from-human-authored-sources-2026-11-17), [ADR-048](DECISIONS.md#adr-048--stage-1-completion-round-2026-11-17)) | PARTIALLY COMPLETED: ROADMAP acceptance checks pass; no natural KK / RU–KK and almost no RU in-domain data exists under a usable licence; independent human review NOT_EVALUATED. Blocked on one owner decision (below) |
| Stages 2–5 | NOT_STARTED; Stage 2 needs owner authorization |

Earlier stages 0–9 are retired. Their code is in Git at `1561971`; their data and evidence
are in `backups/data/`, `artifacts/` and `mlruns/`. Do not resume them.

## What exists now

- CLI: `env`, `check mlflow`, `backup`, `data sources|acquire|verify|taxonomy|corpus`.
- Sources: `configs/sources.yaml` (every investigated source with its decision); raw copies in
  `data/raw/`, verified. Taxonomy v1: `configs/taxonomy.yaml`. Recipe: `configs/corpus.yaml`.
- Corpus `data/corpus/corpus-20261117-26d02054/` (backup `corpus`, pinned; supersedes
  `379aa455`). Reproducible: a rebuild returns the same version (`created: false`). Its
  `REPORT.md` has the full per-language report. Leakage and residual-PII gates: all 0.

| Intent corpus (core) | EN | RU | KK | mixed RU–KK |
|---|---|---|---|---|
| rows / independent families | 34,914 / 30,884 | 10,794 / 10,237 | 636 / 612 | 0 |
| non-out-of-scope rows / families | 6,975 / 6,273 | 59 / 58 | 0 | 0 |
| of which retail-support rows (families) | 6,529 (5,892), role-play | 0 | 0 | 0 |
| v1 intents with ≥1 core row (of 35) | 21 | 4 | 1 (`out_of_scope`) | 0 |

- EN: ABCD role-play (retail) + CLINC150/MASSIVE (greetings, out-of-scope). Missing 14 intents
  (app_help, cart_help, contact_info, delivery_time, feedback, human_agent, invoice, newsletter,
  order_history, payment_methods, personal_data, place_order, product_defect, unclear).
- RU: 39 MINDS-14 banking voice requests (ASR text; `account_help` 24, `technical_issue` 15),
  20 human-localised MASSIVE greetings, 10,735 human-localised out-of-scope rows.
- KK: 636 human-translated xSID out-of-scope rows, validation/test only. No natural KK text.
- Mixed RU–KK: none. The one row v1 flagged was a false positive (fixed detector, ADR-048).
- Grounded (doc2dial, EN): 19,086 turns, 383 document families, split 15,292 / 1,893 / 1,901.

## Label quality

| Method | Result |
|---|---|
| Automatic | schema, licence/policy, dedup, leakage, PII, language-ID gates pass (REPORT.md) |
| Agent review (LLM, not independent) | seeded stratified sample of 307 rows of v1 (`artifacts/reports/stage1_completion/agent_label_audit.csv`): EN in-domain 12/147 wrong, 6 debatable; RU in-domain 24/60 wrong (one systematic rule), 1 debatable; EN / RU / KK out-of-scope 0/30, 0/30, 0/40. All 36 wrong labels are corrected in `26d02054` |
| Independent human review | NOT_EVALUATED (blind sheets in the version's `review/`) |

## Data research (2026-11-17)

HF Hub, arXiv/ACL/TechRxiv, ISSAI, J. Open Humanities Data, GitHub, Kaggle and vendor catalogues.
No new source passed licence + human authorship + support domain (decisions in `sources.yaml`,
section ADR-048). Closest natural KK/RU–KK support data is private (telecom operator) or commercial.

## Next action — one owner decision

Approve commissioning native-speaker data with independent review (no money is spent by
sending the spec for quotes; any purchase needs a second approval with the quoted price):
- what: requests written from scenario briefs by ≥ 3 native writers per slice — KK, mixed RU–KK,
  RU — 35 intents × 10 independent scenarios × 3 writers ≈ 1,050 rows per slice; plus fluent
  reviewers who did not write the data fill the three `review/` sheets (300 rows) and
  double-label 10 % for agreement;
- why: closes the natural in-domain KK / RU–KK / RU gap and the NOT_EVALUATED human review,
  the two items keeping Stage 1 below COMPLETED;
- cost: no confirmed price (FutureBeeAI, Pangeanic, IndecalAI publish none); quotes come first.

Stage 2 must not start before the owner authorizes it.
