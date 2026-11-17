# ROADMAP — stages and acceptance

Draft after the reset (ADR-046). The owner approves each stage's scope before it starts.
Status lives in [PROGRESS](PROGRESS.md).

## Rules for every stage

1. Start only with owner authorization.
2. Done means: acceptance met with evidence, checks green, docs updated,
   significant results backed up and restore-tested, commit pushed.
3. Report with the format below. Stop and wait for the next authorization.

## Stage 1 — Data foundation

Goal: a versioned, licensed, multilingual (RU/KK/EN/mixed) support dataset.
Acceptance:
- every source is licensed for its use in `configs/sources.yaml`;
- versioned dataset with manifest, SHA256 and per-language counts;
- train/validation/test split by scenario or family, leakage checked;
- per-language quality report, gaps and label provenance stated honestly.

## Stage 2 — Knowledge and tools

Goal: versioned store-policy knowledge source and mock tools (orders, payments, stock, cancellation).
Acceptance: facts are versioned and citable; tools are typed and authorized; tests cover failures.

## Stage 3 — ML pipeline

Goal: baselines, routing (intent/action) and, if justified, fine-tuning.
Acceptance: per-language metrics on validation, one sealed test evaluation, MLflow records,
error analysis (EVALUATION).

## Stage 4 — Service

Goal: end-to-end assistant: routing, evidence, tools, clarification, escalation.
Acceptance: end-to-end scenarios per language, safety cases, health checks.

## Stage 5 — Evaluation and release

Goal: independent quality assessment and a release decision.
Acceptance: targets in EVALUATION met or reported as missed with uncertainty.

## Report format

Stage · Status (COMPLETED / PARTIALLY COMPLETED / BLOCKED) · What changed · Data/experiments ·
Measured results · Checks and unverified items · Git/backups · Problems/decisions · Next action.
