# ROADMAP — deliverables and acceptance

Status is owned by [PROGRESS](PROGRESS.md). Complete one development stage per work cycle;
start the next only with owner authorization. Maintenance is an independent engineering task.

Dependencies: 0 → 1 → 2 → 3 → 4; Stage 5 needs foundation and can overlap data work.
Stage 6 needs KB + retrieval queries; 6.1 extends its evidence; Stage 7 needs SFT + retrieval.
Then 8 → 9 → 10. Preserve earlier frozen results when extending a stage.

## Shared acceptance

Every delivery requires appropriate unit/integration checks, Ruff check/format, strict mypy,
full pytest with skipped checks explicitly unverified, accurate owning documents, significant
artifact backup/deep verification/actual restore, reviewed commit and push to origin/main.
Measurements retain failures, denominators, language slices and uncertainty.
A missed quality target is a documented negative result, never a fabricated pass.

Each owner-authorized large stage/task has one bounded outcome and the criteria below.
Shrimp tracks its internal dependency graph, evidence and remaining work; it neither replaces
this roadmap/PROGRESS nor authorizes another stage. Resume from verified evidence before
creating new tasks. Completed research stages are not rerun to populate a task tracker.
Tool/model maintenance is a separate engineering delivery with the same checks/recovery/Git
requirements, not an implicit authorization for Stage 9 quality experiments or Stage 10.

## Stage 0 — Project audit and research

Audit repository/hardware and research data/model/retrieval options. Establish canonical
instructions, requirements, architecture, contracts, evaluation and roadmap.
**DoD:** consistent documents, recorded constraints/open decisions, verified backup/restore,
commit on main.

## Stage 1 — Architecture and engineering foundation

Python 3.12/uv lock/src package; typed settings/schemas, deterministic IDs/hashes/manifests,
atomic IO/redacted logs, CUDA environment check, local MLflow, Qdrant mode decision,
snapshot/restore CLI, pytest/Ruff/mypy/pre-commit.
**DoD:** reproducible clean-clone setup, actual CUDA/service round trips, working backup/restore,
green checks and verified RUNBOOK.

## Stage 2 — Data acquisition and source registry

Review licences/revisions/use/overlaps; implement pinned resumable downloads with upstream
integrity verification and streaming raw-record conversion. Reject unknown/prohibited licences.
**DoD:** every used source registered, original files reproducible and immutable, idempotency
and integrity verified, acquisition/conversion manifests and inventory available.

## Stage 3 — EDA and data quality engineering

Profile source/language/intent/length/register/defects; review >=100 random and up to100
available flagged rows per source. Normalize/mask PII/fill synthetic slots, exact/near dedup,
family/scenario grouping and executable quality gates.
**DoD:** accounting and canonical contracts pass, validity >=99.5%, reproducible interim
outputs/EDA/review with provenance and unresolved quality issues.

## Stage 4 — Multilingual dataset engineering

Freeze taxonomy/source mapping; compare/localize/filter RU/KK MT with QE/back-translation;
add native/authored gaps (mixed, ambiguity, complaints, OOS, multi-turn).
Use altered training KB values and grounded/abstention/conflict/injection targets.
Split by scenario/family and decontaminate; build separate gold >=300 items/language.
**DoD:** versioned manifests/gates, exact leakage exclusions, reported language distribution
against30/35/35 target, sample translation errors and honest per-item review status.
The quality-bounded SFT volume/split/language targets in PROJECT remain in force.
Fluent independent review is required for a human-verification claim.

## Stage 5 — Knowledge base and document extraction

Render canonical synthetic facts into PDF/DOCX/XLSX/CSV/JSON/TXT across RU/KK/EN.
Measure parser alternatives; preserve hierarchy/tables/locators; chunk by document type.
**DoD:** 100% canonical facts recovered at correct source locators, metadata/language/evidence
gates pass, incremental change/delete and immutable version/activation demonstrated.
Evaluate OCR only when scans require it.

## Stage 6 — Retrieval and RAG experiments

Compare licensed pinned multilingual encoders (E5/BGE/Qwen and relevant alternatives),
dense/sparse Qdrant/BM25S, fusion, rerankers, top-k, chunk views and budgeted context.
Measure mono/cross/mixed retrieval, latency and memory; build resumable incremental indexes.
**DoD:** selected pipeline has per-language ranking results/paired CIs vs targets,
documented alternatives/limits, an ADR and exercised update/rollback/recovery.

## Stage 6.1 — Large-scale multilingual RAG and robustness evaluation

- Preserve original KB values/citations, baseline results and exposed-holdout warning.
- Expand to >=200 document IDs, >=600 language versions, >=7,500 substantive chunks,
  >=2,500 canonical facts; audit diversity, locators, multilingual agreement and validity.
- Build >=2,500 original needs, >=1,000 cross variants, >=300 mixed needs and >=750
  unanswerable needs. Translations/slots/paraphrases do not add independent needs.
- Cover confusers, multi-document/partial/conflicting support, temporal/rare/typo cases.
  Freeze family-disjoint development/holdout after leakage/coverage audits.
- Compare ranking/fusion/reranking controls; report Recall/MRR/nDCG, full/packed proof,
  language/direction/mixed/difficulty slices, scenario/family95% CIs and error census.
- Report unanswerable FPR, answerable FNR and abstention separately, including unresolved limits.
- Isolate >=100k synthetic load chunks; measure exact/ANN agreement, upload/build,
  update/recovery, latency/throughput, RAM/VRAM/storage. Synthetic load is not quality.

**DoD:** all volume/diversity/protocol requirements have verified evidence and shared checks/
delivery requirements pass. Protocol: `configs/retrieval_scale.yaml` and immutable
[preregistration](reviews/stage6_scale_protocol.md). No semantic or human acceptance is
implied by synthetic benchmark execution.

## Stage 7 — Multilingual LLM baselines, grounding and fine-tuning

- Audit compact candidates: licence/full revision, KK tokenization, measured RU/KK/EN,
  Windows/Kaggle compatibility, memory and speed.
- Real inference and AssistantResponse boundary; compare zero-/few-shot/schema/constrained
  modes. Keep raw output, deterministic validation and delivered fallback separate.
- Full validation baselines with language/task slices; preserve initial0/6 negative smoke.
- Reproducible completion-only TRL/PEFT LoRA/QLoRA comparisons (rank/LR/targets/length/
  packing/precision), checkpoint/resume, reloaded adapter and >=3 seeds where feasible.
- Compare encoder and LLM routing across 35 intents with macro-F1, actions, escalation,
  latency and resources; retain a typed RouteDecision.
- Seal model/prompt/adapter/router before independent test/gold; no retuning from holdout.
  Report paired scenario95% CIs and family sensitivity for finalists.
- Execute safety >=50 needs/language and full/partial/irrelevant/absent/conflicting
  evidence studies, multi-document support and packing loss; distinguish pipeline failure causes.
- Report schema/intent/action/language/abstention/escalation and credible semantic metrics.
  Missing calibrated judges or fluent reviewers require NOT_EVALUATED and a readiness limitation.
- Preserve model card, configs, predictions, adapters/checkpoints, manifests and MLflow.
  Measure cost scopes, latency/throughput, sizes and local RTX 3050 isolated/joint compatibility.

**DoD:** every requested experiment has terminal/source/checksum-verified accounting;
best configuration vs baseline has per-language paired CIs, or a supported negative conclusion.
State readiness for Stage 8, including semantic/KK/resource/critical-escalation limits;
backups/restores and shared engineering checks pass. Do not implement Stage 8.

## Stage 8 — End-to-end SupportPilot

Authorized synthetic orders/inventory/payments SQLite APIs; router/tools/retrieval/generation/
validators/fallback/escalation orchestration; FastAPI /chat,/health,/ready,/escalations;
Streamlit citations/debug view; auth/error/timeout/injection/PII and multi-turn scenarios.
**DoD:** one-command multilingual e2e suite and local laptop demo work.

## Stage 9 — System evaluation and optimisation

Gold/e2e/safety evaluation; latency/throughput/resource/load tests; measured quantization,
caching/batch/context trade-offs; isolated1M+ record stress test.
**Engineering outcome:** reproducible per-language system assessment, measured laptop limits,
controlled optimisation decisions and verified million-record recovery. Its delivery can be
complete while quality acceptance remains PARTIALLY COMPLETED; see the preserved measured report.
**Quality DoD:** EVALUATION targets assessed separately for RU/KK/EN/mixed, calibrated independent
semantic assessment, adjudicated overlap/critical-risk coverage, no Stage 8 regressions and
shared delivery checks. Missing reviewers/calibration means NOT_EVALUATED, not a waived gate.
Use fresh family-disjoint development/confirmation for further improvements; do not tune exposed
Stage 7 or Stage 9 outcomes. No new data generation/training is authorized by tool maintenance.

## Stage 10 — Production readiness and delivery

Reproducible packaging/Compose, CI (lint/types/tests/small evaluation), operational logs/
metrics/health/rate limits, security review, backups/restore and model/data cards.
**Entry:** separate owner authorization after Stage 9 quality acceptance, or an explicit owner
decision changing the delivery goal while retaining the unresolved limitations.
**Outcome:** a reproducible, operated guarded deployment with a measured declared workload.
**DoD:** documented clean-machine setup and RU/KK/EN/mixed demo pass; CI runs lint/types/full
applicable tests and a frozen small evaluation; auth/ownership/injection/overload/timeout and
restart/recovery checks pass. Record p95/throughput/error rate and resource budgets for the
declared request mix/concurrency, including rejected work; do not generalize deterministic-tool
latency to generation. Verify persistent state and exact model/data identities after actual
restore. Security findings, observability/rate limits, cards and remaining quality restrictions
are reviewed; shared backup/Git delivery passes. No distributed infrastructure is prespecified:
adopt it only for a measured capacity requirement and an ADR with rollback.

## Report format

Stage or maintenance task · Status (COMPLETED / BLOCKED / PARTIALLY COMPLETED) ·
Implemented · Data/experiments (when applicable) · Measured results · Checks/unverified items ·
Git/backups · Problems/decisions · Next action · Completion decision with evidence.
