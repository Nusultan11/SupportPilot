# ROADMAP — deliverables and acceptance

Status is owned by [PROGRESS](PROGRESS.md). Complete one development stage per work cycle;
start the next only with owner authorization. Maintenance is an independent engineering task.

Dependencies: 0 → 1 → 2 → 3 → 4; Stage 5 needs foundation and can overlap data work.
Stage 6 needs KB + retrieval queries; 6.1 extends its evidence; Stage 7 needs SFT + retrieval.
Then 8 → 9 → 10. Preserve earlier frozen results when extending a stage.

Historical stages retain their original IDs and delivered evidence. Their completion is not
retroactive certification of every product target; unmet targets carry forward explicitly.
Each stage below has an entry dependency, bounded output and exit evidence. Current states
are deliberately kept only in PROGRESS.

| Stage | Entry dependency | Output handed to the next stage |
|---|---|---|
| 0 | Owner scope + available repository/hardware | Audited requirements, constraints and engineering plan |
| 1 | Stage 0 decisions | Reproducible typed foundation, tests, actual recovery/services |
| 2 | Foundation + approved registry rights | Verified immutable raw/conversion manifests |
| 3 | Verified converted records | Accounted canonical/pending/rejected data and quality review |
| 4 | Canonical data + taxonomy/rights + explicit data scope | Isolated versioned SFT/gold inputs with draft/review limits |
| 5 | Foundation + canonical fictional facts/specifications | Gated source-located immutable KB |
| 6 | KB + separately frozen retrieval needs | Selected measured retrieval and exercised update/rollback |
| 6.1 | Stage 6 baseline + separately authorized expansion | Larger substantive KB/benchmark and isolated load/recovery evidence |
| 7 | SFT + retrieval + licensed pinned model candidates | Sealed behavior model/router assessment and explicit readiness limits |
| 8 | Stage 7 assessment + authorization for guarded integration despite quality limits | Authorized synthetic tools, API/UI, durable assistance and measured demo |
| 9 | Stage 8 + authorized frozen system protocol | Separate engineering assessment and product-quality acceptance decision |
| 10 | Separate owner authorization, after quality acceptance or an explicit changed delivery goal | Qualified guarded release and operated workload/recovery evidence |

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

## Stage transition and implementation control

Before a substantial authorized increment, identify its scope/package IDs, entry evidence,
named control, changed boundary, dependencies, resource/cost scope and rollback. Link an existing
protocol or register a new one only when new experiments need it. Shrimp tracks these internal
deliverables/dependencies; it is neither an approval scheduler nor a second status document.

| Gate | Required evidence to close | When unavailable or failed |
|---|---|---|
| G0 Authorization/entry | Owner-authorized bounded outcome, required artifact identities, unresolved dependency list | Do independent design/checks only; do not start the next stage or infer permission from task status |
| G1 Engineering correctness | Actual affected interfaces/transactions, applicable regressions and full shared checks; DSL validate/parse/inspect when boundaries change | Failed/skipped/unrun scope remains explicit; tests do not establish product quality |
| G2 Product quality | EVALUATION targets and reviewed methodology, separate RU/KK/EN/mixed results, failures/denominators/uncertainty; independent calibration where required | Missed target = not accepted; missing semantic evidence = NOT_EVALUATED; retain negative attempts |
| G3 Safety/authority | Auth/ownership, wrong/stale references, explicit confirmed writes, injection/conflict/unknown/failure handling and actual queue checks | No autonomous facts or unsupported tool actions; widen deployment only after findings resolved |
| G4 Capacity/operation | Declared request mix/concurrency/token limits, load/first/warm timing, error-inclusive throughput/p95, joint memory, dependency and restart behavior | UNKNOWN outside measured workload; queue success or rejected traffic cannot prove an SLO |
| G5 Reproducible delivery | Sealed provenance, reviewed docs/model, deep-verified/restored private results, reviewed public diff, commit/bundle restoration and exact origin/main readback | Engineering delivery incomplete until verified; preserve important private/staged work |

Gate applicability is explicit: architecture-only maintenance closes G0/G1/G5 and reviews G3/G4
as design; it cannot close product G2 or deployment G4 through documentation. A research stage
can deliver an engineering/negative assessment while quality acceptance stays open. Do not mark
the whole product ready because a Shrimp task, test suite or experiment execution finished.

Completion reports name evidence for every applicable gate, distinguish implementation from
quality and operation, and state the next authorized action. If a delivery goal changes, record
the owner decision and remaining restrictions; do not silently waive a failed target.

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
This historical guarded-engineering acceptance does not imply semantic support, a full operator
workspace, production security or sustained concurrent-serving acceptance.

## Stage 9 — System evaluation and optimisation

Gold/e2e/safety evaluation; latency/throughput/resource/load tests; measured quantization,
caching/batch/context trade-offs; isolated1M+ record stress test.
**Engineering outcome:** reproducible per-language system assessment, measured laptop limits,
controlled optimisation decisions and verified million-record recovery. Its delivery can be
complete while product-quality acceptance remains open; see the preserved measured report.
**Quality DoD:** EVALUATION targets assessed separately for RU/KK/EN/mixed, calibrated independent
semantic assessment, adjudicated overlap/critical-risk coverage, no Stage 8 regressions and
shared delivery checks. Missing reviewers/calibration means NOT_EVALUATED, not a waived gate.
Use fresh family-disjoint development/confirmation for further improvements; do not tune exposed
Stage 7 or Stage 9 outcomes. No new data generation/training is authorized by tool maintenance.

[ADR-040](DECISIONS.md#adr-040--owner-accepted-stage-9-local-engineering-and-research-scope)
records the owner's accepted completion scope: local research, engineering implementation and
self-assessment, supported by terminal experiments, retained negative evidence, shared checks,
restore-tested backups and verified publication. This permits stage completion in that scope;
it does not satisfy the separate Quality DoD. Unmet numerical targets remain NOT_MET,
independent human/semantic assessment NOT_EVALUATED and autonomous factual support NOT_READY.
Unsatisfied product-acceptance criteria stay open. PROGRESS owns the current stage state.

### Target architecture work packages

These packages implement ARCHITECTURE §7 (ADR-038) without renumbering completed stages.
They are a dependency plan, not new authorization or claims of completion. Resume the existing
Stage 9 continuation before proposing another experiment; its protocol/seal and once-only markers
remain authoritative. Schema changes, new data scope or further training need their own bounded
authorization. Reviewer/calibration absence is a real dependency, not a software task to fake.

| Package / target IDs | Entry | Concrete output and measurable exit evidence |
|---|---|---|
| 9Q1 Service interpretation and dialogue / A1,A3 | Stage 8 control + current authorized continuation's fresh frozen inputs | Terminal paired control/candidate development and once-only sealed confirmation; all35 classes/four slices, actual tool values/queue/reasons/followups, wrong-type and confirmation regressions; preserve archived control and every deviation/failure |
| 9Q2 Independent qualification / A2,A7 | Verified reviewer identities/fluency, human-labelled calibration, adjudicated overlaps, separately authorized assessment scope | EVALUATION §3/11 calibration and independent assessment requirements, family isolation and critical coverage; numeric targets owned by EVALUATION; otherwise semantic acceptance remains NOT_EVALUATED |
| 9Q3 Grounded delivery and quality acceptance / A1,A2,A3,A7 | 9Q1 terminal interpretation; activation additionally requires 9Q2 independent evidence + fresh authorized development/confirmation | Implement explicit qualified service acceptance with unknown/unsupported/error/timeout fallback; qualify the real assessor and confirm delivered facts/citations/language/critical escalation against EVALUATION targets in RU/KK/EN/mixed. Engineering doubles and a dormant adapter cannot close product acceptance |
| 9Q4 Workload qualification / A6 | 9Q1 behavior identity + declared workload; generation workload additionally needs qualified 9Q3 path | Error-inclusive baseline including auth/tool/queue/KB and optional generation separately, cold/warm/load/joint RAM/VRAM and saturation/restart; freeze target workload and measured limits before choosing capacity changes |

9Q1 development gains alone do not close 9Q2/9Q3. Frozen original Stage 9 engineering/stress
evidence is reused with its limitations, not rerun to populate these packages. Candidate changes
after exposed confirmation require new development/confirmation scope; technical retries preserve
inputs/settings and every prior failure. A missed intent/critical/grounding target keeps the quality
gate open. Grounded-answer/language and generation-serving numerical targets still requiring
definition must be preregistered from development evidence before confirmation, not chosen afterward.

## Stage 10 — Production readiness and delivery

Reproducible packaging/Compose, CI (lint/types/tests/small evaluation), operational logs/
metrics/health/rate limits, security review, backups/restore and model/data cards.
**Entry:** separate owner authorization after Stage 9 quality acceptance, or an explicit owner
decision changing the delivery goal while retaining the unresolved limitations.
Both cases require separate explicit authorization to start Stage 10; acceptance of Stage 9
alone grants none.
**Outcome:** a reproducible, operated guarded deployment with a measured declared workload.
**DoD:** documented clean-machine setup and RU/KK/EN/mixed demo pass; CI runs lint/types/full
applicable tests and a frozen small evaluation; auth/ownership/injection/overload/timeout and
restart/recovery checks pass. Record p95/throughput/error rate and resource budgets for the
declared request mix/concurrency, including rejected work; do not generalize deterministic-tool
latency to generation. Verify persistent state and exact model/data identities after actual
restore. Security findings, observability/rate limits, cards and remaining quality restrictions
are reviewed; shared backup/Git delivery passes. No distributed infrastructure is prespecified:
adopt it only for a measured capacity requirement and an ADR with rollback.

| Package / target IDs | Dependency after authorized Stage 10 entry | Concrete output and exit evidence |
|---|---|---|
| 10P1 Release qualification / A4 | Explicit release scope + frozen qualified component identities | Specify release-binding schema using existing manifests/seals; clean-machine installation/start, code/config/prompt/router/model/KB/index mismatch rejection, explicit KB/index sync + service reopen, version rollback drill without losing committed business state |
| 10P2 Operational and security controls / A5 | 10P1 release + declared exposure/threat model | CI with existing quality commands/small frozen evaluation; redacted request/outcome/dependency diagnostics, authenticated readiness scope, abuse/rate/admission limits, secret handling, receipt/queue retention-growth policy; auth/ownership/injection/storage/timeout/restart regressions and findings reviewed |
| 10P3 Measured capacity / A6 | 9Q4 baseline + 10P1/10P2 controls | Reproduce declared workload and error-inclusive p95/throughput/resources; optimize only observed bottleneck with quality parity and rollback; retain single-worker baseline. Multiple workers need tested session/idempotency/storage ownership plus ADR, never only a Uvicorn worker flag |
| 10P4 Recovery and handover / A8 | 10P1–10P3 evidence + declared recovery objectives | Actual release + business DB + index/file restore, model/data cards, achieved RPO/RTO, operator handover and local/off-device loss limitations; off-device transfer/service/cost only when separately authorized; exact reviewed Git publication |

Production acceptance closes G1–G5 for the declared delivery goal and workload. A narrower guarded
goal, if explicitly chosen by the owner, retains withheld factual generation and all unaccepted
quality dimensions in the release/readiness statement. External CRM, real PII/payment execution,
OCR and distributed infrastructure are not implicit work packages.

## Report format

Stage or maintenance task · Status (COMPLETED / BLOCKED / PARTIALLY COMPLETED) ·
Implemented · Data/experiments (when applicable) · Measured results · Checks/unverified items ·
Git/backups · Problems/decisions · Next action · Completion decision with evidence.
