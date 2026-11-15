# ARCHITECTURE — current system and target design

Audience: implementers and reviewers completing the multilingual support application.
The single editable C4 model is [workspace.dsl](architecture/workspace.dsl). This document
interprets its boundaries; ROADMAP owns implementation gates, PROGRESS verified state,
DATA_CONTRACTS schemas and DECISIONS alternatives. ADR-038 records this target design.
`Implemented` means code exists, not quality acceptance. `Experimental` marks research paths
that cannot authorize customer facts. `Planned` means unavailable. Target design is a specification
for later authorized implementation, not a claim of delivered functionality.

## 1. Baseline and responsibilities

Keep a modular monolith: FastAPI and service libraries in one process, Streamlit as an HTTP
client. Offline CLI jobs reuse the package with a separate lifecycle. Qdrant and SQLite have
separate persistence boundaries. No measured need justifies microservices, Redis, another
vector store, a workflow platform or a replacement inference framework.

| Code boundary | Implemented responsibility | Limitation / target |
|---|---|---|
| `api.py`, `service/contracts.py` | Bearer principals, strict HTTP contracts, scoped customer/operator access, generic error responses | Local synthetic identities; operational controls planned |
| `service/runtime.py` | Load sealed CPU router, Store, optional retriever/backend; release resources | One worker; no hot promotion/distributed scheduler |
| `service/orchestrator.py` | Session/reference state, independent risk guard, route interpretation, authorized execution, transactional publication | Serialized chat; no autonomous KB-answer delivery |
| `routing/router.py`, `configs/taxonomy.yaml` | Frozen encoder head, shared `RouteDecision`, research routing guards | Uncalibrated confidence; critical-risk quality unaccepted |
| `data/langid.py` | Language/script and code-switch detection | Cyrillic KK; Latin KK unreliable; mixed detection heuristic |
| `service/store.py` | Owned order/payment reads, shared stock, cancellation requests, queue/replay receipts | Synthetic state; cancellation request is not completed cancellation or refund |
| `retrieval/search.py`, `index.py` | Version readiness, dated/language filters, dense/sparse exact search, fusion, rerank and packing | Service corpus filter follows reply language; relevance is not answerability |
| `llm/backend.py`, `generation.py`, `common/grounding.py` | Pinned inference, deterministic diagnostics, `ClaimVerifier` protocol | Research only; default UNKNOWN and service withholding remain enforced |
| `apps/ui.py` | Chat, labeled source excerpts, delivery diagnostics, authorized queue listing | No status editor; operator transitions use the API |
| `data/`, `kb/` | Rights/provenance, preparation, MT/QE, isolated splits; extraction, structure/locators, immutable KB | Draft semantic/linguistic review; born-digital extraction scope |
| Domain evaluation modules in `llm/`, `retrieval/`, `service/` | Training/resume, development selection, frozen assessment, overlap/semantic review, resources | Offline evaluation cannot promote online delivery or consume new holdout implicitly |
| `common/`, `ops/`, `cli.py` | Typed settings/manifests, atomic IO, redacted logs, tracking helpers, recovery and commands | MLflow and engineering MCPs are not online dependencies |

The pending Stage 9 continuation implements `service/decisions.py` over the sealed encoder
and changes queue-reason priority (draft ADR-037, unpublished). Code exists in the working tree; confirmation
and publication remain unfinished. It creates no new process/store or accepted semantic
capability. Archived controls and original assessments remain distinguishable.

## 2. Current online flow and dialogue

1. HTTP validates `ChatRequest`; bearer identity comes from configuration, never text/slots.
   Principal plus `request_id` scopes replay; different input under the same ID returns409.
2. A nonblocking chat lock rejects concurrent work with503. Existing receipts return before
   inference. Foreign/expired sessions return404; failed publication cannot advance a copied
   session. New sessions default to EN until supported detection/preference is available.
3. One component thread performs LID/read/inference work within the request deadline. Explicit
   reply preference overrides detection; supported detection wins over previous language.
   Unsupported/undetermined detection does not establish reliable language support.
4. Critical/requested-human and repeated-failure guards precede learned routing. Explicit
   operations or encoder intent lead to allowlisted tools, KB retrieval, clarification,
   refusal or assistance. Model slots/confidence never authorize execution. Continuation
   interpretation/reference-type corrections await their own confirmation.
5. Missing/ambiguous references clarify; cancellation additionally requires explicit intent
   and confirmation. Store checks ownership. Policy is KB evidence; order/payment/stock
   values come only from authorized Store execution.
6. KB requests retrieve and pack sources. Default generation-off queues assistance. Research
   mode can generate/validate a candidate, but the service still queues even if the library
   could accept a supplied verifier. There is no service path delivering a generated KB answer.
   Authorized tools deliver deterministic returned values directly.
7. Business mutation if any, queue insertion/reuse and serialized `ChatResult` receipt commit
   in one SQLite transaction. Session state publishes only after commit. Storage errors return503
   without a delivery claim. A lost HTTP reply can replay the same ID without repeating writes.

Sessions carry owned references, pending intent, failures and bounded structured summaries;
TTL/capacity/turn limits apply. Restart expires sessions; receipts do not restore them. UI
history is a separate display buffer, not model memory. Generation receives query plus packed
evidence, not an unrestricted transcript. Reference followups and repeated-failure escalation
are finite state behavior, not general dialogue understanding.

## 3. Offline data, ML and release flow

```text
approved pinned sources -> immutable raw/conversion -> cleaning/accounting/EDA
  -> native selection + checkpointed MT/QE + altered training facts
  -> family/scenario-isolated SFT + separate gold/retrieval/system inputs

canonical fictional facts -> rendering -> extraction -> structure/tables/locators
  -> gated immutable KB -> explicit activation -> checksummed portable index
  -> verified Qdrant synchronization

frozen training input -> completion-only LoRA/QLoRA -> bound checkpoints/resume
  -> inference export -> development selection seal -> once-only assessment
  -> retained raw outcomes/failures + MLflow + readiness conclusion

target: qualified code/config/model/router/prompt/KB/index identities
  + quality/security/capacity/recovery evidence -> reviewed release -> guarded operation
```

Evaluation remains in domain modules. Online serving reads qualified artifacts; it never trains,
authors labels or turns raw chats into training examples. Kaggle is authorized free offline
compute with version-specific terminal/source/checksum verification. This design needs no new
cloud run. Small sealed SFT, planned data volume, substantive KB and random-vector stress count
different units; none replaces another's acceptance. Scanned arbitrary-document extraction is
not established: evaluate OCR only when actual scan requirements justify it.

## 4. Interfaces and dependency direction

| Producer -> consumer | Contract / owner | Authority and failure behavior |
|---|---|---|
| UI -> API | `ChatRequest` / `ChatResult`, DATA_CONTRACTS §18 | No client-selected principal;401/403/404/409/422/503 remain distinct |
| Runtime -> service | `open_service(settings)` / `SupportService` | Composition owns model loading/closing and sealed identity checks |
| LID -> service | `LanguageResult` | Advisory detection; explicit preference and known unsupported-script limits |
| Router -> service | `RouteDecision`, taxonomy; DATA_CONTRACTS §17 | Intent proposal interpreted against request/reference/confirmation; no generated tool authority |
| Store -> service | Allowlisted operation + principal + reference -> value mapping | Store owns access/state checks; foreign/absent orders share observable failure |
| Retriever -> generation/UI | `RetrievalContext` / `Source`; DATA_CONTRACTS §15/18 | Only packed canonical chunks with KB/index/date/locator provenance |
| Backend -> verification | Raw text -> `GenerationResult`; DATA_CONTRACTS §6/17 | Deterministic checks precede explicit support verdict; execution exceptions stay failures |
| Service -> Store -> operator | Queue/receipt / `Escalation` | Local recording; operator-only PATCH open -> in_progress -> resolved, no response promise |
| Offline pipeline -> runtime | Immutable manifests/seals | Read-only published inputs; mismatch blocks use instead of replacing bytes |

Target direction remains `UI -> API -> service -> domain adapters -> stores/models`.
Retrieval/routing/generation do not import API/UI. Offline harnesses may compose the service;
request execution must not depend on them, MLflow, Shrimp or Structurizr. Preserve existing
public contracts; breaking storage/API changes need versioning and an ADR before migration.

## 5. Safety and trust boundaries

| Crossing | Current control | Target verification |
|---|---|---|
| Client -> principal | Unique >=24-character configured bearers, constant-time comparison, strict request fields | Stage 10 transport/exposure, rotation and abuse controls before network expansion |
| Request/session -> state | Store ownership, typed references, explicit confirmed cancellation, parameterized SQL | Cross-principal/stale/wrong-type/replay/rollback regressions |
| User/KB/model -> actions | Text is data; independent risk policy; generated tool/slot proposals never execute | Fresh multilingual risk/negation/ambiguity review; finite regex coverage cannot certify it |
| Evidence -> factual answer | Packed evidence; library deterministic checks plus `supported` verifier; service withholds KB answers | Calibrated verifier AND explicit service acceptance wiring, tested for unknown/unsupported/error/timeout |
| Service -> operator | Transactional queue acknowledgement, scoped listing, operator transitions | Actual committed reasons/evidence and restart recovery; external CRM/SLA outside scope |
| Files/Hub/cloud -> runtime | Rights/revision registries, manifests/seals, no arbitrary remote model code | Controlled offline publication; no private code/data/logs sent to research tools |

Only in-date KB and authorized tools support facts. Retrieval scores, model confidence, citation
membership and numeric matching cannot establish semantic support. Dropped chunks are unavailable
to generation. Target verification covers every material claim, exceptions, conflict and
answerability on the exact delivered evidence; candidate self-assertion is not proof.
`unsupported`, `unknown`, verifier failure or deadline expiry prevent factual delivery.
Queue recording is an operational outcome, not an answered policy question.

Logs redact by default; authorized research prompts/candidates remain private. Sessions and
queue summaries exclude raw messages, but replay receipts serialize `ChatResult`, including
retrieved excerpts and delivered synthetic tool values. No receipt expiry is implemented.
Stage 10 must specify access/retention/bounded growth for long-lived operation. Real customer
PII and payment execution remain excluded.

## 6. Storage, consistency and recovery

| Store | Writer / reader | Consistency / recovery unit |
|---|---|---|
| Versioned local files | Offline publisher / runtime and evaluator | Immutable raw/dataset/KB/index/model revisions; atomic per-artifact publication, no global transaction |
| Qdrant Docker named volume | Controlled synchronization / retriever | Mark unready before writes, complete inventory verification before ready, matching KB/index per retrieval |
| Service SQLite | Store / API | Business state + queue + replay in one transaction; sessions intentionally ephemeral |
| MLflow SQLite + artifacts | Offline experiment/import / reviewer | Close writers before snapshot; importer identity differs from execution provenance |
| Recovery archives/bundles | Backup CLI/Git / isolated verification | Pinned SHA256/deep verification/actual restore; same disk leaves disk-loss risk |
| Shrimp task files | One bounded MCP writer / engineer | Private internal work; upstream nontransactional writes; closed-writer snapshots |

KB activation and index synchronization are separate. Brief unavailability is accepted; stale
combinations fail closed. The retriever loaded at process start retains its artifact: synchronize
and reopen the service to adopt a new version. Target release qualification binds these identities
and exercises restart/rollback; no hot reload/zero-downtime promise exists. Sparse IDF stays scoped
to live KB. Restore matching files/index/business state and verify identities before serving.
Rolling back code does not justify rolling back committed business mutations/receipts blindly.

## 7. Target capabilities and implementation sequence

Target: a guarded, measured multilingual support application on existing boundaries. Design IDs
map to [ROADMAP work packages](ROADMAP.md#target-architecture-work-packages); listing is not authorization.

| ID | Gap -> target design | Boundary | Gate and rollback |
|---|---|---|---|
| A1 | Separate language/risk/intent/requested action from authorized action | Existing LID/router/service policy | 9Q1/9Q3 fresh four-slice assessment; revert service intervention only, preserve evidence |
| A2 | Calibrated support over exact packed evidence plus explicit factual-delivery decision | Planned verifier behind `ClaimVerifier` AND service integration | 9Q2/9Q3 independent calibration/untouched assessment; default off/UNKNOWN-to-queue |
| A3 | Bounded owned slots and correct policy/state/mixed-language followups | Session copy/publish and existing request/result types | 9Q1/9Q3 cross-turn/reference tests; restart expires sessions, keeps receipts |
| A4 | Qualified release binding code/config/model/router/prompt/KB/index and evidence | Planned offline release qualification using existing manifests; specify schema before coding | 10P1 clean-machine start/mismatch/rollback, preserve original identities |
| A5 | Observable bounded operation, dependency diagnostics and durable-state growth policy | Planned operational controls in current API/service | 10P2 error-inclusive SLO/security/timeout/retention checks; stop unsafe deployment |
| A6 | Capacity changes justified by workload and joint residency | Existing worker/resource boundary; process isolation only if measured hard-cancellation need | 9Q4 -> 10P3 quality parity and error-inclusive load; retain serialized baseline |
| A7 | Independent data/semantic/linguistic qualification and controlled promotion | Existing offline pipelines plus planned release qualification | 9Q2/9Q3; unavailable reviewers/labels keep gates open; no holdout relabel/retuning |
| A8 | Recoverable long-lived operation with declared recovery objectives | Existing snapshots plus operator procedures | 10P4 restore drill with achieved RPO/RTO; off-device storage/cost needs authorization |

Do not translate every online request through EN. Target reply preference, LID, corpus search
language, model intent, service action and fluent output quality remain separate measurements.
Current service passes reply language as corpus filter. Cross-language fallback search is a
candidate requiring fresh retrieval/delivery evidence, not implemented capability.

## 8. Resources, failures and scaling

Budget the actual 4 GB RTX3050 / about15.4 GB RAM host plus bounded Docker memory. Measure
joint routing/retrieval/rerank/generation, token limits, load/first/warm calls, Python RSS,
Torch allocated/reserved and Docker/shared memory separately. Default generation stays off.
Offline training/stress jobs must not contaminate controlled serving measurements.

| Failure | Implemented behavior | Target qualification |
|---|---|---|
| Component timeout/busy | Discard late results; occupied single component worker rejects further work | Record outcome/error-inclusive latency; measure process isolation before adopting hard cancellation |
| Permanent driver hang | Python cannot forcibly cancel; orderly shutdown can wait for the thread | Operator process restart and durable receipt verification; no bounded graceful-shutdown claim |
| Chat overload | Nonblocking serialized mutation lock returns503 | Qualify single-user workload first; bounded admission needs measured demand and saturation behavior |
| Missing/stale index | Retrieval fails closed; request queues assistance or fails | Dependency-specific diagnostics; do not promote `/ready` to full RAG readiness |
| SQLite failure | Roll back/503 without delivery claim | Disk-full/lock/restart and retention/capacity checks for declared operation |
| Bad model/verifier output | Research candidate withheld, error retained | Target actual HTTP supported/unsupported/unknown/failure tests, not just library mocks |

`/health` is liveness. Current authenticated `/ready` checks closed/busy state and SQLite
quick_check only; it probes neither all models nor Qdrant and certifies no semantic quality/SLO.
Keep its scoped contract and distinguish guarded tool/fallback availability from dependency health.

Scaling order: measure workload; remove unnecessary residency/work; evaluate bounded admission
or isolated inference; consider multiple workers only after session ownership, durable idempotency,
SQLite contention and deployment routing are redesigned/tested together. PostgreSQL/Redis/vLLM/
Kubernetes need a measured bottleneck and a separate migration/cost/rollback ADR. None is selected.
Million random vectors establish numerical load/recovery, not semantic retrieval/service capacity.

## 9. Review and change control

Source review covers the boundary files above, domain pipelines, tests/configuration and unchanged
[Stage 7](reviews/stage7_results.md), [Stage 8](reviews/stage8_results.md), original
[Stage 9 assessment](reviews/stage9_results.md). Continuation evidence has its own protocol/seal
and is not promoted by this design. Accepted dependencies/decisions remain in force.
Private audit/check evidence: `artifacts/reports/architecture/`; final index:
`artifacts/reports/architecture_delivery_receipt.json`.
Structurizr validate/parse/inspect checks DSL structure, tags and views, not code equivalence,
licences, semantic support or acceptance. Recheck affected interfaces, trust boundaries,
persistence and target IDs when code changes; ROADMAP owns package closure evidence.

## 10. Design references

[DECISIONS](DECISIONS.md#references) owns source references and alternatives. This redesign uses
repository evidence and adopted versions; it adds no external service/model/library or runtime code.
