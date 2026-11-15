# ARCHITECTURE — components and boundaries

This document separates implemented libraries from future integration.
Current deployment/experiment identities belong to [PROGRESS](PROGRESS.md);
rationale belongs to [DECISIONS](DECISIONS.md), schemas to [DATA_CONTRACTS](DATA_CONTRACTS.md).

## 1. Implemented components

All runtime logic lives in one Python package; no microservice split is justified.

| Package | Responsibility |
|---|---|
| `supportpilot.common` | Typed schemas/settings, IDs/hashes/manifests, atomic IO, redacted logging, grounding diagnostics, MLflow helper |
| `supportpilot.ops` | Environment and service checks; verified snapshot creation/restore |
| `supportpilot.data` | Licensed acquisition, conversion, cleaning/dedup, taxonomy/localisation/MT/QE, SFT/gold/retrieval datasets |
| `supportpilot.kb` | Deterministic document rendering, format parsers, structure/locators, chunking, gates and version activation |
| `supportpilot.retrieval` | Pinned embeddings, dense/sparse indexes, fusion/reranking, date/language filters, context packing, evaluation/recovery |
| `supportpilot.llm` | Prompts, Transformers/PEFT inference, structured decoding, training/resume, selection/seals, experiment evaluation |
| `supportpilot.routing` | Shared RouteDecision, encoder/LLM implementations, deterministic guards and comparison |
| `supportpilot.cli` | Operational entry point; JSON stdout, logs stderr |

Evaluation currently lives in domain modules, not a separate `supportpilot.eval` package.
`supportpilot.service` implements SQLite tools, bounded sessions, guarded orchestration and
the durable operator queue. `supportpilot.api` and `apps/ui.py` are thin HTTP/UI boundaries.

## 2. Implemented online flow (Stage 8)

```text
chat → authorized session + language → router
    → retrieve KB / call authorized tool / clarify / escalate
    → structured generation from packed evidence and tool results
    → schema + evidence + language + claim verification
    → AssistantResponse or safe fallback
```

FastAPI composes existing libraries through `service.runtime.open_service`. Bearer principals
constrain order/payment lookup; stock is an authenticated read of shared synthetic inventory.
Explicit operation/slot recognizers handle supported tool requests. Policy/method terminology
bypasses these shortcuts; policy questions and other messages use the
frozen CPU encoder router. Unknown actions clarify or queue assistance; model-proposed tools
never execute. Cancellation additionally requires explicit confirmation and records a request,
not completed cancellation. Queue insertion and its idempotent response commit before delivery.

Sessions are process-local, TTL/capacity/turn bounded, and store only owned order/SKU slots
and structured intent/delivery summaries. Restart expires sessions; business state, operator
queue and request receipts survive in SQLite. Replayed receipts do not resurrect sessions.
Run one process/worker; distributed sessions/delivery are outside this stage.

An independent multilingual risk guard precedes learned routing. The selected retriever checks
active KB/index readiness on each search. Sources are retrieved excerpts with locators, not
validated answers. Optional NF4 generation uses the existing schema/grounding boundary without
a calibrated semantic verifier: factual candidates are withheld. Generation is off by default
because UNKNOWN cannot deliver factual answers and loading the model consumes resources.
The explicit research mode exercises real inference without changing the sealed assessment.

Stage 9 bounds English risk words and RU/KK root prefixes to avoid internal-word collisions
before authorized tools. It does not add a general semantic risk classifier. Offline system
evaluation, overlap audit, performance and locator-reference studies live in `service/` and
reuse the actual API/runtime; they are not runtime dependencies or a distributed serving layer.
Their draft/raw-routing/delivered-outcome and semantic-calibration boundaries are specified in
DATA_CONTRACTS §19 and EVALUATION §11. ADR-035 owns the narrow intervention and rejected changes.

One component worker serializes LID/routing/retrieval/generation. Deadline expiry discards late
results, which cannot mutate business state; a still-running worker rejects more component work.
GPU driver hangs require process restart; Python cannot forcibly cancel a driver call. Service
requests serialize session/SQLite mutation; concurrent requests receive503. SQLite failures
return503 without claiming queue insertion. This local queue confirms recording, not external
CRM delivery or an operator response. Decisions: ADR-034; commands: RUNBOOK §18.

## 3. Offline flows and storage

```text
source registry → immutable raw revisions → records → clean/canonical
  → native selection + checkpointed MT/QE + altered training KB
  → scenario-disjoint SFT train/validation/test + separate gold/retrieval benchmarks

KB facts/catalogue/document specs → render → parse → structure/chunk → gates
  → immutable KB version + changes/evidence map → active pointer
  → checksummed embedding cache → portable index → materialized Qdrant collection

frozen training input → completion-only LoRA/QLoRA → bound checkpoints/resume
  → verified training adapter → immutable inference export
  → validation selection seal → independent test/gold/safety → raw observations + MLflow
```

Canonical layouts, identities and publication rules are specified in DATA_CONTRACTS.
Models, payloads and reports remain local under the artifact locations in AGENTS §7.

## 4. Actual library boundaries

- `with supportpilot.retrieval.open_retriever(settings) as retriever:` reuses models.
  `retriever.retrieve(...)` returns scored canonical chunks; `retriever.context(...)`
  returns budgeted `RetrievalContext`. Query language and corpus-language filters differ.
- `TransformersBackend(InferenceConfig)` loads a pinned revision and optionally a
  manifest-verified adapter; `generate(messages)` returns raw completion text.
- `generate_grounded(backend, query, language, context, ...)` returns `GenerationResult`:
  raw text, deterministic problems, semantic verdict, fallback flag and AssistantResponse.
- `RouteDecision` is shared by encoder/LLM routers. Experiments supply language explicitly;
  router metrics do not evaluate upstream language detection or confidence calibration.

Schema/few-shot generation, SFT and evaluation share the message formatter.
Selected prompt mode and frozen train-only demonstrations must be supplied explicitly.
The legacy default preserves the earlier interface; it is not the selected schema prompt.

## 5. Safety boundary

Only packed KB chunks and authorized tool results are evidence. Ranked-but-dropped chunks
are unavailable to generation. User/retrieved text is untrusted data, never instructions.
Deterministic validation checks schema, citations and factual surfaces; it cannot establish
full claim support, missing exceptions or semantic answerability.
A candidate is accepted only after deterministic checks and an explicit `supported` result
from ClaimVerifier. The default `unknown` verdict returns localized operator assistance.
It does not claim a completed handoff. Backend execution failures remain recorded failures.

Deterministic routing guards and raw model decisions have separate evaluation scopes.
Neither a confidence field nor nonempty retrieval establishes safe autonomous service.

## 6. KB/index consistency

Immutable KB revisions retain source locators and effective dates. Search representations
project to canonical chunk IDs without creating new evidence.
Embedding caches bind model/protocol/text/role and vector checksums.
Each recipe/namespace has a materialized collection and verified readiness state.
Synchronization marks it unready before changes, validates parent deltas or exact inventories,
and publishes readiness only after payload/vector inventory checks.
Queries reject incomplete or stale KB/index combinations. Version transitions accept brief
unavailability; zero-downtime dual collections are not implemented.
KB activation and index synchronization are separate operations; rollback repeats both.
Sparse IDF is scoped to the live KB version.

## 7. Resource and operational constraints

Use Docker Qdrant with a named volume; memory mode supports tests.
Embedded Qdrant can retain Windows file locks and is unsuitable for larger workloads.
Local GPU/RAM budgets cover retrieval plus generation, not one model in isolation.
Record process RSS, Torch allocation/reservation, Docker memory and shared-memory limits
separately. Optimize only measured bottlenecks.

Config is YAML/Pydantic with environment overrides. Logs redact sensitive data by default.
External IO uses bounded retries/timeouts. Future serving observability, rate limits,
multi-user SLOs and scaling require Stages 8–10; Redis, vLLM and Kubernetes are not dependencies.

## 8. Design references

[DECISIONS](DECISIONS.md#references) records sources and accepted alternatives.
