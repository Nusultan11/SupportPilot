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
No `supportpilot.tools`, `supportpilot.api` or `apps/ui` implementation is supplied.

## 2. Target online flow (Stage 8)

```text
chat → authorized session + language → router
    → retrieve KB / call authorized tool / clarify / escalate
    → structured generation from packed evidence and tool results
    → schema + evidence + language + claim verification
    → AssistantResponse or safe fallback
```

The future API orchestrates these libraries; it must not duplicate business logic.
Planned endpoints: /chat, /health, /ready, /escalations. Mock orders/inventory/payments
use deterministic synthetic SQLite data; Streamlit provides the demo UI.
Session memory is bounded, and customer IDs constrain every business lookup.
Tool execution, a real escalation queue and an actual completed handoff are not implied by
a generated action field. Retries/regeneration policies still require Stage 8 implementation.

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
