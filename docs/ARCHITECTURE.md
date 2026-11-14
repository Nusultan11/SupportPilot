# ARCHITECTURE

The target design. Status of each component: see PROGRESS.md. Reasons: DECISIONS.md.

## 1. Components

Logical components share one Python package. They are not separate microservices until a measured need appears.

| Component | Package | Responsibility | Runs |
|---|---|---|---|
| Common | `supportpilot.common` | config, logging (redaction), schemas (DATA_CONTRACTS), IDs, hashing, manifests, atomic IO, provenance, MLflow run helper | library |
| Ops | `supportpilot.ops` | environment/GPU report, data/model snapshots, MLflow and Qdrant round-trip checks | `supportpilot` CLI |
| Data pipeline | `supportpilot.data` | registry policy → acquire → convert to raw records (Stage 2) → clean (`textnorm`, `placeholders`, `langid`) → dedup and grouping (`neardup`, `build`) → draft labels (`labels`) → gates → EDA (`eda`, `report`, `review`) (Stage 3) → label v1 (`taxonomy`, `select`) → language adaptation (`locale`, `mt`, `mtrun`, `qe`, `embed`) → training KB (`kbtrain`, `textfmt`) → authored scenarios (`scenarios`) → SFT (`sft`, `sftbuild`, `split`, `sftgates`) → gold (`gold`) (Stage 4) | `supportpilot data ...`, CPU |
| Knowledge ingestion | `supportpilot.kb` | specs + facts (`spec`) → render (`render`, `writers`) → parse PDF/DOCX/XLSX/CSV/JSON/TXT (`parse_pdf`, `parse_office`, `parse_text` → `blocks`) → KBDocument (`structure`) → KBChunk (`chunk`) → gates (`checks`, `locate`) → versioned KB (`build`, `register`) (Stage 5, ADR-027) | `supportpilot kb ...`, CPU |
| Retrieval | `supportpilot.retrieval` | embeddings, Qdrant index, sparse/BM25, fusion, reranking, context assembly | library + index CLI |
| LLM | `supportpilot.llm` | prompts, inference backend, structured decoding, LoRA/QLoRA training | library; training on Kaggle/Colab |
| Routing | `supportpilot.routing` | language ID, intent/action/escalation decision, policy guards | library |
| Mock business APIs | `supportpilot.tools` | orders, inventory, payments over a synthetic SQLite DB | in-process or FastAPI router |
| Evaluation | `supportpilot.eval` | metrics, benchmark runners, reports, MLflow logging | CLI |
| Backend API | `supportpilot.api` | FastAPI `/chat`, `/health`, `/escalations`, admin reindex | uvicorn |
| Web UI | `apps/ui` | Streamlit chat + citations + debug panel | streamlit |

## 2. Online request flow

```
client ──► API /chat ──► Session (last N turns, carried slots e.g. order_id)
                           │
                           ▼
                    Language ID (en | ru | kk, code_switch flag)
                           │
                           ▼
                    Router ──► intent, action, slots, escalate?
          ┌────────────────┼─────────────────────┬───────────────┐
          ▼                ▼                     ▼               ▼
   call_tool(...)   retrieve(query, lang)   ask_clarification  escalate
   orders/stock/     hybrid → rerank →           │               │
   payments          context assembly            │               │
          └───────┬────────┘                     │               │
                  ▼                              │               │
        LLM generation (structured JSON,         │               │
        evidence = tool results + chunks)        │               │
                  ▼                              │               │
        Validators: schema · citations ⊆ evidence · language ·   │
        number/date/price consistency with evidence · policy     │
                  │ fail → regenerate once → fallback/abstain/escalate
                  ▼                              ▼               ▼
               AssistantResponse (DATA_CONTRACTS §6) ◄───────────┘
```

Rules:
- Tool results and KB chunks are the only evidence. The model never sees other customers' data.
  Tools are called with the session's authorised customer id.
- Retrieved text and user text are untrusted. They are wrapped as data in the prompt, and
  instructions inside them are ignored (injection tests in EVALUATION.md).
- The router may be an encoder classifier, the LLM itself, or both (decided in Stage 7). The
  interface `route(message, session) -> RouteDecision` stays fixed.
- Deterministic escalation triggers (explicit "operator", legal threats, safety, repeated
  failure) run before the model.

The Stage 7 implementation lives in `llm/` and `routing/`. `TransformersBackend` loads
an exact model revision and optionally a manifest-verified PEFT adapter. Schema/few-shot
prompts and optional constrained decoding produce raw candidate text; they do not certify
claim support. `generate_grounded` returns a `GenerationResult` containing the raw output,
deterministic problems, semantic verdict and delivered `AssistantResponse`. Only an explicit
`supported` verdict from `ClaimVerifier`, after deterministic checks pass, accepts a candidate.
The default `unknown` verdict produces a localized escalation fallback and makes no claim
that a handoff has already occurred. Citation membership and numeric matching remain
diagnostics, including when all required-looking chunks are present.

`RouteDecision` (`routing/models.py`) is the shared encoder/LLM routing boundary. Router
evaluation separates raw decisions from deterministic guards and fallback. Language is
supplied explicitly in these experiments; those results do not evaluate upstream language
detection. Encoder probabilities are not independently calibrated; LLM confidence is prompted
as null. No routing confidence threshold or semantic verifier is implied by the interface.
These modules do not instantiate the Stage 8 API, tool services or orchestration.
For the measured schema/few-shot configurations, pass the selected `prompt_mode` and
frozen train-only demonstrations to `build_messages`/`generate_grounded`. Those modes
use the same formatter as SFT and evaluation; the default legacy mode preserves the
earlier boundary interface and is not a substitute for the selected prompt.

## 3. Offline flows

```
Data:      configs/sources.yaml (registry + licence policy)
           → data/raw/<source>/<rev>/ (immutable, upstream-verified, RawManifest)      [Stage 2]
           → data/interim/records/<source>/<rev>/records.parquet (RawRecord, lineage) [Stage 2]
           → data/interim/clean/<source>/<rev>/clean.parquet (cleaned, LID, slots, keys) [Stage 3]
           → data/interim/canonical/ (examples/lang=*, pending, rejected, groups, pairs,
             quality_report, manifest; dedup + families + scenarios + gates)            [Stage 3]
           → data/interim/stage4/selected (labels v1, EN backbone, MT queue)             [Stage 4]
           + data/interim/stage4/native (licensed native EN/RU/KK, train-only)          [ADR-028]
           → data/interim/stage4/mt (resumable MT cache) → translated (QE, glossary)      [Stage 4]
           + authored/ (native scenarios, answer templates, training KB, gold sources)    [Stage 4]
           → data/processed/kb_train (training KB variant chunks)                         [Stage 4]
           → data/processed/sft/<dataset_version>/{train,validation,test}.parquet + manifest.json
           → data/gold/<benchmark_version>/ (never used for training/tuning)
           + authored/retrieval/ (independently authored question triples and judgements)
           → data/benchmarks/retrieval/<benchmark_version>/ (active-KB chunk IDs + qrels)

Knowledge: kb_sources/demo_store/ (facts.yaml, catalog.yaml, docs/<doc_id>.yaml specs)       [Stage 5]
           → render → data/kb/demo_store/sources/<lang>/<doc_id>.<pdf|docx|xlsx|csv|json|txt>
             + register.json (CMS metadata) + expected.json (fact ground truth)
           → parse → KBDocument → chunk → KBChunk (+metadata) → 13 gates
           → data/kb/demo_store/versions/<kb_version>/ (immutable: sources copy, documents,
             chunks, changes = chunk ids to delete/upsert, evidence_map) + current.json
           → embed (batch, resumable) → Qdrant collection kb_<schema> (upsert by chunk_id)
           → sparse index (Qdrant sparse vectors or bm25s, decided in Stage 6)

Training:  processed SFT version → (Kaggle/Colab or local) LoRA/QLoRA → adapter + run manifest
           → MLflow run → eval on validation → selected adapter → artifacts/models/<run_id>/
```

## 4. Interfaces (stable signatures, Pydantic models in DATA_CONTRACTS)

```python
detect_language(text) -> LanguageResult
route(message, session) -> RouteDecision
retrieve(query, lang, filters, k) -> list[ScoredChunk]
generate(request: GenerationRequest) -> AssistantResponse
tools: get_order(order_id, customer_id), get_stock(sku), get_payment(order_id, customer_id),
       request_cancellation(order_id, customer_id)  # returns a ticket; no real side effects
```

Every external or IO call has a timeout, bounded retries with backoff (idempotent calls only),
and a typed error that maps to a fallback response.

## 5. Knowledge updates

- A document is identified by `doc_id`, and each revision by `version` (content sha256).
- Reindexing is incremental. For a changed document, delete its old chunks by `doc_id` filter
  and upsert the new ones. Removed documents get a delete. Unchanged documents are skipped.
  No full reindex for single-document edits.
- Chunks carry `effective_from/to`. Retrieval filters out expired policies.

Stage 6 materialisation: `retrieval.corpus` creates search representations retaining
canonical evidence IDs; `encode` supplies pinned model inference and verified resumable
caches; `index` stores portable snapshots and synchronises one dense+sparse Qdrant
collection per recipe. Changed-parent updates validate KB `changes.json`; rollback uses
the inventory diff and cached vectors. Readers fail closed while the collection is being
mutated and until its exact KB-version inventory is verified. No stale-version fallback.
`search` supplies dense/sparse/fusion/reranking, date/document/language filters and
token-budget context packing. `evaluate`, `experiments` and `profile` measure it without
gold input, LLM generation, training or business API execution. The serving lifecycle is
`with supportpilot.retrieval.open_retriever(settings)`; the evidence retains KB locators.

## 6. Scaling path

| Concern | Local MVP | Scaled |
|---|---|---|
| ETL | Polars lazy/streaming over partitioned Parquet (`lang=`, `source=`) | same code on a bigger machine. Parquet in object storage |
| Embeddings | batched GPU inference with checkpointed shards | Kaggle/Colab or dedicated GPU, same shard format |
| Vector store | Qdrant server in Docker (`compose.yaml`); in-memory mode for tests (ADR-014) | Qdrant server/cluster, on-disk vectors, quantisation |
| Inference | 4-bit or GGUF compact model on a 4 GB GPU | vLLM or llama.cpp server on a larger GPU |
| API | single uvicorn process | horizontal replicas behind a load balancer. Sessions in Redis (only when needed) |

Measure the bottleneck first and optimise second. Kubernetes and queues are excluded until proven necessary.

## 7. Cross-cutting

- Config: `configs/*.yaml` loaded into Pydantic settings. Env vars override. Secrets in `.env`.
- Logging: structured JSON logs with request id, stage timings and route decision. No message
  text in production-mode logs (PII). Demo mode may log text.
- Observability: `/health` (liveness), `/ready` (model + Qdrant loaded), latency histograms
  per stage (Stage 10).
- Rate limiting per session on `/chat` (Stage 10).

## 8. References that shaped this design

See DECISIONS.md §References.
