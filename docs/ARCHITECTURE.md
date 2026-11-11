# ARCHITECTURE

The target design. Status of each component: see PROGRESS.md. Reasons: DECISIONS.md.

## 1. Components

Logical components share one Python package. They are not separate microservices until a measured need appears.

| Component | Package | Responsibility | Runs |
|---|---|---|---|
| Common | `supportpilot.common` | config, logging (redaction), schemas (DATA_CONTRACTS), IDs, hashing, manifests, atomic IO, provenance, MLflow run helper | library |
| Ops | `supportpilot.ops` | environment/GPU report, data/model snapshots, MLflow and Qdrant round-trip checks | `supportpilot` CLI |
| Data pipeline | `supportpilot.data` | registry policy → acquire → convert to raw records (Stage 2) → clean (`textnorm`, `placeholders`, `langid`) → dedup and grouping (`neardup`, `build`) → draft labels (`labels`) → gates → EDA (`eda`, `report`, `review`) (Stage 3) → language adaptation → label v1 → build SFT → split → version | `supportpilot data ...`, CPU |
| Knowledge ingestion | `supportpilot.kb` | parse PDF/DOCX/XLSX/CSV/JSON/TXT → structured docs → chunks + metadata | batch CLI, CPU |
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

## 3. Offline flows

```
Data:      configs/sources.yaml (registry + licence policy)
           → data/raw/<source>/<rev>/ (immutable, upstream-verified, RawManifest)      [Stage 2]
           → data/interim/records/<source>/<rev>/records.parquet (RawRecord, lineage) [Stage 2]
           → data/interim/clean/<source>/<rev>/clean.parquet (cleaned, LID, slots, keys) [Stage 3]
           → data/interim/canonical/ (examples/lang=*, pending, rejected, groups, pairs,
             quality_report, manifest; dedup + families + scenarios + gates)            [Stage 3]
           → language adaptation, labels v1, splits                                     [Stage 4]
           → data/processed/sft/<dataset_version>/{train,validation,test}.parquet + manifest.json
           → data/gold/<benchmark_version>/ (never used for training/tuning)

Knowledge: kb_sources/ (synthetic store docs in several formats)
           → parse → KBDocument → chunk → KBChunk (+metadata)
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
