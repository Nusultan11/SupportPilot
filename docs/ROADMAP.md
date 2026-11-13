# ROADMAP

One stage per work cycle. Never start the next stage without user confirmation.
Status lives in PROGRESS.md, not here.

Dependencies: 0 → 1 → 2 → 3 → 4. Stage 5 needs 1 and can run in parallel with 2–4. Stage 6
needs 5 and the retrieval queries from 4. Stage 7 needs 4 and 6 (grounded SFT needs retrieved
context). Then 8 → 9 → 10.

---

## Stage 0 — Project audit and research

Sub-tasks: audit the folder, git and hardware; research models, embeddings, rerankers, vector
store and data sources; write the instruction and doc set; define the roadmap; take a verified backup; commit.
**DoD:** all docs exist without contradictions; environment facts recorded; open questions
listed for the user; backup and restore verified once; commit on `main`.

## Stage 1 — Architecture and engineering foundation

- `pyproject.toml` + `uv.lock`, Python 3.12, src layout, package skeleton only for modules used in Stages 2–3.
- Ruff (lint + format), mypy, pytest, pre-commit hook running them.
- Torch with CUDA 12.x wheel from the pinned index, plus a GPU smoke check script (`supportpilot.common.env`).
- Config loading (Pydantic settings + YAML), structured logging, deterministic ID helpers, manifest/sha256 helpers.
- Pydantic schemas from DATA_CONTRACTS §2–4 with unit tests.
- Backup and restore CLI for data snapshots (tar + sha256 manifest + retention), restore tested.
- MLflow local backend verified (one dummy run logged and read back).
- Decide whether Qdrant runs in Docker or embedded mode for dev (both checked).

**DoD:** `uv sync` on a clean clone works; lint, type and test checks are green; GPU check
prints a CUDA device; backup and restore round-trip test passes; RUNBOOK commands verified.

## Stage 2 — Data acquisition and source registry

- Research and evaluate candidate sources (DECISIONS ADR-006 list + new search). Record licence, revision, size, purpose and overlaps in `configs/sources.yaml`.
- Downloader with pinned revisions (HF `revision=` commit sha), sha256 manifests, resumable and incremental (skips unchanged revisions).
- Converters from each source to raw canonical rows (`record_id` lineage).
- Tests: manifest integrity, idempotent re-run, unknown licence blocks the source.

**DoD:** every used source has a registry entry with a verified licence; raw data reproducible
from the registry on a clean machine; integrity check passes; source inventory in PROGRESS.md.

## Stage 3 — EDA and data quality engineering

- EDA report per source: counts, unique scenarios, language, intent and category
  distribution, text/token lengths (with Qwen tokenizer), missing and corrupt rows,
  placeholders, template patterns, register tags.
- Qualitative review: ≥ 100 random + ≥ 100 flagged rows per source, findings written down.
- Cleaning (normalisation, artefacts, placeholders → synthetic slot values), exact dedup
  (normalised hash), near-dup (MinHash/LSH and/or embedding similarity), scenario/family clustering.
- Automatic quality gates (DATA_CONTRACTS §10) as code + report.

**DoD:** reproducible `data/interim` with the canonical schema; gates pass (validity ≥ 99.5%);
EDA report in `artifacts/reports/` with key numbers copied to PROGRESS.md; issues listed with decisions.

## Stage 4 — Multilingual dataset engineering

- Freeze the intent taxonomy v1 and the source→SupportPilot intent mapping. Label actions and escalation.
- RU/KK creation: compare MT options on a human-checked sample, then translate and localise
  (KZT, local cities, names, delivery services). Back-translation and QE filters. Kazakh
  quality checks (LID, KK-specific letters, RU leakage, human spot checks).
- Native-style generation for gaps: code-switching, ambiguous requests, complaints,
  out-of-scope, multi-turn. Reviewed samples only.
- Training KB variant (different policy values) and grounded SFT tasks (DATA_CONTRACTS §5),
  including abstention, conflicting context and injection.
- Group split by `scenario_id` (target 90k/5k/5k if quality allows), leakage checks.
- Gold benchmark ≥ 300/lang authored and reviewed per EVALUATION §1.

**DoD:** versioned SFT dataset with a manifest; language distribution reported vs the 30/35/35
plan; leakage = 0; translation error rate measured on a sample; gold benchmark versioned with
review status per item.

## Stage 5 — Knowledge base and document extraction

- Author the synthetic demo-store KB: canonical facts (YAML) rendered to PDF/DOCX/XLSX/CSV/JSON/TXT in RU/KK/EN, with versioned policies.
- Parsers per format. Candidates: PyMuPDF / pdfplumber, Docling, python-docx, openpyxl. Pick by a measured extraction-accuracy test. Hierarchy and table structure kept. OCR only for scanned pages.
- Chunking per doc type (FAQ = Q+A; policy = section-aware; catalogue = row/record), metadata per DATA_CONTRACTS §8.
- Extraction tests: every canonical fact (price, date, period, ID) recoverable from chunks with the correct locator.

**DoD:** 100% of canonical facts found in the chunks with correct locators; incremental update
and delete tested; KB version manifest.

## Stage 6 — Retrieval and RAG experiments

- Retrieval set (per lang + cross-lingual). Embedding candidates: multilingual-e5 (base/large), BGE-M3, Qwen3-Embedding-0.6B (+ any newer model found).
- Qdrant collection with dense + sparse vectors. BM25 variant (Qdrant IDF sparse vs bm25s), with KK tokenisation checked.
- Fusion (RRF vs weighted), rerankers (bge-reranker-v2-m3, Qwen3-Reranker-0.6B), top-k, context assembly with a token budget.
- Comparisons per EVALUATION §5. Latency and memory measured on the laptop.

**DoD:** chosen pipeline backed by a results table with CIs; Recall@5 per language reported
vs target; ADR written; index build is resumable and incremental.

## Stage 6.1 — Large-scale multilingual RAG and robustness evaluation

- Preserve Stage 6 versions, decisions, reports and its exploratory holdout warning.
- Expand the same fictional store to >=200 document IDs, >=600 language versions,
  >=7,500 substantive chunks and >=2,500 canonical facts. Preserve earlier policy values
  and gold citations; live order/payment/stock state belongs to tools. Check semantic
  diversity, multilingual agreement, provenance, locators and overlapping validity periods.
- Build >=2,500 original information needs with RU/KK/EN, >=1,000 cross-language variants,
  >=300 distinct mixed needs and >=750 unanswerable needs. Translations/paraphrases do
  not increase independent needs. Include difficult confusers, incomplete/contradictory
  conditions, multi-document support, temporal cases, typos and rare products.
- Freeze a family-disjoint benchmark before selection. Preserve SFT/gold decontamination
  and label authorship/review status honestly. Use development only for tuning; holdout
  once after sealed selection. An inspected prior holdout is not fresh confirmation.
- Compare the BGE-M3 hybrid control with ranking/fusion/reranking alternatives. Report
  grouped 95% CIs, paired comparisons, Recall@5/MRR@10/nDCG@10, full evidence and packed
  coverage, per-language/directional/mixed/difficulty slices and a real error census.
- Evaluate semantic abstention separately: unanswerable FPR, answerable FNR and abstention
  accuracy, including partial proofs and tool-only requests. Keep unsolved limits open.
- Isolate >=100k synthetic load chunks; measure indexing/update/recovery, exact/ANN,
  p50/p95, throughput, RAM/VRAM and storage. Synthetic load cannot stand in for quality.

**DoD:** every requirement above has verified artifacts; standard checks pass with skipped
checks explicitly unverified; significant data/index/report/MLflow backups are deep-verified
and restored; PROGRESS/EVALUATION/DECISIONS/RUNBOOK updated; commit and push `origin/main`.
Stage 7 does not start in this work cycle. Protocol: `configs/retrieval_scale.yaml` and
`docs/reviews/stage6_scale_protocol.md`. Current execution status is only in PROGRESS.md.

## Stage 7 — LLM baselines and LoRA/QLoRA fine-tuning

- Model audit (ADR-008 shortlist + newer releases): licence, tokenizer efficiency on KK, VRAM, tooling support on Windows/Kaggle.
- Zero-shot and few-shot baselines on validation and gold (JSON validity, intent, grounding, language, abstention, escalation).
- Training pipeline (TRL SFT + PEFT; Unsloth if it measurably helps), checkpoint/resume, manifest, MLflow.
- Smoke test locally, then sweeps on Kaggle/Colab per EVALUATION §5. Router comparison: encoder classifier vs LLM.
- Error analysis per language. Adapter saved with model card.

**DoD:** best configuration vs baseline with paired-bootstrap CIs per language; or a documented
conclusion that fine-tuning does not help enough, with evidence. Artifacts backed up.

## Stage 8 — End-to-end SupportPilot

- Mock APIs (orders, inventory, payments) over a synthetic SQLite DB with deterministic seed data.
- Orchestration per ARCHITECTURE §2: router, tools, retrieval, generation, validators, fallback, escalation queue.
- FastAPI (`/chat`, `/health`, `/ready`, `/escalations`), Streamlit UI with citations and a debug view.
- Tests: API validation, timeouts, tool errors, injection, PII access control, e2e scenarios.

**DoD:** e2e scenario suite runs from one command with per-language results; demo works locally on the laptop.

## Stage 9 — System evaluation and optimisation

- Full benchmark (gold + e2e + safety), latency and throughput load test, memory profile.
- Optimise measured bottlenecks (quantisation, caching, batch sizes, context size).
- 1M+ record stress test of the data and indexing pipelines (synthetic scale-up). Record throughput, RAM and disk.

**DoD:** final metrics table with CIs; scaling limits documented; regressions = 0 against Stage 8 results.

## Stage 10 — Production readiness and delivery

- Dockerfiles + Compose (API, UI, Qdrant), env-based config, CI (lint, type, test, small eval), structured logs, metrics, rate limiting, security review, backup and restore drill, final README and model/data cards.

**DoD:** clean-machine setup from the README works; CI is green; restore drill passes; limitations documented honestly.

---

## Report format (after every stage)

Stage · Status (COMPLETED / BLOCKED / PARTIALLY COMPLETED) · Implemented · Data · Experiments ·
Results (real numbers) · Checks (what ran, passed, could not be verified) · Git and backups
(commits, snapshot IDs, paths) · Problems · Engineering decisions · Next stage · Completion
decision with evidence.
