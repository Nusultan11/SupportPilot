# DECISIONS — architecture decision records

Format: context → decision → alternatives → risks → revisit when. Status: `accepted`,
`provisional` (to be confirmed by an experiment in the stage named), or `superseded`.

---

### ADR-001 Single canonical agent instructions — accepted (2026-11-11)
- **Decision:** `AGENTS.md` is the only rules file. `CLAUDE.md` contains just `@AGENTS.md` (Claude Code import).
- **Alternatives:** one file per tool (drifts out of sync); rules inside README (mixes audiences).
- **Revisit:** a tool that cannot read either file.

### ADR-002 Documentation language: English — accepted (2026-11-11)
- **Decision:** docs, code, comments and commit messages in English. Reports to the user in Russian.
- **Why:** fewer tokens for agents than Cyrillic; matches library and model terminology; portfolio readers.
- **Alternative:** Russian docs (more natural for the owner, but costlier for agents).
- **Revisit:** if the owner prefers Russian docs.

### ADR-003 Runtime: Python 3.12 + uv, Windows-native first — accepted (2026-11-11)
- **Context:** Python 3.12.3 and 3.13 installed; torch 2.12.0+cu126 works in the global 3.12 install; uv 0.12.1; Docker 29.8 on WSL2.
- **Decision:** project venv managed by `uv` (lockfile), Python 3.12 (best wheel coverage for torch/bitsandbytes/flash-linear-attention). Linux-only pieces (Qdrant server; possibly Qwen3.5 kernels such as causal-conv1d) run in Docker/WSL2 or on Kaggle/Colab.
- **Alternatives:** conda (heavier, slower); WSL2 for everything (doubles the environment; Windows GPU works already).
- **Revisit:** a required training dependency with no Windows build → move training env to WSL2.

### ADR-004 Data versioning via manifests + content hashes — accepted (2026-11-11)
- **Decision:** immutable raw revisions, a `manifest.json` per dataset/index (DATA_CONTRACTS §9), snapshots backed up outside the repo.
- **Alternatives:** DVC (remote storage, extra tooling; useful later for sharing); LakeFS (overkill).
- **Revisit:** when data must be shared between machines or collaborators → adopt DVC with the same layout.

### ADR-005 Tabular processing: Parquet + Polars (lazy/streaming) — accepted (2026-11-11)
- **Alternatives:** pandas (higher RAM, no lazy engine); Spark (cluster overhead); DuckDB (good; allowed for ad-hoc EDA SQL, not a second pipeline engine).
- **Revisit:** a measured Polars limitation on 1M+ workloads.

### ADR-006 Training-data backbone and language adaptation — provisional (Stage 2–4)
- **Context:** No open Kazakh e-commerce support dataset was found. The best licensed EN data:
  - Bitext retail e-commerce: 44,884 rows, 46 intents / 13 categories, CDLA-Sharing-1.0, hybrid-synthetic, `{{placeholders}}`.
  - Bitext customer support: 26,872 rows, 27 intents / 11 categories (card lists are inconsistent → verify on data), CDLA-Sharing-1.0.
- **Decision (provisional):** Bitext EN as the scenario backbone. RU/KK produced by MT +
  localisation + filtering, plus reviewed native-style generation for code-switching, ambiguity
  and complaints. 100k is a ceiling to reach only via genuinely distinct examples. Bitext is
  heavily templated (paraphrase families), so the unique-scenario count will be far lower than
  the row count. Measure it in Stage 3.
- **Other candidates to evaluate in Stage 2:** KazParC (KK/EN/RU/TR parallel, CC BY 4.0 per paper; MT evaluation), Tilmash MT model (CC BY 4.0 per paper), NLLB-200 (licence CC-BY-NC-4.0 → eval/reference only unless the use is confirmed acceptable), KazQAD (CC BY-SA), Belebele-kk and KazMMLU (LLM language competence eval), RuNLUIntentClassification (cc-by-4.0, RU diagnostic), wb-questions (CC0, RU marketplace Q&A style), ISSAI KazLLM (CC-BY-NC → not for generating training data by default).
- **Licence obligation:** CDLA-Sharing-1.0 → any *published* derived dataset must carry the same licence. Model weights are not "data" under CDLA, but this is recorded for review.
- **Risks:** MT errors in KK; RU-isms in KK; templated style → low naturalness. Mitigated by QE, back-translation, human spot checks, and gold built separately.
- **Revisit:** a licensed native KK/RU support dataset appears; MT quality fails Stage 4 gates.

### ADR-007 Vector store: Qdrant; FAISS only as a comparison baseline — provisional (Stage 6)
- **Why:** payload filters (lang, doc_type, effective dates), named dense + sparse vectors with server-side fusion (RRF), point-id upserts/deletes for incremental updates, local embedded mode for tests, Docker for dev, a clear scale-out path.
- **Alternatives:** FAISS (fast, but no metadata or update model of its own); pgvector (extra DB); Chroma (weaker hybrid/scaling story).
- **Revisit:** Stage 6 measurements show Qdrant overhead dominating latency on the laptop.

### ADR-008 LLM candidates and training mode — provisional (Stage 7)
| Candidate | Params | Licence | Notes |
|---|---|---|---|
| Qwen3-0.6B / 1.7B | 0.6B / 1.7B | Apache-2.0 | 119 languages (KK not explicitly listed); mature PEFT/Unsloth support; QLoRA 4-bit fits 4 GB for 0.6B, tight for 1.7B. A community card reports Belebele-kk 41.7% for 1.7B (unverified) |
| Qwen3.5-0.8B / 2B | 0.9B / 2B | Apache-2.0 | released 2026-03; 201 languages (KK not listed); hybrid Gated DeltaNet + attention, vision encoder; needs recent transformers + fla/causal-conv1d kernels; Unsloth advises **against QLoRA** for Qwen3.5 → bf16 LoRA on cloud GPU |
| Gemma 4 E2B | ~2.3B effective (5.1B incl. per-layer embeddings) | Apache-2.0 | 140+ pre-training languages; heavier memory than its "effective" size |
- **Decision:** select by Stage 7 measurements, not reputation. Gates: KK zero-shot quality (Belebele-kk + our gold-kk), JSON validity, tokens per KK word, VRAM at inference ≤ 4 GB with 4-bit/GGUF, Windows/Kaggle tooling works. Local laptop: smoke tests and ≤ 1B QLoRA. Full runs: Kaggle/Colab.
- **Fine-tuning scope:** behaviour only (ADR-009). LoRA hyper-parameters in EVALUATION §5 are candidates, not defaults.
- **Revisit:** a newer compact model with explicit Kazakh support and a permissive licence.

### ADR-009 Knowledge in RAG and APIs, behaviour in weights — accepted (2026-11-11)
- **Decision:** policies, prices, catalogue → KB with citations. Orders, stock, payments → tools. SFT uses a *training KB variant* with altered values so the model learns to read context, not memorise.
- **Alternative:** fine-tune facts in (stale on every policy change, unverifiable, hallucination-prone).

### ADR-010 Structured response contract — accepted (2026-11-11)
- **Decision:** the model emits `AssistantResponse` JSON (DATA_CONTRACTS §6), validated server-side (schema, citations ⊆ evidence, numbers in evidence, language). Constrained decoding where the backend supports it.
- **Alternative:** free text + post-hoc parsing (no reliable citation/escalation signals).

### ADR-011 Experiment tracking: MLflow, local backend — accepted (2026-11-11)
- **Alternatives:** W&B (external account, data leaves the machine); plain CSV logs (no comparison UI).
- **Revisit:** multi-machine collaboration needs a shared tracking server.

### ADR-012 Skills and MCP: none at Stage 0 — accepted (2026-11-11)
- **Decision:** no project skills or MCP servers yet. There is no repeated process to capture.
  HF access via `huggingface_hub`; research via built-in web tools.
  Candidate skills once the processes exist: `data-quality-audit` (Stage 3), `rag-eval` (Stage 6), `experiment-review` (Stage 7), `release-readiness` (Stage 10).
- **Revisit:** at the end of each of those stages, if the process was run ≥ 2 times by hand.

### ADR-013 Git on `main` only; backups outside the work tree — accepted (2026-11-11)
- **Decision:** project rule set by the owner. Code/docs backups are `git bundle` snapshots; data/model snapshots are tar + sha256 manifests (Stage 1 tool). Both go to `C:\Users\nurs\SupportPilot-backups\`, with retention: last 10 code bundles, last 3 snapshots per dataset/model, plus all stage-end snapshots.
- **Revisit:** a remote repository (GitHub) is added → push also becomes an off-machine copy.

---

## Open decisions (owner input needed; see PROGRESS.md)
1. Budget for a paid LLM API (MT for KK/RU, data generation, LLM judge). Without it: open MT + local models, with lower expected KK quality.
2. Availability of fluent KK and RU reviewers for the gold benchmark and translation audits.
3. GitHub remote (private/public) for off-machine code backup and CI.

## References
- Qwen3 blog / tech report: https://qwenlm.github.io/blog/qwen3/ , https://arxiv.org/abs/2505.09388
- Qwen3.5 small model cards: https://huggingface.co/Qwen/Qwen3.5-0.8B , https://huggingface.co/Qwen/Qwen3.5-2B
- Unsloth Qwen3.5 fine-tuning guide (QLoRA not recommended): https://unsloth.ai/docs/models/qwen3.5/fine-tune
- Gemma 4 model card: https://ai.google.dev/gemma/docs/core/model_card_4
- Farabi-1.7B card (Qwen3-1.7B Belebele-kk figure, community): https://huggingface.co/nur-dev/farabi-1.7b
- SozKZ (small Kazakh LMs, tokenizer fragmentation): https://arxiv.org/abs/2603.20854
- KazMMLU: https://arxiv.org/abs/2502.12829
- Bitext datasets: https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset , https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- KazParC / Tilmash: https://arxiv.org/abs/2403.19399 , https://huggingface.co/datasets/issai/kazparc
- Qwen3 Embedding / Reranker: https://arxiv.org/abs/2506.05176
- MMTEB: https://arxiv.org/abs/2502.13595
- Qdrant hybrid search: https://qdrant.tech/course/beginners/module-3/hybrid-search-in-qdrant/
