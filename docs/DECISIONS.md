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
- **Context:** Python 3.12.3 and 3.13 installed; torch 2.12.0+cu126 works in the global 3.12 install; uv 0.12.1; Docker 29.8 on WSL2. Stage 1: the project venv uses torch 2.14.1+cu126 (ADR-015).
- **Decision:** project venv managed by `uv` (lockfile), Python 3.12 (best wheel coverage for torch/bitsandbytes/flash-linear-attention). Linux-only pieces (Qdrant server; possibly Qwen3.5 kernels such as causal-conv1d) run in Docker/WSL2 or on Kaggle/Colab.
- **Alternatives:** conda (heavier, slower); WSL2 for everything (doubles the environment; Windows GPU works already).
- **Revisit:** a required training dependency with no Windows build → move training env to WSL2.

### ADR-004 Data versioning via manifests + content hashes — accepted (2026-11-11)
- **Decision:** immutable raw revisions, a `manifest.json` per dataset/index (DATA_CONTRACTS §9), snapshots backed up in the git-ignored `backups/` dir (ADR-013).
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
- **Stage 2 outcome (2026-11-11):** 16 sources reviewed and recorded in `configs/sources.yaml`.
  - **Acquired:**
    - `train`: Bitext retail and Bitext customer support (backbone); while-ai e-commerce intent (CC-BY-4.0, naturalistic EN dialogues; inclusion decided by Stage 3 quality).
    - `eval_only`: Belebele kk/ru/en.
    - `reference_only`: support-json-ru (CC-BY-4.0, RU policy-conditioned decisions, SaaS domain).
  - **Gated, so unavailable without an HF account:** KazParC, FLORES+, KazQAD.
  - **Rejected, with reasons:**
    - kazakh-instruction-v2: licence conflict (says MIT, is translated Alpaca, which is CC-BY-NC);
    - wb-questions: scraped real-user content, so it stays a candidate and is not acquired;
    - the rest: off-domain, unverifiable provenance, or NC.
  - No usable native KK customer-support data exists among public sources. The plan above (MT + localisation + generation) stands.
  - Not dataset sources, so not in the registry: NLLB-200 (CC-BY-NC-4.0), Tilmash (CC BY 4.0 per paper) and KazLLM (CC-BY-NC). They are MT/LLM options for Stage 4.

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
- **Stage 3 check (2026-11-11):** the data-quality audit ran once, and it is now code
  (`data clean`, `data gates`, `data eda`), not a manual process. So no skill yet. Revisit
  when Stage 4 repeats the review on translated data.
- **Revisit:** at the end of each of those stages, if the process was run ≥ 2 times by hand.

### ADR-013 Git on `main` only; backups in the git-ignored `backups/` dir — accepted (2026-11-11, location changed the same day)
- **Decision:** project rule set by the owner.
  - Code/docs backups are `git bundle` snapshots. Data/model snapshots are tar + sha256 manifests (Stage 1 tool).
  - Both go to `<project>/backups/` (`code/`, `data/`). That directory is in `.gitignore`
    (`/backups/`), and the backup tool refuses any root inside the repo that git does not ignore.
  - Retention: last 10 code bundles, last 3 snapshots per dataset/model, plus all stage-end snapshots.
- **Change (2026-11-11, owner request):** the location moved from `C:\Users\nurs\SupportPilot-backups\` (outside the work tree) to the project root.
  - All 4 bundles and 8 snapshots were moved and re-verified: sha256, `git bundle verify`, deep snapshot verify, and a test restore.
- **Accepted risk:** backups now share a folder with the project. Deleting or losing the project directory, or the disk, also loses the data snapshots. Code is still also on GitHub. Ruff and pre-commit ignore the directory through `.gitignore`, and pytest/mypy only scan `src/` and `tests/`.
- **Revisit:** before any destructive operation on the project directory, or when data snapshots become expensive to recreate (e.g. trained adapters). Then also copy `backups/` to an external disk.
- **Remote:** `origin` = https://github.com/Nusultan11/SupportPilot (public, added 2026-11-11). Pushing `main` after each stage gives an off-machine code copy. Data and model snapshots stay local, because the repo is public.

### ADR-014 Qdrant dev mode: Docker server; in-memory for tests — accepted (2026-11-11)
- **Context:** ADR-007 named Qdrant. Stage 1 had to choose how it runs locally on Windows. All three modes passed the same round trip (named dense + IDF sparse vectors, payload filter, prefetch + RRF, delete-by-filter). Benchmark: 1024-dim cosine, filtered top-10, 50 queries.

  | Mode | Upsert, 5k | Query p50 / p95, 5k | Upsert, 50k | Query p50 / p95, 50k |
  |---|---|---|---|---|
  | embedded (`path=`) | 364 pts/s | 71 / 83 ms | 198 pts/s (253 s total) | 604 / 716 ms |
  | server v1.20.0 (Docker) | 1950 pts/s | 6 / 27 ms | 1276 pts/s | 11 / 61 ms |

  The server container used 277 MiB at 50k points. The embedded client keeps `storage.sqlite` locked on Windows until the process exits.
- **Decision:**
  - **Server** (`compose.yaml`, `qdrant/qdrant:v1.20.0` matching qdrant-client 1.20.0) for development, indexing, Stage 6 experiments and e2e.
  - **memory** mode for unit and integration tests (no Docker).
  - **embedded** only for quick experiments under about 5k points.
  - Storage uses a *named* Docker volume, because Qdrant warns that Windows bind mounts are not POSIX-compatible. Persistence across `compose down/up` was verified.
  - Ports are bound to 127.0.0.1. The client URL is `http://127.0.0.1:6333`, because `localhost` cost about 2 s per call (IPv6 first).
- **Alternatives:** embedded only (no Docker, but 50× slower queries at 50k, brute force, and file locks); Qdrant Cloud free tier (data leaves the machine, needs an account).
- **Revisit:** Docker Desktop's ~6 GB VM becomes a RAM problem together with training/inference on the 15 GB laptop.

### ADR-015 Dependency layout and CUDA torch — accepted (2026-11-11)
- **Decision:**
  - Runtime deps in `[project]`: pydantic, pydantic-settings, pyyaml, mlflow, qdrant-client, psutil.
  - `dev` group: ruff, mypy, pytest, pre-commit, stubs.
  - `gpu` group: torch, from the explicit `pytorch-cu126` index (resolved 2.14.1+cu126).
  - Both groups install by default. CI can skip `gpu`.
  - Polars/PyArrow, Transformers and FastAPI are added in the stage that first uses them.
  - Stage 2 added `huggingface-hub` 2.4 (pinned downloads, upstream hashes), `polars` 2.1 (lazy reading of records) and `pyarrow` 26 (streaming Parquet writer). pyarrow was previously only transitive.
  - Stage 3 added `lingua-language-detector` 2.2 (LID, ADR-020), `tokenizers` 0.23 (Qwen token
    lengths from the pinned `tokenizer.json`, without the heavy `transformers`), and `numpy`,
    `scipy`, `matplotlib`. Those three were already transitive (via mlflow) and are now
    explicit, because MinHash, sparse Jaccard and the EDA charts import them.
- **Why cu126:** it is verified on driver 566.07 (CUDA 12.7 runtime). Newer CUDA builds need a newer driver. Updating the driver is the owner's call.
- **Alternatives:** torch as a main dependency (forces a ~2.5 GB CUDA wheel on every install); `uv pip install` outside the lock (not reproducible).
- **Revisit:** driver update → consider a newer CUDA index; training libraries that need a specific torch.

### ADR-016 Logging: stdlib + own JSON formatter with redaction — accepted (2026-11-11)
- **Decision:**
  - Logging uses stdlib `logging` with JSON (default) and console formatters.
  - Every record passes a redaction step: sensitive keys become `***`; tokens, e-mails, KZ/RU phone numbers and 12–19-digit runs (cards, IIN) are masked; customer text under `text`/`prompt`/`answer`/... is replaced by `{len, sha256_8}` unless `logging.log_text=true`.
  - Third-party HTTP loggers are kept at WARNING.
- **Alternatives:** structlog (nicer API, extra dependency, same redaction work); loguru (global state, weaker stdlib interop with mlflow/httpx).
- **Revisit:** a log pipeline (Stage 10) needs OpenTelemetry fields.

### ADR-017 Data/model snapshots: tar + sha256 manifest — accepted (2026-11-11)
- **Decision:**
  - One uncompressed tar plus a `manifest.json` per snapshot. The manifest holds per-file sha256 values, the archive sha256 and a content hash. Files are hashed in the same pass that writes the archive.
  - Snapshots are created atomically. Identical content is deduplicated.
  - Restore is verified before files appear and never overwrites existing data.
  - Retention: N newest per name plus pinned snapshots.
- **Why no compression:** parquet and safetensors are already dense, and gzip would make 1 GiB snapshots CPU-bound (measured uncompressed: 3.8 s create, 2.6 s deep verify per GiB).
- **Alternatives:** restic or borg (deduplication and encryption, but an external tool and opaque format); DVC (ADR-004, later); plain copies (no integrity check).
- **Revisit:** snapshots need to go off-machine or be encrypted, or total size exceeds disk budget → restic to an external disk or cloud bucket.

### ADR-018 Licence and usage policy enforced in code — accepted (2026-11-11)
- **Decision:**
  - `LICENCE_MAX_USE` (in `common/schemas.py`) caps `allowed_use` per licence:
    - `train`: CDLA-Sharing, CC-BY, CC-BY-SA, CC0, MIT, Apache;
    - `eval_only`: CC-BY-NC(-SA);
    - `blocked`: `unknown` and `conflicting`.
  - A licence missing from the list fails validation, so new licences are a deliberate choice.
  - `supportpilot.data.policy` sets the minimum level per step: acquire, convert and analysis need ≥ reference_only; evaluation needs ≥ eval_only; training needs train.
  - The step that loads records (`scan_records`) checks every requested source. A rejected source is never downloaded.
- **Why NC is eval-only:** the owner may use the project commercially (portfolio); evaluation use is conservative, training use is not.
- **Alternatives:** a licence note in docs only (not enforced); per-file licence tags (overkill for whole-dataset licences).
- **Revisit:** the owner fixes a commercial or non-commercial status for the project; a source has row-level licences.

### ADR-019 Acquisition and raw-record pipeline — accepted (2026-11-11)
- **Decision:**
  - **Download:** `huggingface_hub.hf_hub_download` into a stable `.partial-{revision}` directory, which gives resume; it uses `HF_TOKEN` for gated repos.
  - **Integrity:** our own check against the upstream LFS sha256 or git blob sha1. An atomic rename makes the revision final, and a finished revision is never modified.
  - **Conversion:** per-source converters (`data/converters.py`) stream rows (csv module, JSONL lines, Parquet batches) into one `RawRecord` contract.
  - **Output:** Parquet via pyarrow `ParquetWriter`, with a content hash. Rejected rows are logged, the reject-ratio gate applies, and the result is swapped in atomically.
- **Verified:**
  - All 5 selected sources downloaded (151.5 MB) and matched upstream hashes.
  - Re-runs skip without network.
  - A real `kill -9` during a 70 MB download resumed, re-verifying and keeping the completed files.
  - A forced re-conversion reproduced identical Parquet bytes and content hashes.
- **Alternatives:**
  - `datasets.load_dataset`: converts to Arrow cache, the original files are not kept as-is, and it adds a heavy dependency.
  - `snapshot_download`: no per-file upstream verification of our own, and it is harder to restrict to verified-and-renamed.
  - DVC `import-url`: extra tooling, see ADR-004.
  - Converting into the canonical schema directly: that would mix lineage with labelling decisions that belong to Stages 3–4.
- **Revisit:** a non-HF source appears (add a `url` fetch type with expected sha256); a source exceeds about 10 GB (shard-level resume, subset include patterns).

### ADR-020 Stage 3 cleaning, language ID and outputs — accepted (2026-11-11)
- **Decision:**
  - **Two checkpoints.** `data/interim/clean/<source>/<rev>/` holds one cleaned row per input
    record (`clean_record/1`, rejected rows included with `reject_reason`). Then
    `data/interim/canonical/` holds the build: `examples/lang=*/` (canonical/1), `pending.parquet`,
    `rejected.parquet`, `groups.parquet`, `pairs.parquet`, `quality_report.json`, `manifest.json`
    (dataset_manifest/1). Every record lands in exactly one of examples / pending / rejected.
  - **Pending, not invented.** Rows that cannot get canonical labels yet (no v0 intent,
    label conflict, suspect label, dialogue, eval-only or reference-only source) are kept
    valid in `pending` with the reason. Labels and actions are never guessed.
  - **Conservative repair.** NFC, control/zero-width/odd spaces, HTML entities, whitespace.
    Typos, case, emoji and profanity are kept (they are the register) and flagged.
  - **PII.** E-mails, phone numbers and Luhn-valid card numbers are replaced by reserved
    synthetic values (`customerNNNN@example.com`, `+1 202 555 01NN`, `4000 0000 0000 0002`).
  - **Placeholders.** Text placeholders get synthetic slot values (deterministic per record).
    The same placeholder in the row's response gets the same value. Other response
    placeholders stay unchanged and are classified (`kb_fact`, `ui_label`,
    `template_instruction`), so no store fact is invented. The currency symbol stays `$`
    because the source responses write `$<amount>`; KZT localisation is part of Stage 4.
  - **Language ID: lingua 2.2** (Apache-2.0, offline models) over en/ru/kk plus 20 other
    languages. A non-supported language wins only at confidence ≥ 0.9 (else the supported
    language that fits the script). A detected language replaces the declared one only at
    confidence ≥ 0.8. Code-switch needs ≥ 2 lower-case words in each script, so Latin brand
    names in Kazakh or Russian text do not count. Belebele calibration: 2,699/2,700
    questions and 2,700/2,700 passages correct.
  - **Generation failures.** Texts with ≥ 3 scripts each ≥ 3% of ≥ 20 letters (token soup)
    are rejected as `garbled_text`. Single-script word salad is not detected (known gap).
  - **Gates** (`data/gates.py`) re-read the written files and re-check them independently:
    accounting, schema (validity ≥ 99.5%), ids, language (LID re-run, ≥ 99%), duplicates,
    placeholders, PII, licence, labels. Splits/leakage are `not_applicable` until Stage 4. A
    failing build is not swapped in; it is kept as `canonical.failed/`.
- **Alternatives:** fastText lid.176 / GlotLID (good, but a fastText wheel on Windows with
  numpy 2 is fragile, plus 126 MB+ model files); langid.py (weaker on short texts); a single
  canonical file with nullable labels (breaks the canonical contract); relabelling while-ai
  by heuristics (would invent labels).
- **Risks:** typo-heavy short EN gets low LID confidence (8–10% of Bitext rows are flagged
  `lid_low_conf`, all reviewed ones are English); Latin-script Kazakh is not detected.
- **Revisit:** RU/KK data arrives in Stage 4. Then measure LID on real KK/RU support text and
  add a GlotLID comparison if KK–RU confusions appear.

### ADR-021 Draft intent map v0 and rule-based actions — provisional (Stage 4 freezes v1)
- **Context:** canonical/1 requires `intent` and `action`. The taxonomy is frozen in Stage 4.
- **Decision:** `configs/intent_map.yaml` maps each source label to one v0 intent or to
  `null` with a reason. It is validated by `IntentMap`.
  - Bitext retail: 38 of 46 labels mapped. Bitext CS: 23 of 27. while-ai: none (payment-agent
    labels for whole dialogues).
  - **Actions.** Most labels use one fixed action. `needs_reference` gives `call_tool` when
    the text contains an order/invoice reference (filled slot or literal `#12345`), else
    `clarify` (F1, F7). Escalation is used only for explicit human requests and complaints (F8).
  - **Label noise.** `suspect_unless` sends rows without the label's expected wording to
    pending. It is used once: retail `return_product_in_store`, where 265 of 994 rows are
    product-information requests (237 after dedup).
  - Every canonical row carries the flag `labels:intent_map_draft_v0`.
- **Alternatives:** leave canonical empty until Stage 4 (fails the Stage 3 DoD and delays
  the gates); infer intents with a classifier (invents labels).
- **Revisit:** Stage 4 freezes v1. The open items are customer_service vs human_agent,
  technical support, cart, `pay`, GDPR requests, shipping-address changes and the
  return/tracking conflicts listed in the EDA report.

### ADR-022 Deduplication, paraphrase families and scenarios — accepted (2026-11-11)
- **Decision:**
  1. **Exact duplicates.** Two rows are duplicates if they share the normalised template key
     (NFKC, casefold, no punctuation, slots as `<slot>`) or, for single messages, the
     normalised final text. Dialogues compare all turns, and policy decisions compare text
     plus facts and policies. In a label-compatible group the most permissive source is kept,
     then registry order, then row order. The others are rejected with `duplicate_of`.
     Incompatible labels make the group `label_conflict`. Texts that differ only in case,
     punctuation or inserted profanity (equal similarity key) with incompatible labels are
     conflicts too.
  2. **Near-duplicates.** Character 5-grams of the similarity key (profanity removed) →
     MinHash (128 perms, seeded PCG64, 31-bit universal hashing) → LSH (32 bands × 4 rows).
     Every candidate is verified by exact Jaccard (sparse intersection, chunked for RAM).
     LSH recall against brute force: 95% at J ≥ 0.5, 100% at J ≥ 0.6, 0.7 and 0.8.
  3. **Family** = connected components of pairs with J ≥ 0.8 **and** compatible labels, plus
     source alignments (Belebele translations, support-json-ru `scenario_group_id`). Labels
     are compatible within a source if the source label and action are equal, and across
     sources if the mapped v0 intent and action are equal.
  4. **Scenario** = families with the same scenario key, which is the source label plus the
     required action, or a source scenario id. Exact duplicates across sources join their
     keys. Lexical similarity alone never merges different labels.
  5. **Ids.** `fam_`/`scn_` + sha256 of the smallest member example_id or key. This keeps ids
     stable when new members join.
- **Calibration (real data):**
  - At J ≥ 0.6–0.8, 7–12% of same-source near-duplicate pairs carry different labels. They
    are minimal edits that change meaning ("modify" vs "close my account", "in store" vs
    "online"), so the label guard is required.
  - Without it, at J 0.5 / 0.6 / 0.7, 93% / 68% / 19% of Bitext texts chain into families of
    ≥ 100 (one per intent). At 0.8 the largest family has 109 texts and 0.3% are in families
    of ≥ 100, so 0.8 was chosen.
- **Consequence for Stage 4:** Bitext has 68 scenarios, and the canonical set has 55.
  - A scenario-disjoint split cannot keep most intents in both train and test.
  - A family-disjoint split only removes near-copies (J ≥ 0.8); test items stay paraphrases
    of training items (J 0.5–0.8).
  - So in-distribution validation/test from Bitext measures paraphrase robustness, and
    generalisation must be measured on the independent gold set.
  - DATA_CONTRACTS still says "splits by scenario_id". Stage 4 must decide the split unit
    with this evidence. This is an open item, and the contract is not changed silently.
- **Alternatives:** `datasketch` (pure Python, slower, same algorithm); all-pairs inside
  label blocks (misses cross-source duplicates and does not scale to 1M); embedding
  clustering (multilingual-e5 or BGE-M3: needs a model download and GPU, and inside a
  homogeneous paraphrase set its clusters are arbitrary). It is the candidate for Stage 4
  cross-lingual or semantic dedup, once translations exist.
- **Revisit:** Stage 4 decides the split unit, or generated or translated data needs semantic dedup.

---

## Open decisions (owner input needed; see PROGRESS.md)
1. Budget for a paid LLM API (MT for KK/RU, data generation, LLM judge). Without it: open MT + local models, with lower expected KK quality.
2. Availability of fluent KK and RU reviewers for the gold benchmark and translation audits.

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
- Qdrant troubleshooting (Windows bind mounts → named volumes): https://qdrant.tech/documentation/common-errors/
- Qdrant 1.20 release: https://qdrant.tech/blog/qdrant-1.20.x/
- PyTorch CUDA wheel indexes: https://download.pytorch.org/whl/cu126
- Dataset cards reviewed in Stage 2: URLs and revisions in `configs/sources.yaml`
- Stanford Alpaca data licence (CC BY-NC 4.0): https://github.com/tatsu-lab/stanford_alpaca
- FLORES+ (evaluation only): https://huggingface.co/datasets/openlanguagedata/flores_plus
