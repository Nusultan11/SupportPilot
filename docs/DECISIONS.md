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

### ADR-006 Training-data backbone and language adaptation — accepted, refined by ADR-024/025 (2026-11-12)
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

### ADR-012 Skills and MCP: none at Stage 0; MCP dev tools on demand — accepted (2026-11-11, amended 2026-11-12)
- **Decision:** no project skills or MCP servers yet. There is no repeated process to capture.
  HF access via `huggingface_hub`; research via built-in web tools.
  Candidate skills once the processes exist: `data-quality-audit` (Stage 3), `rag-eval` (Stage 6), `experiment-review` (Stage 7), `release-readiness` (Stage 10).
- **Stage 3 check (2026-11-11):** the data-quality audit ran once, and it is now code
  (`data clean`, `data gates`, `data eda`), not a manual process. So no skill yet. Revisit
  when Stage 4 repeats the review on translated data.
- **Stage 4 check (2026-11-12):** the MT review and the gold authoring ran once each, and
  the repeatable parts are code (`data mt-compare`, `data sft`, `data gold`). No skill yet.
- **Stage 5 check (2026-11-12):** the KB build, its gates and the update cycle are code
  (`kb render|build|validate|activate`); the parser comparison ran once. No skill yet.
- **Revisit:** at the end of each of those stages, if the process was run ≥ 2 times by hand.
- **MCP amendment (2026-11-12, owner request before Stage 6):** two project MCP servers in
  `.mcp.json`, both verified to start and answer (RUNBOOK §4a). Skills: still none.
  - `huggingface`: the hosted `https://huggingface.co/mcp`, used anonymously (no token in
    the repo). It gives `hub_repo_search`, `hub_repo_details` (card metadata, licence) and
    `hf_fs`. Need: Stages 6–7 compare embedders, rerankers and base LLMs, and every candidate
    needs a licence check for the source registry (§2). Today this needs ad-hoc web reads.
    Data sent: search terms and repo ids only, never project data.
  - `mlflow`: `mlflow mcp run` from the locked env (`mlflow[mcp]` in the dev group), local
    stdio, `MLFLOW_MCP_TOOLS=experiments,runs`, tracking URI `sqlite:///mlruns/mlflow.db`,
    telemetry off. Need: Stages 6–7 compare runs per language (EVALUATION protocol). The tools
    read the same store that `common/tracking.py` writes. The server also exposes write tools
    (`create_*`, `delete_*`, `restore_*`, `rename_experiment`, `update_experiment`,
    `link_traces_to_run`). Agents must use only the read tools (`search_experiments`,
    `get_experiment`, `list_runs`, `describe_run`). Runs are logged by code, never through MCP.
  - **Rejected:** `mcp-server-qdrant`, because it embeds with FastEmbed and writes its own
    collections, so its queries would not match our e5 prefixes. Inspect the KB with
    `kb status` or the Qdrant REST API instead. GitHub MCP, because `gh` covers it.
    Context7/docs servers, because the built-in web tools suffice and there is no measured gap.
  - **Deferred:** Kaggle's hosted MCP (`https://www.kaggle.com/mcp`, OAuth). It needs the
    owner's Kaggle account, so it waits for Stage 7 when training moves to Kaggle.
  - **Rollback:** delete the entry from `.mcp.json` (and `mlflow[mcp]` from the dev group).
- **Dev-tool amendment (2026-11-12, owner request before Stage 6): five more servers, all on demand.**
  - **New evidence:** the owner asked for Context7, Serena, GitHub, Playwright and Exa as
    standard agent tools. This reopens the earlier rejection of GitHub MCP and Context7
    (above). All five are dev tools: nothing in `src/`, configs or dependencies uses them.
  - **Client facts (Claude Code 2.1.321 in the desktop app, measured):** tool schemas load
    lazily (tool search on by default), so idle servers cost only their tool names. Every
    enabled project server is connected at session start; the desktop app approves
    `.mcp.json` servers without a prompt. A stdio server is a local process for the whole
    session. The `/mcp` toggle (`disabledMcpServers`, per project in `~/.claude.json`)
    stops a server from starting; per the docs, re-enabling connects it in the running
    session. The agent cannot toggle project servers itself (`set_session_connector_enabled`
    covers claude.ai connectors only).
  - **Decision:**
    - Remote HTTP servers stay on: `context7`, `exa` (tools `web_search_exa`,
      `web_fetch_exa`), `github`, `huggingface`. Cost per session: one connect and tool
      list each, no local process, no project data.
    - Local stdio servers are off by default: `serena`, `playwright` and now also
      `mlflow` (it ran in every session). The owner turns one on in `/mcp` for the task that
      needs it. A committed `disabledMcpjsonServers` was rejected because it hard-rejects a
      server and `/mcp` cannot re-enable it.
    - `github` is read-only: `X-MCP-Readonly: true` and toolsets `context,repos,issues,
      pull_requests,actions` (29 read tools). Token from the `GITHUB_MCP_PAT` user env var
      (fine-grained, read-only). Writes go through `gh` when a task asks for them.
      The local Docker image was rejected: it is a local process and needs Docker running.
    - `serena` is pinned (`serena-agent==2.1.0`) with context `claude-code`, the tool
      interface (2.1 defaults to a Python REPL tool), no dashboard, no telemetry, and modes
      `no-onboarding` and `no-memories`, because PROGRESS/DECISIONS stay the only memory.
      `.serena/project.yml` is versioned; everything else under `.serena/` is ignored.
    - `playwright` is pinned (`@playwright/mcp@0.0.84`, the newest release ≥ 10 days old),
      headless, isolated profile, installed Edge (no browser download). Output goes to the
      git-ignored `artifacts/playwright`. It stays off until the Stage 8 UI exists.
    - Context7 and Exa run keyless (anonymous rate limits). Keys, if ever needed, go into
      local scope, because an empty optional `Bearer` header breaks the connection.
    - `.claude/settings.json` denies the MLflow write tools and Playwright
      `browser_run_code_unsafe` (closes the earlier open item).
  - **Alternatives:** everything always on (a Serena + Pyright and a Node process in every
    session, for tools most tasks do not need); stdio variants of Context7/Exa (local
    processes for no gain); a launcher script that edits client state (couples the repo
    to a private client file format).
  - **Risks:** a fresh clone starts the local servers until the owner disables them once
    (RUNBOOK §4a step 1); anonymous rate limits; queries leave the machine (rule: public,
    minimal queries only, AGENTS §12).
  - **Verified:** RUNBOOK §4a (client health check plus one safe call per server, no
    leftover processes).
  - **Rollback:** remove the entry from `.mcp.json` (and `.serena/`, the deny rules), or
    disable it in `/mcp`.
  - **Revisit:** if the client gains agent-driven activation of project servers, use it for
    the local servers; if a remote server's handshake cost or rate limits become a problem,
    turn it off by default as well.

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
  - Stage 4 added the `ml` group (installed by default): `transformers` 5.22, `sentencepiece`,
    `sacremoses`, `accelerate` and `bitsandbytes` 0.50 (4-bit MADLAD on the 4 GB GPU, ADR-024).
    Building the SFT dataset from an existing MT cache does not import them (lazy imports).
  - Stage 5 added `pdfplumber` 0.11 (MIT, PDF text/tables), `python-docx` 1.2 (MIT),
    `openpyxl` 3.1 (MIT) and `reportlab` 5.1 (BSD, PDF writer): chosen by the ADR-027
    measurement. PyMuPDF (AGPL-3.0) was measured and not adopted.
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

### ADR-021 Draft intent map v0 and rule-based actions — superseded for Stage 4 by ADR-023 (still used by the Stage 3 build)
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

### ADR-023 Intent taxonomy v1 — accepted (2026-11-12)
- **Context:** the Stage 3 draft v0 left 11 Bitext labels unmapped and several taxonomy
  questions open (ADR-021). Stage 4 must freeze the taxonomy before building SFT targets.
- **Decision:** `configs/taxonomy.yaml` is the single source of taxonomy v1: 35 intents in 10
  groups, 5 actions, 4 mock tools, 10 escalation reasons. Each intent records its default
  action, tool, KB documents and its clarify / escalate / refuse conditions.
  - New intents: `change_address` (always clarify: profile address vs placed order),
    `product_defect` (warranty; needs_reference; escalate on injury or fire), `contact_info`
    (how to reach support; answer_kb, not a request for a person), `personal_data` (escalate,
    privacy_request), `newsletter`, `cart_help`, `app_help`, `technical_issue` (escalate).
  - Ambiguous labels are split by ordered regex `rules` on explicit wording; a text that matches
    no rule stays unlabelled. Bitext `customer_service`: "how/call/hours" -> contact_info,
    "talk/speak" -> human_agent, the rest unlabelled. `pay`: "transfer" excluded, "pay" ->
    payment_methods. A paraphrase family split across two v1 labels is dropped.
  - Excluded: `check_cancellation_fee` (telecom concept; labelling it out_of_scope would misread
    the customer) and all while-ai labels (dialogue-level payment-agent labels, 52% hard defects
    in the Stage 3 review). No while-ai dialogue enters Stage 4.
  - `schemas.INTENTS_V0` stays for the Stage 3 build and gate, so Stage 3 outputs are unchanged;
    `schemas.INTENTS` (v1) is a superset and a unit test keeps it equal to the YAML.
- **Alternatives:** editing `configs/intent_map.yaml` in place (changes the Stage 3 build hash
  and silently rebuilds it); a classifier to relabel ambiguous rows (invents labels).
- **Revisit:** gold or validation errors concentrate on one intent pair (merge or split), or a
  new tool (e.g. address change) makes a clarify intent actionable.

### ADR-024 RU/KK creation: MADLAD-400 + glossary post-edit + QE filter — accepted (2026-11-12)
- **Context:** no paid API budget (open decision 1) and no fluent reviewer (open decision 2).
  Candidates had to be open, licence-compatible for training data and fit a 4 GB GPU.
- **Comparison.** 100-item stratified EN sample; the first 50 were judged per system by the LLM
  agent (`docs/reviews/stage4_mt_review.yaml`). Major-error rate before QE:

  | system | licence | RU major errors | KK major errors |
  |---|---|---|---|
  | Opus-MT en-ru / en-trk | Apache-2.0 | 52% | 92% |
  | M2M100-418M | MIT | 50% | 100% |
  | MADLAD-400-3B (nf4) | Apache-2.0 | **40%** | **34%** |
  | MADLAD en->ru->kk pivot | Apache-2.0 | n/a | 52% |

  Not tried: NLLB-200 (CC-BY-NC, outputs not usable for training under ADR-018), Tilmash (gated,
  no HF account), paid APIs (no budget).
- **Decision:**
  - MADLAD-400-3B-MT in 4-bit (bitsandbytes nf4, 2 beams, batch 8, output capped at 2.5x the
    input) for both languages. Opus-MT ru-en and MADLAD kk-en back-translate for QE.
  - Bitext texts are localised before MT: USD amounts -> KZT at a fixed synthetic rate. Names
    and cities are Kazakhstan values already (Stage 3 catalogue) and must survive translation.
  - Domain glossary from the reviewed errors. Repairable mistranslations are fixed by suffix
    tables (kk "тапсырма" -> "тапсырыс", ru "приказ" -> "заказ", "тележка" -> "корзина"); the
    others reject the row (cart as "fence" / "trash bin", "human agent" calques, "article").
  - QE filter, all must hold: language ID = target, Cyrillic script, at most 34% Latin words,
    numbers and names preserved, length ratio 0.45–2.6, no decoder loop, no Russian function
    words in Kazakh, back-translation chrF >= 0.30, multilingual-e5 cosine >= 0.80.
  - Every MT row keeps `translation_of`, `translation_system`, the raw MT, the back-translation
    and QE values; rejected rows stay in `data/interim/stage4/translated/` with their reasons.
  - The cache (`data/interim/stage4/mt/`) makes translation resumable; re-runs translate nothing.
  - GPU time bounds the volume: 100 Bitext rows per source label and action per language
    (7,554 per language) at about 2.5 sentences/s on the RTX 3050.
- **Risks:** an LLM judged the errors; Kazakh fluency is not verified by a human. MT of typo-heavy
  templated text stays unnatural even when accepted. Mitigation: native authored data
  (ADR-025), reported QE and post-QE error rates, and a human audit as an open item.
- **Revisit:** a paid MT/LLM budget or a fluent reviewer; KazParC / Tilmash access.

### ADR-025 SFT construction: native authored data, training KB variant, templated grounded targets — accepted (2026-11-12)
- **Context:** Bitext has 55 scenarios and its responses are unfit as targets (Stage 3). MT alone
  gives RU/KK copies of the same templates. There is no budget for a generator LLM.
- **Decision:**
  - **Native authored scenarios** (`authored/scenarios/`, 81 situations, 896 utterances,
    en / ru / kk, RU–KK code-switching, multi-turn) written by the Stage 4 agent and labelled
    `origin=llm_generated`, `source_id=supportpilot_authored`. They add what Bitext lacks:
    late or "delivered but not received" orders, refused cancellations, double charges, safety
    risks, fraud, legal threats, repeated failure, injection attempts, out-of-scope requests.
  - **Training KB variant** (`authored/kb_train/`): 6 fictional stores x 8 documents x 3
    languages, plus outdated versions: 1,734 chunks with contract chunk ids. Every policy value
    differs between variants and from the demo store (`kb_sources/demo_store/facts.yaml`,
    `kb_variant_distinct` check).
  - **Targets** are `AssistantResponse` JSON rendered from authored answer templates
    (`authored/responses/templates.yaml`) with the values of the cited chunk or the tool result.
    Russian plural agreement and Kazakh case suffixes are table-driven (`data/textfmt.py`).
  - **Tasks.** answer_kb utterances draw grounded 62% / abstain 12% / conflicting 14% /
    injection 12%. call_tool -> tool_answer with synthetic `get_order`, `get_payment`,
    `get_stock`, `request_cancellation` results (7% order not found). clarify, escalate and
    refuse_oos follow the action. Every example passes `SFTExample` and
    `common.grounding.check_response`, or it is dropped and counted.
- **Alternatives:** Bitext reference responses (prompt leaks, invented facts); free-form LLM
  generation (no budget, unverifiable); translating authored EN (loses native phrasing).
- **Risks:** templated answers are less varied than real agents; one author for the scenarios and
  the gold set limits independence (ADR-026). Revisit with a generator budget or human writers.

### ADR-026 Split unit and held-out sets — accepted (2026-11-12)
- **Context:** DATA_CONTRACTS splits by `scenario_id`. After v1 relabelling Bitext still has one
  scenario per label and action, so a scenario-disjoint Bitext split removes whole intents from
  test, and a family split leaves paraphrases (J 0.5–0.8) on both sides (ADR-022).
- **Decision:** keep the contract (split by `scenario_id`) and change what feeds held-out sets.
  - Every Bitext-derived example (EN originals and their RU/KK translations, which share family
    and scenario) goes to **train**.
  - **Validation / test** = authored scenarios, assigned by scenario_id 70/15/15, stratified by
    action, with at least one scenario per action in each held-out set.
  - Held-out user texts that are exact, lexical (char-5-gram J >= 0.8), 13-gram or semantic
    (e5 cosine >= 0.95, digits masked) near-copies of a train text are removed; the gate then
    re-checks scenario/family, exact and lexical overlap = 0.
  - The **gold** benchmark is authored separately (312 scenarios x 3 languages) and checked
    against all SFT splits. Its review status is `draft`: there is no human reviewer.
  - Volume is what quality allows: 90k/5k/5k is not reached and nothing is duplicated to reach
    it. English is down-sampled toward EN 30 / RU 35 / KK 35; RU and KK are never up-sampled.
- **Consequence:** validation and test measure generalisation to new authored situations, not
  Bitext paraphrase robustness, and they are small.
- **Revisit:** more independent scenarios (human-written, or generated with a budget) allow larger
  held-out sets; Stage 7 shows that validation does not track gold.

### ADR-027 Demo-store KB: rendered sources, measured parsers, verified extraction, versioned builds — accepted (2026-11-12)
- **Context:** Stage 5 needs a demo-store KB in RU/KK/EN across PDF/DOCX/XLSX/CSV/JSON/TXT
  whose extraction is provably complete, plus versioned, incremental updates for Stage 6.
  There is no real store, so the documents are synthetic and generated.
- **Decision:**
  - **Sources.** Canonical values live once: `kb_sources/demo_store/facts.yaml` (policy values,
    shared with Stage 4) and `catalog.yaml` (30 products, 12 delivery zones). Each document is a
    spec `docs/<doc_id>.yaml` (kb_doc/1) with native text per language and `{fact:format}`
    placeholders (Stage 4 `textfmt` formats + `seconds`, `minutes`, `date_full`). Every
    language states the same value because no value is typed twice. 10 documents x 3 languages:
    2 PDF, 3 DOCX, 2 TXT, 1 JSON (FAQ export), 1 XLSX (catalogue), 1 CSV (delivery tariffs).
  - **Rendering** is byte-deterministic: reportlab `invariant=1` with DejaVu Sans from
    matplotlib (Kazakh Cyrillic), python-docx / openpyxl with zip entry times and core dates
    rewritten. Re-rendering unchanged specs gives identical sha256, so a document's `version`
    changes only when its content does. Rendering also writes the ground truth
    (`expected.json`: every fact placement with its exact surface string, every content string)
    and a CMS-like `register.json` (doc metadata, section ids and headings).
  - **Parsers** read only the file: PDF font size/weight, DOCX styles, XLSX/CSV header rows,
    TXT setext/Q:A:/pipe tables, JSON keys. Section ids are attached by matching found headings
    to the register; unmatched headings stay `None` and fail a gate. Scanned PDF pages raise an
    error (no OCR engine is configured, and none is needed for born-digital PDFs).
  - **Parser choice (measured)** on the 6 rendered PDFs plus 3 multi-page stress PDFs with a
    60-row table split across pages, and on the 9 DOCX / 3 XLSX files:

    | candidate | licence | word recall | fact recall | table rows exact | font info | time |
    |---|---|---|---|---|---|---|
    | **pdfplumber 0.11.10** | MIT | 1.000 | 1.000 | **1.000** | yes | 4.2 s |
    | PyMuPDF 2.1 | AGPL-3.0 | 1.000 | 1.000 | 0.389 | yes | 17.3 s |
    | pypdf 7.0.1 | BSD | 1.000 | 1.000 | 0 (no tables) | no | 1.3 s |
    | **python-docx 1.2.0** | MIT | 1.000 | 1.000 | 1.000 | styles | 0.8 s |
    | docx2python 3.7.2 | MIT | 1.000 | 1.000 | 1.000 | no styles | 0.5 s |
    | **openpyxl 3.1.5** | MIT | records 1.000 | ints 90/90 as int | | | 0.45 s |
    | python-calamine 0.9.1 | MIT | records 1.000 | ints 0/90 (floats) | | | 0.04 s |

    Times include rendering the same files for every candidate. Docling was not measured: it
    pulls torch layout models (GBs, GPU busy with Stage 4 MT) for a problem the born-digital
    files do not have; revisit for scanned or complex-layout real documents.
  - **Chunking by type:** policy/terms = one chunk per section with the heading path as the
    first line and tables as pipe tables (header kept); FAQ = one Q+A; records = one row with
    `header: value` lines, sheet and key; intro text = one chunk. Over the budget (400 Qwen3
    tokens) a unit splits at paragraphs, then sentences; tables split by rows with the header
    repeated. `translation_group = <doc_id>#<section_id>` links languages.
  - **Gates (13), all must pass before a version is activated:** ids, sources, metadata,
    sections_linked, facts (100% of placements in the chunk of their section; integers parsed
    back), coverage (every canonical fact placed in every language), locators (the source text
    at the chunk locator contains the fact, re-read from the file), completeness (no lost and no
    extra words), cross_lingual (same numbers and fact keys per translation group), language
    (lingua LID), tokens, pii, evidence (gold `doc.section` references resolve in en/ru/kk).
  - **Versions:** `data/kb/<kb>/versions/<kb_version>/` is immutable and self-contained (its own
    copy of the sources, documents, chunks, changes, evidence map, quality report, manifest).
    Unchanged documents (same sha256 and processing params) are reused, not re-parsed; removed
    documents drop out; `changes.json` lists the chunk ids to delete/upsert for Stage 6.
    Identical content and ground truth produce no new version. A version that fails its gates
    is kept as `.failed` and never activated; `kb activate` rolls back.
- **Alternatives:** hand-written binary documents in git (no ground truth, values drift
  between languages); one parser library for everything (PyMuPDF: AGPL and weaker tables);
  fixed-size token windows (break Q+A pairs and table headers); a vector DB as the version
  store (Stage 6 indexes from these versions instead, ADR-007).
- **Risks:** generated documents are cleaner than real ones (no scans, merged cells,
  multi-column layouts); parsers are proven on this generator's layouts only. KK/RU texts are
  LLM-authored, not reviewed by a fluent human. Mitigation: parsers depend on layout signals
  (font size, styles), not on the generator; `ExtractionError` instead of silent loss.
- **Result (2026-11-12):** 30 documents, 288 chunks (catalog 90, policy 126, faq 45, terms 27);
  912/912 fact placements found and locator-verified (per language: money 78, periods and
  numbers 46, identifiers 47, dates and opening hours 4, condition lists 3, text 126; 304 per
  language); 221 canonical facts covered in every language; word recall 1.0 and no extra words in all 30 documents; LID 198/198 prose
  chunks; max chunk 280 Qwen3 tokens (p50 81). The data bug the cross-lingual gate found in
  Stage 4's `facts.yaml` (an unquoted comma dropped the house number from the RU/KK showroom
  address) is fixed and the loader now rejects such values.
- **Revisit:** real store documents (scans, complex layouts) -> OCR / Docling evaluation;
  a KB larger than memory -> streaming build.

---

### ADR-028 Native-language external data — accepted (2026-11-12)

- **Need:** the existing SFT was dominated by Bitext paraphrases/MT; native KK and OOS
  examples were scarce. Reuse two pinned CC-BY-4.0 sources already acquired through the
  registry: the 406-question public mangi KK support sample and MASSIVE EN/RU. The commercial
  full mangi corpus is not used. Attribution, revisions and licence evidence are in
  `configs/sources.yaml`; questions alone are used, never upstream reference answers.
- **Choice:** a checkpointed `data native` step, alongside the English MT backbone. It
  maps only reliable OOS/greeting labels using the existing taxonomy rules, keeps one row
  per family, caps per label/language and preserves provenance. Banking/telecom questions
  must contain the needed domain cue in their text; source-domain metadata alone cannot
  justify a refusal. Generic payment, phone-feature and router questions are excluded.
  Native means no pipeline MT; it does not assert human authorship of mangi questions.
- **Alternatives:** MT expansion would repeat the same few scenarios and retain the
  documented KK semantic-error risk; including whole banking/telecom labels produced false
  OOS labels for ambiguous questions. Rejected both; original synthetic situations cover
  the store-specific behavioural gaps.
- **Cache correctness:** source registry content joins canonical version, taxonomy hash,
  sampling caps and seed in the input identity. Changed language/usage policies cannot
  silently reuse a previous selection.

### ADR-029 Independent evaluation coverage and retrieval data — accepted (2026-11-12)

- **Need/evidence:** action-only splitting covered just 14/12 intents in validation/test.
  The first expanded build still lost 10 language/intent cells when the semantic filter
  deleted all corresponding test questions. A new external greeting also matched one gold
  question. Demo KB coverage and answer lexical diversity were limited.
- **Choice:** expand to 300 authored situations; stratify trilingual situations by intent
  with actions interleaved. Keep scenarios/families disjoint. Preserve the assigned held-out
  questions and remove training copies at the existing exact/lexical/13-gram/semantic
  thresholds (amends ADR-026). Gold questions are a decontamination exclusion list only:
  reject copied SFT inputs without changing gold questions or expected answers. Check final
  coverage after filtering, including train and every language. The second city in a move
  scenario must differ from the destination. Answer-variant RNG is separate from evidence
  and tool-state RNG, so wording choices cannot silently change the facts.
- **KB:** four additional trilingual documents (pickup/delivery FAQ, sizes, repair service,
  bonus/gift-card FAQ) and six deliberately similar catalogue products; preserve existing
  demo facts and gold abstention assumptions. Training policy values remain a different
  fictional-store variant. Existing deterministic render/parse/version gates remain in use.
- **Retrieval:** original question triples with explicitly authored graded section
  judgements resolve to the active KB's real chunk IDs. Include table lookups, similar
  products, multi-section needs, ambiguous scopes, unanswerable live-tool/unlisted facts,
  cross-language and code-switch queries. Build a manifest and standard qrels without
  training an encoder, creating an index or running a RAG system.
- **Alternatives:** keyword-based automatic qrels would make a BM25 benchmark circular;
  gold paraphrases would compromise independence; more slot fillings would inflate volume
  without increasing scenario coverage. Rejected. For retrieval semantic similarity is a
  diagnostic flag, consistent with gold: independently authored requests can share intent.
  Exact/lexical/13-gram copies block publication; no benchmark questions are silently removed.
- **Limits:** a single LLM authored/re-read content. RU/KK fluent human review and independent
  relevance assessment remain open. Existing MT quality estimates cannot be interpreted
  as measurements of the new authored questions. No authored-data redistribution licence
  or real customer data was introduced.

### ADR-030 Measured multilingual retrieval and portable index — accepted (2026-11-12)

- **Problem:** Stage 7 needs evidence with stable KB provenance on 4 GB VRAM, including
  Kazakh, cross-language queries and policy changes. A high English-only average is
  insufficient. The fixed 681-query/369-chunk benchmark is LLM-authored and remains draft.
- **Choice:** pinned BGE-M3 dense embeddings (CLS, 1024 dimensions, fp16 CUDA), Unicode
  word BM25-style sparse vectors with Qdrant IDF, min-max weighted fusion (.75 dense),
  20 candidates from each channel, BGE-reranker-v2-m3 over 20 canonical candidates,
  output up to 10, whole-chunk JSON evidence within 1024 Qwen tokens. No section
  diversification or automatic score gate by default. Reuse Transformers/Qdrant/MLflow;
  BM25S is an experimental reference dependency, not an additional runtime service.
  Model cards, licences, full revisions/protocols: `configs/retrieval_models.yaml`;
  measured comparison/CI artifacts: `artifacts/reports/retrieval/` (backed up locally).
- **Evidence:** 96 configurations/variants, four encoders and two rerankers. Dense-only
  mono Recall@5 RU/KK/EN: E5-small .9259/.6411/.9330; E5-base .9530/.7459/.9565;
  BGE-M3 .9775/.9463/.9693; Qwen3-Embedding-0.6B .9775/.8834/.9770. Cross-language
  Recall respectively .7283/.7536/.9275/.7029. Architecture reputation did not decide.
  Final reranked Recall .9898/.9678/.9760, cross .9420; draft qrels and bootstrap limits
  apply. Against the development fusion baseline, the reranker improves nDCG by .0339
  [95% paired CI .0209,.0467]; recall delta .0105 [.0010,.0212]. These are development
  comparisons, not an independent deployment acceptance test.
  The matched k20 control gives nDCG delta .0335 [.0209,.0466]. K20 versus k10
  reranking is not statistically established (.0012 [-.0019,.0053]); k20 follows
  the development minimum-language recall rule. K10 remains a latency alternative
  to revisit with Stage 7 residency/latency measurements, not a proven inferior option.
- **Alternatives/negative results:** word sparse loses morphology and cross-language
  meaning; prefix sparse helps mono retrieval but not translation. RRF, dense weights
  .25/.5/.75, top-k 10/20/40 and E5-base canonical/window/overlap/adjacent views were
  measured. Qwen reranking is substantially slower, and its k20 cross recall is .8551
  versus .9420 for BGE. Qwen batch 8 did not materially improve throughput over 4;
  batch 16 was slower. No universal optimum across all chunking/encoder combinations
  is claimed. Failed setup, interrupted inference and negative threshold runs are retained.
- **Answerability limit:** a positive reranker score is not evidence of answerability.
  Per-chunk pruning drops legitimate background. A whole-query .02 gate initially
  met mono targets but reduced development cross recall .9744 -> .6667. The selection
  rule was amended to protect cross-language/RU–KK recall within .01 of ungated output;
  all positive tested thresholds fail. Keep the gate as an optional measured mechanism,
  disabled by default. Consequently default top-k returns candidates for all 48 empty-qrels
  queries (FPR 1.0). Empty filters, expired policies, insufficient context and configured
  low-confidence rejection fail closed, but semantic abstention is **not solved**.
  No factual answer may be inferred from a nonempty candidate list alone. Downstream
  claim grounding and live-tool routing are required before customer-facing use.
- **Evaluation honesty:** after the initial gate experiment, holdout was already viewed.
  Final holdout/CIs are exploratory; no untouched confirmation claim. Gold never selected
  parameters. Keep all statuses draft and retain `gate_selection_history.json`.
- **Index lifecycle:** immutable hashed local snapshots + checksummed batch caches;
  one materialised collection per recipe, OS-owned writer locks, bounded IO retries,
  inventory-verified upsert/delete, fail-closed readiness. Parent `changes.json` is
  validated; non-parent rollback uses the exact inventory diff. Interruption resumes
  from committed batches. Metadata-only date/type edits replace payloads while retaining
  vectors; complete payload inventories are compared on no-work and completion checks.
  Unchanged runs perform no inference/upsert/delete. Brief
  unavailability during version changes is an explicit trade-off against dual collections.
  Deleted IDs leave the live index; internal tombstones are Qdrant's responsibility.
- **Restore correction:** global sparse IDF changed after delete/reinsert/restore despite
  byte-identical named vectors. Scope IDF to live `kb_version` using Qdrant 1.20's corpus
  filter. Recompute all prepared searches and comparisons with `kb_scoped_idf/1`.
  Snapshot recovery actually uploads the local file into a separate collection; compare
  all payloads, all named-vector hashes, and six dense/sparse query results (normal
  floating-point tolerance). Parent rollback, forward 90 deletes/171 upserts/198 retained
  vectors and no-work rerun were verified against the real server.
- **Resources/operation:** selected encoder and reranker coexist locally; no LLM residency
  or multi-user SLO has been verified. Full online measurements, process RSS, PyTorch
  peaks, index bytes and indexing times are in `performance.json`/`index_drill.json`.
  Exact search is appropriate to this 369-chunk experiment; do not infer million-row scale.
- **Revisit/rollback:** require a fresh independently reviewed multilingual/unanswerable
  cohort, evidence of unacceptable recall/latency, larger documents or Stage 7 residency
  measurements before changing this choice. Activate a previous KB, rebuild/synchronise
  that recipe from verified artifacts/cache, then serve; never mix KB versions. Original
  KB/SFT/gold contracts are unchanged. Model-specific thresholds require recalibration.

### ADR-031 Scale evidence protocol and experiment isolation — accepted (2026-11-13)

- **Problem:** Stage 6's 369 chunks and 175 needs cannot establish scale or semantic
  answerability. Its inspected holdout cannot become a fresh confirmation set. The
  Stage 6.1 request also requires jointly sufficient multi-document evidence and a
  separate 100k load corpus; translations and slot repetitions cannot satisfy diversity.
- **Choice:** explicit authored need/family keys, canonical fact support groups and
  a preregistered family split. Separate relevance rankings from AND-of-OR sufficiency
  and packed-context coverage. Freeze atomically only after volume, integrity, leakage
  and a source-bound substantive-diversity review pass. Preserve draft/LLM review labels;
  human review is not fabricated. Record scenario bootstrap plus family sensitivity.
  Development selects configurations; holdout is inaccessible to selection and is only
  evaluated after a checksummed selection. Do not lower gates to publish a small pilot.
- **Alternatives:** pooling translated variants as independent observations, keyword
  qrels, random row splitting or a score-only answerability label would introduce known
  measurement bias. Reusing the old exploratory holdout would not resolve it. Rejected.
- **Isolation:** optional validated retrieval namespace separates the Qdrant collection
  and local state. Empty namespace retains original paths/names and recipe identities.
  New comparison pair caches use a separate data root. Baseline reports/index snapshots
  are not rewritten. A staged KB can be audited by explicit version without activation.
- **Load control:** independent normalized random vectors, bounded generation through
  a memmap, checksummed chunk/vector sources, exact search as the ANN reference, HNSW
  ef32/64/128, on-disk vectors/index and explicit upload/build/recovery phases. This uses
  the existing Qdrant 1.20/NumPy/MLflow stack. No new service or framework is justified.
  ANN agreement is not semantic retrieval quality; precomputed upload is not encoding.
- **Evidence/negative result:** the original 100k random-vector run indexed every vector,
  verified add/change/delete and a full snapshot restore. ANN Recall@10 ef32/64/128 was
  .163/.294/.466, despite lower latency than exact. This workload does not justify an
  approximate-search default for the real multilingual system. Keep exact unchanged.
  Process RSS excludes Docker; a later Docker memory sample is not its peak or the
  isolated collection's memory. Detailed measurements/state belong to PROGRESS.md.
- **Reproducibility correction:** load generator identity includes its text template,
  not just dimensions/seed. Rebuilding the 100k data with that explicit identity produces
  byte-identical vectors/chunks to the measured run; keep historical run/source identities
  unchanged. Final tracking receipts are written after resource-context finalization.
  Earlier operational receipts remain historical; corrected final counters are separate
  delivery evidence, not silently edited experiments.
- **Revisit/rollback:** require a full substantively reviewed scale cohort and paired
  multilingual results before changing retrieval defaults. Keep the previous KB active
  while the larger cohort is staged; delete no baseline collection or backup. No Stage 7
  start, human semantic acceptance, solved abstention or production SLO is implied.
- **Primary sources:** pinned Qdrant client 1.20 documentation,
  https://qdrant.tech/documentation/operations/optimize/ and
  https://qdrant.tech/documentation/operations/snapshots/ (checked 2026-11-12).

## ADR-032 — explicit technical profiles and dated sources for scale evaluation

- **Date:** 2026-11-13. **Status:** accepted for controlled fictional evaluation.
- **Problem/evidence:** the 18-document draft failed the requested scale/diversity gates.
  Repeating policy paragraphs or counting translations would not add independent facts.
  The expanded source must preserve all old values, SKUs and gold citations.
- **Alternatives:** clone rules with substituted product slots; append 200 records to the
  original catalogue; or author explicit subject/attribute specifications as separate source
  documents. Rule cloning adds no policy complexity. Rewriting the old catalogue changes
  existing source hashes/citations. Use the third option with a small typed source adapter
  on the existing render/parse/gate pipeline, no new service/framework or synthetic runtime.
- **Choice/cost:** 200 explicit fictional products across 20 groups, 2,600 numeric attributes,
  native RU/KK/EN names/units, four formats and an explicit stable SKU registry. Values
  are authored, not generated by random draws or arithmetic scaling. Separate four dated
  archival policies provide historical confusers without overwriting current policy values.
  Optional validity fields extend existing contracts; normal retrieval still filters dates.
- **Trade-off:** profile-derived questions test controlled subject/predicate discrimination,
  not thousands of independently human-authored customer intents. Shared templates and
  generic attributes correlate needs. Keep 119 families together before splitting, report
  family-cluster CIs and profile/non-profile sensitivity, retain draft judgement provenance
  and the absence of independent fluent reviewers. No human semantic acceptance is implied.
- **Verification:** the 222-document / 666-language-version / 8,865-chunk KB passes all 13
  gates and retains the original 369 chunk IDs. Required fact groups resolve to exact source
  placements; negative reasons cannot carry positive proof. Tests cover stable SKUs,
  idempotent authoring, manual-file protection, signed numeric boundaries, metadata-only
  date changes and overlapping successor rejection. Evaluation results live in PROGRESS.
- **Reproducibility:** normalize source LF before freezing. A pre-selection attempt was
  explicitly terminated after discovering three CRLF source fingerprints; keep its KILLED
  MLflow run and unchanged cohort evidence. Atomic query checkpoints now preserve completed
  candidate work across interruption; embedding and pair caches already resume inference.
- **Rollback/revisit:** activate the preserved baseline and its default namespace/config.
  Do not remove old sources/results/backups. Revisit template-heavy coverage when real
  synthetic conversation diversity or fluent independent review is available; Stage 7 may
  not treat these draft retrieval judgements as certified semantic grounding.

#### Operational application

Retain the frozen control and 1024-token context: alternatives fail selection. Score gates
sacrifice cross recall without adequate semantic rejection, so stay disabled. Exposed
holdout errors cannot tune packing in this cycle. Supplemental wording tests reuse dev
needs and cannot change selection. Use the expanded isolated namespace, preserve baseline
config/index and exercised rollback. No additional service needed. Future packing/gating
changes require new development evidence and a new confirmation cohort. Current results
and status are in PROGRESS.

## ADR-033 — validation-selected behaviour adapter and separate routing boundary

- **Date:** 2026-11-14. **Status:** accepted for frozen independent assessment;
  autonomous factual-answer and critical-escalation acceptance is not implied.
- **Problem/evidence:** the initial legacy smoke returned 0/6 valid responses. Full
  validation compares schema/few-shot 0.6B and 2B bases; eight 80-step OFAT training pilots
  compare rank, LR, targets, length and NF4. These are measured development alternatives,
  not a full factorial search or a universal LoRA-versus-QLoRA result. Unsafe eager
  packing is rejected before training rather than introducing a new attention framework.
- **Choice:** Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`,
  all-linear rank 16 / alpha 32, LR 1e-4, 512 steps, fp16 LoRA/FP32 trainable parameters,
  MT weight 0.5 and schema prompt. Primary seed 20261113 remains fixed. The validation
  score .83906 exceeds the Qwen3.5-2B few-shot control .51049; adapter few-shot fails
  the preregistered KK guard. Keep both finalists for independent assessment.
- **Evidence limits:** matched 0.6B paired schema/intent/action/language CIs exclude
  zero per language; KK intent superiority over the stronger 2B remains inconclusive.
  Absolute KK intent accuracy .51327 is still limited. Secondary seeds each log one
  NaN grad_norm; final weights are finite, but optimizer skip counts are UNKNOWN.
  No favorable-seed substitution or semantic-quality inference from JSON is allowed.
- **Routing:** keep the frozen E5-small/C10 head and typed RouteDecision interface.
  Its validation macro-F1 exceeds the measured base/adapter LLM routers per language.
  Action targets are conditional on omitted context/history; deterministic guards,
  raw decisions and fallback escalation remain distinct. Critical misses prevent an
  autonomous escalation guarantee; confidence is not calibrated.
- **Resources/cost:** use verified free Kaggle jobs for training and full inference.
  Remote fp16 and laptop NF4 are different execution conditions. The 2B joint local
  allocation exceeds physical 4GB; selected-adapter local/joint acceptance requires
  actual observations. Process-wall GPU-hours are separate from UNKNOWN session/billed
  allocation. No paid resource, new runtime service or external agent dependency is added.
- **Isolation/contracts:** seal model revisions/configs, prompts/demonstrations, head,
  validation evidence/source hashes and immutable inference-export manifests before
  opening test/gold. The export excludes training pickle and leaves checkpoints intact.
  Test/gold cannot retune these settings. Gold accepted references are not AND-proofs.
- **Safety:** only an explicit supported claim verdict can accept factual generation.
  With no calibrated independent verifier, UNKNOWN yields localized operator assistance
  without claiming a completed handoff. Semantic quality stays NOT_EVALUATED. Draft
  LLM-authored labels and missing fluent KK review limit all conclusions.
- **Verification/rollback/revisit:** complete raw receipts, per-language paired CIs,
  checkpoint 256 / resume 512, finite-weight audits and restored archives support this
  choice; details in `docs/reviews/stage7_results.md` and the sealed local reports.
  Preserve all failure attempts. Adapter=None and legacy prompt mode retain the earlier
  interface. Revisit parameters only with new development evidence and a fresh independent
  confirmation cohort; held-out failures change readiness, not frozen parameters.

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
