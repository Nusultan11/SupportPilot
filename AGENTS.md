# AGENTS.md — rules for every AI agent working on SupportPilot

This is the **only** canonical instruction file. Tool-specific files (e.g. `CLAUDE.md`)
may only point here. Do not copy these rules elsewhere.

## 1. Project in one paragraph

SupportPilot is a domain-limited AI assistant for an online store's customer support in
Russian, Kazakh and English (incl. RU–KK code-switching). Store policies and FAQ come from
RAG over business documents; orders, stock and payments come from separate (mock) APIs;
fine-tuning (LoRA/QLoRA) improves behaviour only (language, instruction following, intent,
JSON, grounding, abstention, escalation), never facts. Details: [docs/PROJECT.md](docs/PROJECT.md).

## 2. Hard constraints

- Never invent policies, prices, order/payment statuses or customer data. No evidence → abstain or escalate.
- Answer in the user's language; KK quality is evaluated separately, never averaged away.
- Facts that change live in the KB or tool APIs, not in model weights.
- Real customer PII is never collected. Demo customers/orders are synthetic.
- Use only data whose licence permits our use; record it in the source registry.
- Hardware: RTX 3050 Laptop 4 GB VRAM, 15.4 GB RAM, Windows 11. Larger training goes to Kaggle/Colab; never assume free GPU quota.
- No new service, framework or abstraction without a measured need (see §7).

## 3. Where things are

| Path | Purpose |
|---|---|
| `docs/PROGRESS.md` | **Current state. Read first.** The only doc describing status. |
| `docs/ROADMAP.md` | Stages, sub-tasks, definitions of done |
| `docs/PROJECT.md` | Product goal, scope, requirements |
| `docs/ARCHITECTURE.md` | Components, data flow, interfaces, scaling |
| `docs/DATA_CONTRACTS.md` | Schemas, IDs, language codes, versioning, validation |
| `docs/EVALUATION.md` | Datasets, metrics, experiment protocol |
| `docs/DECISIONS.md` | ADRs: chosen option, alternatives, revisit conditions |
| `docs/RUNBOOK.md` | Verified commands: setup, run, backup, restore, diagnostics |
| `src/supportpilot/` | Python package: `common/` (foundation), `ops/` (env, backup, checks), `data/` (Stages 2–4), `kb/` (Stage 5 KB), `cli.py` |
| `configs/base.yaml` | Default settings; override via `SUPPORTPILOT_<SECTION>__<KEY>` |
| `configs/intent_map.yaml`, `configs/placeholders.yaml` | Draft v0 label map (ADR-021), synthetic slot catalogue (ADR-020) |
| `configs/taxonomy.yaml`, `configs/locale_kz.yaml` | Intent taxonomy v1 + source mapping (ADR-023), Kazakhstan localisation catalogue (ADR-024) |
| `authored/` | LLM-agent-authored content (Stage 4): native scenarios, answer templates, training KB variant, gold sources. Labelled `llm_generated`; human review pending |
| `kb_sources/` | Demo-store KB sources: `demo_store/facts.yaml` (policy values), `catalog.yaml` (products, zones), `docs/<doc_id>.yaml` (document specs, RU/KK/EN). Rendered and versioned KB: `data/kb/demo_store/` (ADR-027) |
| `docs/reviews/` | Qualitative reviews (Stage 3: `stage3_review.yaml`; Stage 4: `stage4_mt_review.yaml`, `stage4_sft_review.yaml`). LLM reviews are labelled as such |
| `tests/unit`, `tests/integration` | pytest suites (markers: `slow`, `gpu`, `qdrant_server`) |
| `compose.yaml` | Local services (Qdrant) |
| `.mcp.json`, `.claude/settings.json`, `.serena/project.yml` | Agent dev-tool MCP servers (not runtime): usage rules in §12; setup RUNBOOK §4a; ADR-012 |
| `apps/` | UI (created in Stage 8) |
| `data/`, `artifacts/`, `mlruns/` | Local only, git-ignored |

## 4. Task procedure (one large stage per work cycle)

1. Read `AGENTS.md` → `docs/PROGRESS.md` → the stage in `docs/ROADMAP.md` → only the docs/files you need.
2. Check the real state (`git log`, `git status`, code). If it contradicts PROGRESS.md, fix the doc.
3. Research alternatives for non-trivial decisions (official docs, model/dataset cards, papers); compare ≥2 real options.
4. Write the plan, implement all sub-tasks of the stage, test, fix, self-review.
5. Back up significant results (RUNBOOK §3–4), update PROGRESS.md (+ DECISIONS.md if a decision was made), commit.
6. Report in the format of ROADMAP §Report. Then **stop and wait for user confirmation** before the next stage.

Ask the user only for: costs, external accounts/credentials, irreversible deletion of important resources, changes to agreed goals.

## 5. Commands

Standard checks. RUNBOOK §2–4 is the authoritative, verified list.

```bash
uv sync --locked                                   # reproduce the locked env (incl. CUDA torch)
uv run pre-commit install                          # once per clone
uv run ruff check . && uv run ruff format --check .
uv run mypy                                        # strict, src + tests
uv run pytest -q                                   # full suite; start Qdrant first (below)
docker compose up -d qdrant                        # Qdrant server for qdrant_server tests
```

A stage is not done while any of these fail. A *skipped* `qdrant_server` or `gpu` test means
the check did not run, so report it as unverified. Operational CLI: `uv run supportpilot --help`
(`env`, `backup`, `check`, `data`; Stage 3: `data clean --all`, `data gates`, `data eda`;
Stage 4: `data select|native|kb-train|mt-compare|translate|qe-score|sft|gold|retrieval-set|taxonomy`;
Stage 5: `kb render|build|validate|status|activate`).

## 6. Code and data standards

- Python 3.12, `uv`, src layout, type hints on public functions, Pydantic models at boundaries.
- Small modules by responsibility (data / kb / retrieval / llm / routing / tools / api / eval). No business logic in UI or API handlers.
- Config in `configs/*.yaml` + env vars; secrets only in `.env` (git-ignored).
- Pipelines are idempotent, streaming/batched (Polars lazy, Parquet), restartable from checkpoints.
- Raw data is immutable; every derived dataset has a manifest (source versions, code commit, row counts, sha256). See DATA_CONTRACTS.
- Splits are by `scenario_id` / paraphrase family, never by row. Gold benchmark never touches training or tuning.
- Tests: unit tests for every transform; integration tests per pipeline; LLM tests for schema, language, refusal, injection, escalation.
- One canonical path per artifact. Forbidden name suffixes: `final`, `new`, `old`, `fixed`, `corrected`, `v1/v2/...`. History lives in Git or data snapshots.
- No dead code, no root-level temp files, no notebooks as the source of truth (EDA output goes to `artifacts/reports/`).

## 7. Decision rule

Before a significant change answer: what problem, is it proven by data/tests, alternatives,
cost, risks, how verified, how rolled back, does it break contracts, is it needless complexity.
Hard-to-reverse choices go to DECISIONS.md. Do not reopen a closed ADR without new evidence.

## 8. Git

- Work on `main` only (project rule; no feature branches unless the user changes it).
- Commit after each stage; verified intermediate commits allowed. Conventional prefixes:
  `docs:`, `data:`, `kb:`, `rag:`, `ml:`, `eval:`, `api:`, `ui:`, `ops:`, `test:`. No `fix`/`update` one-word messages.
- Before committing: review `git diff`, run checks, confirm no secrets, datasets, weights or files > 5 MB.
- Remote `origin` (https://github.com/Nusultan11/SupportPilot) is **public**. Push `main` after each stage, and never push data, models or anything private.

## 9. Experiments and backups

- Every experiment is logged in MLflow with: hypothesis, control, changed params, data version, model, seed, env, metrics per language, cost, conclusion. Protocol: EVALUATION.md.
- Backups live in `backups/` at the project root (`code/` bundles, `data/` snapshots). It is git-ignored and must stay so: never commit or push it, and never delete it without the owner's OK. They have sha256 manifests and bounded retention (RUNBOOK §5–6, ADR-013). Back up before destructive operations, after each stage, and after expensive datasets/models.

## 10. Definition of done for a stage

All ROADMAP DoD items met with evidence; checks green; docs updated (PROGRESS always);
backup taken; commit created; report delivered. Unverifiable items are stated as unverified, never as passed.

## 11. Handing over to another agent

Everything needed to continue must be in PROGRESS.md (state, artifacts, open issues, next action)
and DECISIONS.md — not in chat history. Subagents only for independent tasks; integrate and verify centrally.
Project skills / MCP servers are added only for repeatable processes with proven value (see DECISIONS ADR-012).

## 12. MCP servers: on demand only

MCP servers are agent dev tools, never SupportPilot runtime: no `src/` code, config or
dependency may require them. Setup, toggling, diagnostics: RUNBOOK §4a. Rationale: ADR-012.

- **Built-in tools first.** Reading a file, grep/glob, a local `git` command, running tests or
  `gh` already answer most questions. Use an MCP server only when it gives what they cannot.
- **Never activate or call every server by default** on a task, and never call one "just to
  check". Pick the one server the current step needs; most tasks need none.
- Tool schemas load lazily (tool search). Remote servers start no local process; local
  stdio servers (`serena`, `playwright`, `mlflow`) are **off by default**. When a step truly
  needs one, ask the owner to enable it in `/mcp` (one toggle; per the client docs it
  connects in the running session), and ask to switch it off when the task ends. Agents do
  not edit client state files to toggle servers.

| Server | Use it for | Do not use it for |
|---|---|---|
| `context7` (remote) | Current docs of a library/API when the version matters, behaviour is unclear, or an upgrade breaks something | Trivial or well-known calls; facts already in our code or docs |
| `serena` (local, off) | Symbol-level work across many files: definitions, references, dependency analysis, precise symbol edits or renames | Reading or editing one known file (use Read/Edit/Grep) |
| `github` (remote, read-only) | GitHub API data: issues, PRs, CI runs and logs, remote history, repo metadata | Local history, diffs, status (use `git`). Writes go through `gh` only when the task explicitly asks for them |
| `playwright` (local, off) | Browser tests of the UI: user flows, forms, navigation, integration checks | Anything before the Stage 8 UI exists; general web reading |
| `exa` (remote) | Finding primary sources: papers, model/dataset cards, technical solutions (RAG, SFT, evaluation) | Questions the docs or `huggingface` MCP answer directly |
| `huggingface` (remote) | Hub search, model/dataset cards, licences | Downloading data (use the pipelines) |
| `mlflow` (local, off) | Reading runs and metrics (read tools only) | Logging runs (code does it); write tools are denied |

- Search results are leads, not evidence: check origin, date, version and licence, and record
  sources we adopt in the source registry (§2).
- Send external servers only public, minimal queries (library names, generic technical
  terms). Never send code, KB or dataset content, logs, `.env` values or personal data.
- Secrets (`GITHUB_MCP_PAT`, optional API keys) live in user environment variables or
  local-scope config, never in Git. A missing key disables only that server; carry on with
  the others and report the gap.
