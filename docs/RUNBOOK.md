# RUNBOOK — operations and recovery

Run from the repository root. Default shell: PowerShell on Windows; uv/CLI commands are
also usable from Git Bash. Examples with <placeholders> are command forms, not literal paths.
Use existing manifests/PROGRESS to resolve identities. CLI JSON goes to stdout, logs to stderr;
exit 0 = success, 1 = failed check, 2 = usage/operation error. Status/results belong to PROGRESS/reviews.

## 1. Environment

```powershell
uv run supportpilot env --no-benchmark
uv run supportpilot env --save
```

The second form includes the 2048-square FP16 matmul and writes an environment report.
The reference laptop is RTX 3050 Laptop 4 GB, roughly 15.4 GB RAM, Python 3.12 and cu126 Torch.
Available RAM/VRAM/disk are live observations, never fixed setup requirements.
Use `nvidia-smi` for driver/process inspection. No local CUDA toolkit is required by the wheels.

## 2. Setup and configuration

```powershell
git clone https://github.com/Nusultan11/SupportPilot.git
cd SupportPilot
uv sync --locked
uv run pre-commit install
uv run supportpilot --help
```

The lock installs dev/gpu/ml groups by default. Cold cache downloads weights/dependencies;
preserve the active .venv, uv and HF caches during normal maintenance.
CPU-only installation requires separately verified dependency choices; it is not a claim
that every pipeline runs without Torch.
Settings: configs/base.yaml, `--config` or SUPPORTPILOT_CONFIG.
Environment overrides: SUPPORTPILOT_<SECTION>__<KEY>. Secrets belong in ignored .env/env.

## 3. Quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
docker compose up -d qdrant
uv run pytest -q
uv run pre-commit run --all-files
```

Full pytest includes GPU/server checks when resources exist. Skipped gpu/qdrant_server cases
are unverified, not passed. Pre-commit runs only the fast test subset:
`uv run pytest -q -m "not slow and not gpu and not qdrant_server"`.
Read warning/skip/failure summaries and preserve the log.

Use pytest's system-temp default, or --basetemp pointing to a new task-owned directory
outside the checkout. Git-isolation tests otherwise discover the ignored parent repository:
an inside-checkout run failed for that reason, while the isolated test passed unchanged.
Pytest owns/recreates basetemp; never point it at data, models or reports. Record the exact
task-owned path and remove it after verification, retaining required failure evidence.
Other disposable work belongs under artifacts/tmp; create its parent before using it.

If sandboxed uv cannot write its interpreter/cache metadata, use the already-synchronized
`.venv/Scripts/python.exe -m pytest`, `-m mypy`, `-m ruff`, or request the authorized
operation outside the sandbox. Do not rebuild/delete the environment as a workaround.

## 4. Services

```powershell
docker desktop start --detach
docker compose up -d qdrant
docker inspect -f '{{.State.Health.Status}}' supportpilot-qdrant
uv run supportpilot check qdrant
uv run supportpilot check qdrant --mode memory
uv run supportpilot check mlflow
```

Use127.0.0.1:6333, not localhost (observed IPv6 delay). Qdrant uses its named Docker volume.
`docker compose down` stops services without deleting that volume; never add -v for cleanup.
Do not stop unrelated containers/processes. Temporary service isolation requires ownership
checks and restoration of health/collection inventory afterward.

MLflow:
```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5055
```

SQLite and artifacts under mlruns are one recovery unit. Snapshot after writers close;
verify restored SQLite integrity. The UI/MCP is not required to log runs.

## 4a. Agent-tool setup

MCP is optional development tooling, not runtime. Rules: AGENTS §12; decision: ADR-012.
.mcp.json and .serena/project.yml contain project setup; user-local Codex/Claude state is private.
Do not copy secrets into them or edit private client state to change activation.

Remote Context7/Exa/HuggingFace use public minimal queries; GitHub is read-only and uses
GITHUB_MCP_PAT from user environment. Local Serena/Playwright/MLflow are off by default;
the owner enables one when a task needs it, then disables it.
Claude Code's verified2026-11-12 setup uses /mcp to disable local project servers once per clone.
Other clients may differ; use that client's supported controls, never assume identical toggles.
Serena keeps tool/no-onboarding/no-memories settings; Playwright uses installed isolated Edge;
MLflow is read-only. Pipelines/tests must work with all servers disconnected.
Current connector availability/keys are not project implementation status.

### Optional Shrimp + Structurizr workflow (ADR-036)

For substantial authorized engineering work only. Prerequisites: Node.js>=20 (verified24.18),
npm (verified11.16) and existing Docker Desktop. Runtime uv/Compose Qdrant usage stays unchanged.
From this repository root:

```powershell
npm ci --prefix tools/mcp --ignore-scripts --no-fund --no-audit
node tools/mcp/configure.mjs
docker compose --profile engineering up -d structurizr
node tools/mcp/check.mjs
```

The lockfile pins Shrimp1.0.21/SDK and the Compose image is digest-pinned. No install-time scripts
run. configure.mjs adds only two managed entries to ignored project .codex/config.toml, preserves
other settings and refuses conflicting preexisting entries. It is idempotent; no global client
state or permission settings are changed. Codex verifies the entries with:

```powershell
codex mcp get shrimp --json
codex mcp get structurizr --json
codex -c 'mcp_servers.shrimp.enabled=true' -c 'mcp_servers.structurizr.enabled=true'
```

The installed entries default to enabled=false; the last command opts in for that CLI session.
For Shrimp-only work omit the Structurizr override/service. An already-running desktop chat's
catalogue is not hot-reloaded by the installer. A fresh client session/its supported MCP controls
are needed for native agent tool exposure; the checks above use actual protocol clients without
an LLM/account/billing call. No security bypass flags are needed. Keep other local MCPs off unless
their existing activation requirements are met. Managed policy may independently restrict tools.

Claude Code: the existing .mcp.json remains the default unrelated-tool configuration. The separate
opt-in configuration avoids adding engineering servers to every ordinary session:

```powershell
claude --mcp-config tools/mcp/claude.json
```

Run from the root; Claude Code merges its normal configuration unless its documented strict mode
is selected. Existing local MCP controls still apply. JSON config is for clients supporting MCP
stdio and streamable HTTP; a stdio-only desktop client needs a separately verified bridge, not
an assumed transport substitution. Codex config resolution and direct MCP transports were tested;
interactive Claude/other-desktop tool use must be verified in that client before claiming it.

check.mjs checks initialize/tools/list/calls,10 allowlisted Shrimp tools, persistent reads across
separate processes and changed client roots, a competing-writer refusal, forbidden destructive
modes/tools and oversized inputs. It calls Structurizr validate/parse/inspect, checks all four
view keys/types, and rejects invalid DSL. It does not run models or change task status.
Receipts: artifacts/reports/engineering/mcp/attempts/; latest-check.json locates the latest
attempt without replacing earlier success/failure evidence. Two INFO inspections refer to documentation/ADRs
intentionally kept in their owning Markdown documents; no blanket inspection suppression exists.

For explicit MCP calls when the host cannot reload tools:

```powershell
node tools/mcp/client.mjs shrimp list
node tools/mcp/client.mjs shrimp list_tasks <arguments.json> <receipt.json>
node tools/mcp/client.mjs structurizr validate <dsl-arguments.json> <receipt.json>
```

The first lists schemas. list_tasks arguments are `{"status":"all"}`. Structurizr arguments
are `{"dsl":"<workspace text>"}`; create JSON with a serializer, never shell interpolation.
Shrimp split_tasks takes `updateMode: "append"|"selective"` and `tasksRaw`, a JSON-encoded array
of tasks (not a tasks array field). Set name/description/implementationGuide/dependencies/
verificationCriteria and concise evidence/continuation pointers. Dependency references use
existing task names/UUIDs. Read returned errors AND persisted state: upstream tools sometimes
return textual refusals without isError. execute_task marks an eligible task in_progress;
verify_task records summary/score and completes at>=80. A lower score leaves it in progress.
Completion requires actual task evidence, and stage completion still uses ROADMAP criteria.

Resume: read AGENTS -> PROGRESS -> relevant ROADMAP scope -> list/query/get_task_detail for the
current task. Check Git/artifacts before executing; use update_task for an unfinished task's
next action/blocker. Do not seed completed stages or create a parallel journal. Historical state
is referenced in task scope notes; only this authorized engineering task was decomposed here.
There is no automatic stage scheduler or permission to start Stage 10.

Persistent store: artifacts/engineering/shrimp/tasks.json, with UUID dependency edges, statuses,
verification criteria and completed summaries. GUI is off. Client roots never alter its path.
The local adapter excludes rules generation/deletion/reset/thought-history tools, limits each
call to256KiB and<=100 tasks, serializes calls, and refuses a second writer. Do not bypass it
with a direct upstream launch. No secrets are forwarded to the upstream child environment.
Shrimp does not provide transactional/power-loss-safe writes; after interruption preserve all
bytes and restore a verified snapshot to an isolated target before replacing any live state.
For a stale writer.lock inspect its recorded PID and actual process command/ownership; remove
only that task-owned lock after proving its process exited. Never reset tasks to recover a lock.

Stop the client before snapshotting the task store with §6. Deep-verify and actually restore it;
compare all IDs/dependencies/statuses/results and the file hashes. Snapshot engineering receipts
separately; code bundle recovery follows §5. Task state and outputs are ignored, never public.
Inspect `npm audit --prefix tools/mcp --omit=dev` before a dependency change. The verified tree
retains the upstream Zod3 oversized-collection advisory; do not force a breaking major upgrade.
It is local trusted stdio with bounded input, not an exposed task HTTP service. Audit receipts
include the original negative attempt and the updated tree; no zero-vulnerability claim is made.

Structurizr exposes only loopback3010/mcp and only validate/parse/inspect, with no host mounts or
cloud publishing. Treat submitted DSL as executable-tool input; use the reviewed canonical
self-contained model, never untrusted !script/!plugin/remote includes. Stop after verification:

```powershell
docker compose --profile engineering stop structurizr
```

Do not use compose down/-v to stop this tool: preserve Qdrant and its volume/collections.
Shrimp exits with its stdio client; no GUI/viewer/server daemon is required between tasks.

## 5. Code backup and restore

Location: backups/code (ignored; ADR-013). Create a unique bundle:
```powershell
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$bundle = Join-Path (Get-Location) "backups/code/supportpilot-$stamp.bundle"
git bundle create $bundle --all
git bundle verify $bundle
Get-FileHash -LiteralPath $bundle -Algorithm SHA256
```

Record SHA256 in a sidecar/receipt. Actually clone the bundle into a new isolated directory,
compare HEAD and a clean worktree; never restore over the live repository:
```powershell
git clone -b main <bundle-path> <new-restore-directory>
git -C <new-restore-directory> rev-parse HEAD
git -C <new-restore-directory> status --porcelain
```

GitHub/main protects public code only. Same-disk bundles do not protect against disk loss.
Retention eligibility is latest10 plus pinned stage bundles; deletion needs separate owner approval.
After an actual verified drill, the temporary clone can be removed under §17.

## 6. Data/model snapshots

Root: ignored backups. Layout:
`<root>/<data|model>/<name>/<snapshot_id>/{archive.tar,manifest.json}`.
Uncompressed tar and per-file/archive/content hashes support atomic publication and dedup.
A root inside the repository is refused unless Git ignores it.

```text
uv run supportpilot backup create <source> --name <name> --kind data --pin --note "<purpose>"
uv run supportpilot backup list
uv run supportpilot backup verify data/<name>/<snapshot_id>
uv run supportpilot backup verify data/<name>/<snapshot_id> --quick
uv run supportpilot backup restore data/<name>/<snapshot_id> --target <new-or-empty-directory>
uv run supportpilot backup pin data/<name>/<snapshot_id>
uv run supportpilot backup prune --keep 3
```

Verify defaults to deep (archive and all member hashes); quick verifies archive bytes only.
Restore verifies every file before publishing and refuses a nonempty target.
Create reuses identical newest content; pin stage/expensive results.
Prune without --yes is a dry-run eligibility report. **Do not execute deletion or unpinning
without separate owner authorization**, even for unpinned/old snapshots.

Snapshot datasets, KB/indexes/caches, raw experiment outputs/checkpoints, reports and MLflow
after significant results. Record manifest/restore/check receipts, not repeated full path lists
in every document. Verify DB integrity with PRAGMA integrity_check on the restored SQLite copy.
Do not keep full duplicate restore trees indefinitely after the drill.

## 7. Source acquisition and conversion

Registry/rights: configs/sources.yaml and DATA_CONTRACTS §2.
```powershell
uv run supportpilot data sources
uv run supportpilot data acquire --all
uv run supportpilot data convert --all
uv run supportpilot data verify --all
```

Explicit source IDs and --force conversion are supported. Reruns verify and skip unchanged work.
Interrupted .partial-revision downloads resume/reverify with the same command.
Corrupt immutable raw revisions stop acquisition/conversion: restore, never edit them.
A gated source needs owner account/accepted terms/HF_TOKEN, registry eligibility and a converter;
this is not permission to acquire every candidate. Back up newly acquired revisions/records.

## 7b. Cleaning, canonical gates and EDA

```powershell
uv run supportpilot data clean --all
uv run supportpilot data gates
uv run supportpilot data eda
uv run supportpilot data eda --no-tokens
```

Per-source clean and --force rebuild exist. Configuration: clean/eda sections, draft intent map
and synthetic placeholder catalogue. Changed transformation logic must invalidate its cache version.
Atomic checkpoints preserve the prior build on interruption/gate failure; inspect
data/interim/canonical.failed/quality_report.json. Do not erase failure evidence blindly.
EDA samples and annotations are linked by sample identity; changed samples require re-review.
Back up clean/canonical/EDA results. Contracts: DATA_CONTRACTS §§4–4d/10.

## 7c. SFT, MT, training KB and gold

```powershell
uv run supportpilot data taxonomy
uv run supportpilot data select
uv run supportpilot data kb-train
uv run supportpilot data mt-compare
uv run supportpilot data translate --lang kk
uv run supportpilot data translate --lang ru
uv run supportpilot data qe-score
uv run supportpilot data sft
uv run supportpilot data gold
```

Pinned models download into HF cache on first use; set HF_HUB_OFFLINE=1 only after weights
are complete. MT shards resume; SFT does not silently fill missing translations.
--no-semantic diagnostics leave semantic leakage checking unverified.
Never change frozen gold questions to accommodate copied training input: exclude copied SFT
inputs under ADR-029. Gates retain failed builds/current pointer and explicit reasons.
Back up stage4 interim/MT caches, training KB, each published SFT and gold version.
Do not rerun these commands against the active sealed Stage 7 cohort for housekeeping.

## 7d. Knowledge base

```powershell
uv run supportpilot kb render
uv run supportpilot kb build
uv run supportpilot kb validate
uv run supportpilot kb status
```

Explicit --version validation, --force build and `kb activate <kb_version>` support diagnostics/
rollback. Build reads specs/facts/catalogue, renders deterministically, parses/chunks/gates and
publishes immutable versions. Same content reuses a version; changed documents produce changes.json.
Quote YAML values containing commas; scanned PDF/invalid structure fail explicitly.
A failed version never replaces current. Edited rendered files require rerendering, not hash edits.
Restore a self-contained KB snapshot to an isolated data root and validate it before activation.
Activation and retrieval synchronization remain separate (§9/§16).

## 7e. Native inputs and independent retrieval queries

```powershell
uv run supportpilot data native
uv run supportpilot data retrieval-set
```

Native licensed inputs use normal acquire/convert/clean policy; only explicit reliable labels enter.
SFT refreshes selection/native/QE/training KB as needed and preserves heldout coverage.
Retrieval current.json points to queries/qrels/manifest bound to a KB/SFT version.
After real KB/SFT changes rebuild with leakage/reference gates; do not alter frozen cohorts in place.
No training or retrieval inference is performed by data retrieval-set itself.
Review provenance remains draft/LLM where no fluent independent review exists.

## 8. Diagnostics

| Symptom | Action |
|---|---|
| CUDA unavailable | Check nvidia-smi, locked cu126 Torch and driver; do not reinstall blindly |
| CUDA OOM/shared-memory slowdown | Inspect actual scope/other processes; reduce batch/context only in a new authorized configuration |
| Docker npipe unavailable | Start Docker Desktop, wait for docker info, then compose/health |
| Embedded SQLite locked | Close owning process; don't remove live lock/storage files |
| Config error | Check named unknown/invalid YAML/env key; extra fields are forbidden |
| Raw immutable/hash failure | Restore verified original, never repair its manifest |
| Data gate failure | Inspect failed quality_report; previous current output should remain |
| Missing translation/cache | Resume owning pipeline; clear HF_HUB_OFFLINE only for authorized missing downloads |
| Pre-commit line-ending changes | Review, stage fixed LF files, repeat commit |
| Redirected KK text corrupted | Set PYTHONUTF8=1 / PYTHONIOENCODING=utf-8 |
| HTTP/private-kernel access denied | Record status UNKNOWN; use owner OAuth recovery below; do not infer remote failure |

## 9. Retrieval and experiments

Model registry: configs/retrieval_models.yaml. Parameters/namespace: configs/base.yaml.
```powershell
uv run supportpilot retrieval index
uv run supportpilot retrieval query 'Алматыға жеткізу қанша тұрады?' --lang ru --budget 1024 --as-of 2026-11-12
uv run supportpilot retrieval query 'Доставка' --lang kk --doc-id missing-document --as-of 2026-11-12
```

--lang filters corpus, independently of input language; omit to search all languages.
--doc-id/--doc-type repeat; --as-of is inclusive. Serving defaults to today;
experiments pin evaluation_date. Empty filters/budget produce explicit no-evidence reasons.
Metadata consumes context budget; chunks/facts are not cut to fit.
For warm repeated requests use `with open_retriever(settings)`; CLI startup is cold.

Resume retrieval index with the same inputs; preserve checksummed caches/OS lock files.
After KB activation, synchronize index before serving. Explicit --kb-version indexing does
not activate KB; fail-closed readers reject mismatch. Rollback uses exact prior inventories.
Keep KB-scoped sparse IDF; global IDF drifted after delete/restore.

```powershell
uv run supportpilot retrieval experiment --stage prepare
uv run supportpilot retrieval experiment --stage compare
uv run supportpilot retrieval experiment --stage batch
uv run supportpilot retrieval experiment --stage rerank
uv run supportpilot retrieval experiment --stage report
uv run supportpilot retrieval experiment --stage profile
uv run supportpilot retrieval experiment --stage drill
```

These are experiment entry points, not permission to retune an already exposed/frozen study.
Check artifact/environment bindings before resuming. Profile bypasses query/pair caches;
drill restores a saved snapshot into a separate collection and verifies vectors/payloads/queries.
Baseline reports under artifacts/reports/retrieval are frozen; delivery receipt sits outside them.
Do not overwrite raw failures/measurements or infer semantic answerability from retrieval scores.

## 16. Scale retrieval and serving rollback

The historical section number is retained for existing references.
Preregistration: docs/reviews/stage6_scale_protocol.md; configured phases:
```powershell
$cohort='data/benchmarks/retrieval_scale/6fab32c16b3a6e05b29f'
uv run supportpilot retrieval scale --stage audit --semantic --kb-version kb_demo_store-20261113-f3b204a6
uv run supportpilot retrieval scale --stage development --frozen $cohort
uv run supportpilot retrieval scale --stage holdout --frozen $cohort
uv run supportpilot retrieval scale --stage profile --frozen $cohort
uv run supportpilot retrieval scale --stage recovery --frozen $cohort --restore-root artifacts/restore_drills/stage6_scale_runtime
uv run supportpilot retrieval scale --stage diagnostic --frozen $cohort
uv run supportpilot retrieval scale --stage report --frozen $cohort
uv run supportpilot retrieval scale --stage load --count 100000
uv run supportpilot retrieval scale --stage update-drill
```

author/freeze phases also exist; do not regenerate a sealed input during cleanup.
Recovery requires the verified isolated KB/index runtime. Check local/terminal MLflow receipts.
Random-vector load/update collections never enter serving or quality qrels.

Expanded serving uses the namespace/KB named in PROGRESS.
Baseline configuration is preserved at
`artifacts/reports/retrieval_scale/6fab32c16b3a6e05b29f/serving/baseline_config.yaml`.
To roll back, activate baseline KB `kb_demo_store-20261112-d58e3f93`, use global --config
with that file, synchronize and verify evidence. Restore expanded KB before current config.
The actual exercised transition is in serving/activation_rollback.json. Never delete either
collection or modify frozen queries/proofs/selection to force a resume.

## Stage 7 private cloud execution

Verified host clients: Kaggle CLI 2.2.5 / kagglesdk 0.1.38 (2026-11-14).
Prepare/verify the concrete allowlisted private package and respect its existing owner authorization.
Use absolute Windows Dataset paths; relative/normalized upload paths previously failed.
Example command forms from selected-training execution:

```powershell
kaggle datasets status nurs2005/supportpilot-stage7-selected-training
kaggle kernels status nurs2005/supportpilot-stage7-selected-training-run/1
kaggle kernels status nurs2005/supportpilot-stage7-final-evaluation-run/2
kaggle quota -v
```

Dataset create/version and kernel push are external writes, not monitoring actions:
do not submit a duplicate merely because a job is queued. Remaining quota does not prove GPU
assignment; rounded weekly quota is not experiment GPU-hours. Timeout limits execution,
not queue wait. Two queued own batch jobs were observed to occupy the two batch slots.

OAuth recovery when cached login exists but API access/quota is denied:
```powershell
kaggle auth login --force
kaggle quota -v
kaggle kernels status nurs2005/supportpilot-stage7-final-evaluation-run/2
```

Owner completes browser sign-in; never request token/password contents. Browser/CLI login differ.
Keep last observed state plus UNKNOWN_ACCESS_DENIED; stop repeated unauthorized polling.
A transient query failure does not authorize cancelling/restarting the remote computation.

If the existing cached OAuth grant is still valid, the official SDK can renew the access token
without another browser login (verified 2026-11-15). Run with the Kaggle-enabled Python:
```python
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kaggle_creds import KaggleCredentials

api = KaggleApi()
api.authenticate()
with api.build_kaggle_client() as client:
    credentials = KaggleCredentials.load(client=client)
    assert credentials is not None and credentials.get_username() == "nurs2005"
    credentials.refresh_access_token()
```
Then start a fresh CLI query/client and verify the exact version. Never print the credentials.
An invalid/revoked grant still requires owner sign-in; renewal never changes experiment inputs.

Require terminal COMPLETE for an explicit version before quality import:
```powershell
kaggle kernels pull nurs2005/supportpilot-stage7-routing-run/2 -p artifacts/kaggle/stage7_routing/remote/2 -m
kaggle kernels output nurs2005/supportpilot-stage7-routing-run/2 -p artifacts/kaggle/stage7_routing/output/2 --file-pattern '^results/' -q
```

Compare submitted/downloaded source, private metadata/input identity and every result checksum
with llm.results.verify_kernel_source/verify_results before import_results.
Preserve partial/earlier archives; resume downloads in exact disjoint groups against the checksum
manifest. Keep download receipts outside checksummed results directories.
Remote execution provenance differs from local importer environment/Git metadata.

The CLI has no cancel command. SDK cancellation takes kernel_session_id, not kernel ID.
Resolve exact-version owner/slug/version through download_kernel_output and retain its proven
session mapping. Download URLs may later redirect to object storage; use the verified session
mapping with download_kernel_output_zip rather than guessing numeric IDs from a GCS URL.
Before cancelling any own queued task, recheck status and output inventory; preserve observations.
Retain CANCEL_ACKNOWLEDGED and empty archive hash. No notebook/Dataset deletion is needed.

Payload packaging refuses changed bytes at an existing destination; identical replay preserves it.
Export a verified immutable inference adapter first, exclude training pickle, retain raw checkpoints.
A mounted Dataset may expand payload.zip; bootstrap verifies either layout's manifest.
One GPU per subprocess avoids TRL DataParallel mismatch. Checkpoint resume requires the original
runtime/config/input binding; increased steps are a distinct run. Sealed prompts/adapters do not
change during retry or housekeeping. Current job/helpers are listed only in PROGRESS.

## 17. Safe maintenance

1. Inventory tracked/untracked/ignored files, per-directory bytes/counts, active jobs and recovery
   dependencies. Keep a local cleanup plan/receipt under artifacts/reports/maintenance.
2. Protect immutable raw/dataset/KB/index/model/checkpoint/benchmark outputs, MLflow, seals,
   active environments, needed model caches and Qdrant volume. Never modify checksums to hide drift.
3. Remove proven disposable Python/tool caches and finished test work. Never blanket-delete
   .partial files, lock files, ignored directories or all files matched by an extension.
4. A restore-drill tree is disposable only after matching retained backup files/hashes and deep
   archive verification (or a clean cloned HEAD with a verified bundle). Preserve proof/receipts.
   If unique/mismatched, keep it until its purpose is established.
5. Use one shell and literal resolved targets; verify each recursive target stays inside its
   intended workspace directory. Reject symlinks/junctions. Do not delete backups without separate
   owner permission, regardless of age, retention or successful drills.
6. Report logical removed-file bytes/counts separately from observed free-disk change.
   Concurrent system writes, allocation and hardlinks can make them differ.
7. Run applicable/full checks, verify frozen identities/operational health, clear disposable test
   work created by validation, update the owning docs and publish source/docs only.

Keep one canonical maintenance summary and machine-readable evidence; don't copy deletion logs
into PROGRESS. Remove temporary debug helpers when no longer needed; retain a small reproducer
only when needed to audit/repeat a meaningful operation. AGENTS §7 owns general lifecycle rules.

## 18. Local end-to-end prototype (Stage 8)

Use the current local sealed router/adapter and active KB/index from PROGRESS; preserve their
manifests and caches. Start Qdrant using §4. Install dependencies with `uv sync --locked`.
Create local bearer credentials (principal -> secret), initialize synthetic business state, then
start the API in one terminal:

```powershell
$aliceToken = .venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$bobToken = .venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$operatorToken = .venv/Scripts/python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
$env:SUPPORTPILOT_SERVING__TOKENS = (@{demo_alice=$aliceToken; demo_bob=$bobToken; operator=$operatorToken} | ConvertTo-Json -Compress)
uv run python -m supportpilot.service.demo seed
uv run uvicorn supportpilot.api:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

In another terminal, run the UI, enter `$aliceToken` from the first terminal into the password
field (never put it into a public log/document), and choose auto/RU/KK/EN:

```powershell
uv run streamlit run apps/ui.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. Try `Где заказ SP-1001`, then `Оплата`; read shared stock with
`Наличие TV-5501`; ask a policy question and inspect retrieved sources/queue status; request an
operator and refresh tickets. `SP-1002` belongs to demo_bob and must not disclose its state to
Alice. A new session or changed bearer token clears UI history. Confirm a cancellation request
only with the explicit checkbox; the result records a request, never completed cancellation.
Customers cannot advance queue status. An operator bearer may GET /escalations and PATCH
/escalations/<UUID> with `{"status":"in_progress"}` then `{"status":"resolved"}`.
GET /health is public; /ready and all customer/queue operations need a Bearer header.
Use UUID request_id per new /chat operation; preserve it for an identical network retry.

The default mode loads retrieval/router but omits generation; it retrieves source material and
queues questions requiring unverified semantics. To exercise research inference, set
`SUPPORTPILOT_SERVING__GENERATION_ENABLED=true` before starting the API. This uses the selected
adapter with operational NF4/45-second generation limit, not the sealed FP16 assessment mode.
No semantic verifier is enabled. Driver hangs require process restart; sessions expire on
restart, but SQLite business state/queue/receipts survive. Stop task-owned servers with Ctrl+C.
Do not run multiple workers, expose these demo services publicly, or use real customer data.

One command runs the controlled multilingual HTTP/SQLite/UI suite and actual local components:

```powershell
uv run python -m supportpilot.service.demo e2e --generation
```

For only the 300 scenario-state cases and separate boundary/UI tests:
`uv run pytest tests/integration/test_service.py -q`.
For only the actual-model operational probe: `uv run python -m supportpilot.service.demo probe --generation`.
Both probe modes create isolated synthetic databases under ignored attempt directories and retain
failures, prompts, completions and observations; `artifacts/reports/service/latest_probe.json`
points to the latest attempt. They do not overwrite live demo state or evaluate exposed gold.
Use full shared checks in §3 as well. Semantic/KK/generalization/sustained-serving limitations
remain separate from successful execution; evaluation scope is EVALUATION §10.

### Owner manual LLM workbench

Run the separate read-only inspection UI; no bearer token or main API setup is required:

```powershell
uv run streamlit run apps/manual_testing.py --server.address 127.0.0.1 --server.port 8502 --server.headless true --browser.gatherUsageStats false
```

Open http://127.0.0.1:8502, choose RU/KK/EN reply language and ask a synthetic question, including
mixed RU-KK. The selected Qwen3-0.6B adapter generates actual raw completions using the current
KB/index and schema prompt. Each question is independent; display history is not model memory.
Sources are exactly the packed chunks sent to the model, including dates/locators/checksums.
Invalid output is shown verbatim, with no repair or substituted answer. Diagnostics reuse the
schema, citation/number/tool checks and language detector; they never certify semantic support.
The worker generates even with empty evidence so abstention behavior can be inspected explicitly.

Only existing cached models are loaded (Hub/Transformers offline, telemetry disabled); no paid
API, tool execution, Store/queue access or qualified verifier is involved. Qdrant must be healthy
on loopback (§4); index readiness remains enforced. Retrieval runs on CPU to reserve laptop VRAM;
generation uses the operational NF4/45s profile, not the frozen FP16 assessment configuration.
A serialized disposable child owns models; the180s parent deadline terminates a hung worker.
The next query reloads it automatically. Errors/timeouts stay in ignored
`artifacts/reports/manual_testing/attempts/<id>/record.json`, alongside prompt, raw output, packed
sources, configuration/selection/source identity and available resource observations. An unknown
call count after worker loss remains UNKNOWN. UI download exports the same private record.
Use fictional data only; these local records are not training/gold or independent qualification.
Screen clearing removes only displayed history. Stop the task-owned Streamlit process to release
its worker. Leave the primary API/UI, models, KB/index and all earlier evidence unchanged.

## 19. System evaluation (Stage 9)

For the separately authorized quality continuation, use the new frozen inputs/protocol and
private `artifacts/reports/system/quality/` directory. Do not reuse original exposed holdout:

```powershell
uv run python -m supportpilot.service.quality_evaluation freeze
uv run python -m supportpilot.service.quality_audit
uv run python -m supportpilot.service.quality_evaluation evaluate --smoke
uv run python -m supportpilot.service.quality_evaluation evaluate --phase development --variant control
uv run python -m supportpilot.service.quality_evaluation evaluate --phase development --variant candidate
uv run python -m supportpilot.service.quality_evaluation select --control <control-report.json> --candidate <candidate-report.json>
uv run python -m supportpilot.service.quality_evaluation evaluate --phase confirmation --variant control
uv run python -m supportpilot.service.quality_evaluation evaluate --phase confirmation --variant candidate
uv run python -m supportpilot.service.quality_evaluation report
```

These commands describe the continuation sequence, not permission to repeat consumed confirmation.
Selection/runtime mismatch and existing confirmation markers fail closed. Preserve failed and
superseded development attempts; a source change requires a new development candidate before
sealing. The audit reuses checksum-verified pinned reference embeddings and verifies original
reference readbacks; prior system needs and cross-split embedding flags are added explicitly.
Flags never automatically remove/relabel rows or certify semantic independence. The full local
evaluation needs access to Qdrant, model caches and MLflow SQLite; sandbox denial is an execution
failure, not evidence of remote/service/model quality.

Run only on isolated synthetic state, with the existing Qdrant healthy. No new paid service,
model selection or exposed Stage 7 test/gold rerun is involved. Protocol and source inputs are
immutable after freeze. Results/status belong to the Stage 9 review and PROGRESS.

```powershell
uv run python -m supportpilot.service.evaluation freeze
uv run python -m supportpilot.service.evaluation_audit
uv run python -m supportpilot.service.evaluation evaluate --smoke
uv run python -m supportpilot.service.evaluation evaluate --phase development --variant stage8
uv run python -m supportpilot.service.evaluation evaluate --phase development --variant boundary_guard
uv run python -m supportpilot.service.evaluation select --control <control-report.json> --candidate <candidate-report.json>
uv run python -m supportpilot.service.evaluation evaluate --phase holdout --variant stage8
uv run python -m supportpilot.service.evaluation evaluate --phase holdout --variant boundary_guard
```

These are the verified original sequence, not permission to refresh inspected confirmation.
`freeze_pointer.json` locates the immutable input; attempts/selection/holdout markers retain exact
identities. Existing selection/holdout markers reject blind reruns. A technical retry must retain
the failed attempt and exact source/configuration/input identity; changing labels or creating a
fresh output path cannot restore independence. `stage8` swaps only the captured original guard;
both variants otherwise use the same current frozen runtime/components.

```powershell
uv run python -m supportpilot.service.performance retrieval
uv run python -m supportpilot.service.performance http
uv run python -m supportpilot.service.performance precision --prompts <captured_prompts.json> --quantization none
uv run python -m supportpilot.service.performance precision --prompts <same-captured_prompts.json> --quantization nf4
uv run python -m supportpilot.service.performance precision_joint --prompts <same-captured_prompts.json> --quantization nf4
```

The retrieval study emits captured prompts in its unique attempt directory. Do not run concurrent
model jobs during resource measurements. HTTP uses an ephemeral loopback port/credentials and
isolated SQLite; its task-owned Uvicorn thread stops in finally. Report failures/503 and successful
throughput separately. `precision_joint` performs actual retrieval with the generator resident.
Limits, output-token truncation and semantic UNKNOWN remain visible; schema validity is diagnostic.

Without independent review, the semantic entry point emits NOT_EVALUATED:

```powershell
uv run python -m supportpilot.service.semantic_evaluation --output artifacts/reports/system/semantic_review_status.json
```

Optional `--reviews <review-array.json>` validates ClaimReview and independent fluent calibration
declarations (DATA_CONTRACTS §19). The owner verifies human provenance. Never invent reviewers.

The million workload uses the existing scale engine and isolated collection; do not replace its
1024-dimensional vectors with a cheaper semantic stand-in. Original and technical-continuation
attempts remain separate. A research deadline override changes this driver only, not base.yaml.

```powershell
uv run python -m supportpilot.service.system_stress --count 1000
uv run python -m supportpilot.service.system_stress
uv run python -m supportpilot.service.system_stress --resume <failed-receipt.json> --timeout 120
```

Continuation requires the same complete source/upload/index before incremental mutations,
matching collection count and distributed payload/vector samples. It preserves query-vector
bytes and original timings as inherited, avoids upload/build replay, and retains the negative
attempt. It is not a new favorable benchmark. `stress_latest.json` identifies the latest wrapper;
actual scale-engine receipts remain under `artifacts/reports/retrieval_scale/load/`.

Additional measured diagnostics and final analysis:

```powershell
uv run python -m supportpilot.service.retrieval_assessment
uv run python -m supportpilot.service.semantic_overlap
uv run python -m supportpilot.service.system_probe
uv run python -m supportpilot.service.system_report
```

The locator supplement is once-only and rejects an existing attempt marker. Embedding overlap
does not exclude inspected rows; changed diagnostic references are read from the starting Git
blob only when its SHA256 matches the lexical freeze. The actual-generation wrapper puts its
isolated demo outputs under `system/live_generation/`, preserving Stage 8's original latest pointer.
Final analysis imports terminal execution receipts and the retained preflight failure with
execution provenance distinct from the analysis/import run. It never promotes JSON/locator checks
to semantic acceptance. Preserve all source-version archives and failed/partial attempts.

Full-scope continuation analysis and declared guarded capacity:

```powershell
uv run python -m supportpilot.service.quality_evaluation analyse
uv run python -m supportpilot.service.workload --smoke
uv run python -m supportpilot.service.workload
```

`analyse` reads terminal sealed confirmation and writes a separate derived analysis; it never
rewrites raw observations or refreshes confirmation. Corrected action scoring and uncertainty
limits are defined in EVALUATION. A new source increment is not the archived confirmation
runtime: retain its exact source archive and checksums. Do not reseal consumed confirmation.
The workload uses configs/system_workload.yaml, isolated synthetic SQLite/credentials and a
new attempt directory, retains failures and stops its own HTTP server/restart child. Its smoke
is one HTTP request. Do not run other model jobs concurrently. No capacity/config change is
selected by this command. A real qualified verifier requires the trusted offline factory and
verified owner packet in DATA_CONTRACTS; no default implementation or enable flag is supplied.

The local-only increment's historical sequence is below. Both confirmation variants and full
support diagnostic markers are consumed: do not rerun or reseal them. Use the exact archived
executed source and frozen inputs when inspecting an old attempt, not the current runtime.

```powershell
uv run python -m supportpilot.service.local_evaluation freeze
uv run python -m supportpilot.service.local_evaluation audit
uv run python -m supportpilot.service.local_evaluation evaluate --phase development --variant control
uv run python -m supportpilot.service.local_evaluation evaluate --phase development --variant balanced
uv run python -m supportpilot.service.local_evaluation select --control <control-report> --balanced <balanced-report>
uv run python -m supportpilot.service.local_evaluation evaluate --phase confirmation --variant control
uv run python -m supportpilot.service.local_evaluation evaluate --phase confirmation --variant balanced
uv run python -m supportpilot.service.local_evaluation report
uv run python -m supportpilot.service.local_support --smoke
uv run python -m supportpilot.service.local_support
uv run python -m supportpilot.service.local_support --format enum --smoke
uv run python -m supportpilot.service.local_support --format enum --enforce-enum --smoke
uv run python -m supportpilot.service.local_support --format enum --enforce-enum
```

`system/qualification/` retains train-only head provenance, overlap, terminal reports, negative
attempts, checks and the owner acceptance packet. Grammar enforcement requires the existing
constrained decoder; it never repairs raw strings. No local diagnostic installs a qualified
verifier or changes the selected service head. See the results review and local delivery receipt
before considering owner acceptance; the original quality gates remain separately recorded.
