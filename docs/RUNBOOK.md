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
remain separate from successful execution; evaluation scope is EVALUATION §9.
