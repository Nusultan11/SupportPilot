# RUNBOOK: verified operational commands

Only commands that were actually run go here, each marked with the stage that verified it.
Shell: Git Bash on Windows unless stated otherwise. Run everything from the repo root.

## 1. Environment check (verified Stage 1)

```bash
uv run supportpilot env --save      # JSON report; --save writes artifacts/reports/env/env-<ts>.json
uv run supportpilot env --no-benchmark   # skip the 2048² fp16 matmul GPU test
```

Exit code 1 means `problems` is non-empty (e.g. CUDA unavailable). Reference result from
2026-11-11, saved as `artifacts/reports/env/env-20261111T113143Z.json`:

| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5800HS, 8 cores / 16 threads |
| RAM | 15.41 GiB total, ~6.4 GiB available with a normal desktop load |
| GPU | RTX 3050 Laptop, compute 8.6, 4.0 GiB VRAM (3.23 GiB free), bf16 supported |
| Driver / CUDA | NVIDIA 566.07; torch 2.14.1+cu126, CUDA 12.6, cuDNN 9.10.2 |
| GPU smoke test | fp16 matmul correct, 7.64 TFLOPS |
| Disk | 728 GiB free on C: |
| Python | 3.12.3 (venv by uv 0.12.1). No CUDA toolkit (`nvcc`) installed |

## 2. Project setup (verified Stage 1, clean clone)

```bash
git clone https://github.com/Nusultan11/SupportPilot.git && cd SupportPilot
uv sync --locked              # dev + gpu groups by default; torch comes from the cu126 index
uv run pre-commit install     # once per clone
```

- `uv sync --locked` took 20 s on a clean clone with a warm uv cache. A cold cache downloads
  about 3 GB (mostly torch and CUDA libraries). The `.venv` is about 4.6 GB.
- Without a GPU, or for CI: `uv sync --locked --no-group gpu`. **Not verified yet** (planned
  for CI in Stage 10). The code path that reports "torch not installed" exists but has not
  been run.
- Configuration: `configs/base.yaml`. Override with env vars `SUPPORTPILOT_<SECTION>__<KEY>`,
  or a file via `--config` / `SUPPORTPILOT_CONFIG`. Secrets go in `.env` (git-ignored).
- Agents: disable the local MCP servers once per clone (§4a step 1).

## 3. Quality checks (verified Stage 6)

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict; src + tests
uv run pytest -q              # Stage 6: 367 passed, 0 skipped, 114 s; GPU + Qdrant up
uv run pytest -q -m "not slow and not gpu and not qdrant_server"   # fast subset
uv run pre-commit run --all-files                                  # all hooks
```

Markers:
- `slow`: MLflow and GPU tests.
- `gpu`: needs CUDA.
- `qdrant_server`: needs a server at `QDRANT_URL` (default `http://127.0.0.1:6333`). It is
  skipped, not failed, when no server is reachable, so read the skip summary.

The pre-commit hook runs the fast subset.

## 4. Services (verified Stage 1)

**Qdrant** (ADR-014). Requires Docker Desktop to be running:
```bash
docker compose up -d qdrant                       # qdrant/qdrant:v1.20.0, named volume
docker inspect -f '{{.State.Health.Status}}' supportpilot-qdrant   # -> healthy
uv run supportpilot check qdrant                  # server-mode round trip, exit 0 = ok
uv run supportpilot check qdrant --mode memory    # no Docker needed
docker compose down                               # stop; data stays in volume supportpilot_qdrant_storage
```

Always use `127.0.0.1`, not `localhost`. On this machine `localhost` added about 2 s to every
call because of the IPv6-first lookup.

**MLflow** (ADR-011):
```bash
uv run supportpilot check mlflow       # logs a run and reads it back (params, metric history, tags)
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5055   # browse runs
```

Experiments are named `supportpilot/<name>`. The database is `mlruns/mlflow.db` and
artifacts are in `mlruns/artifacts/`. Both are git-ignored, so back them up with §6.

## 4a. MCP servers for agents (verified 2026-11-12, ADR-012)

Dev tools only, not SupportPilot runtime. Usage rules: AGENTS.md §12. Client: Claude Code
(2.1.321, the Claude desktop app's Code tab). It reads `.mcp.json` at session start and
loads tool schemas lazily (tool search), so an idle server costs no schema tokens.

| Server | Transport (no secrets in Git) | Default | Auth |
|---|---|---|---|
| `context7` | HTTP `https://mcp.context7.com/mcp` | on | none (anonymous rate limit) |
| `exa` | HTTP `https://mcp.exa.ai/mcp?tools=web_search_exa,web_fetch_exa` | on | none (anonymous rate limit, direct IPv4) |
| `github` | HTTP `https://api.githubcopilot.com/mcp/`, `X-MCP-Readonly: true`, toolsets `context,repos,issues,pull_requests,actions` (29 read tools) | on | `GITHUB_MCP_PAT` user env var |
| `huggingface` | HTTP `https://huggingface.co/mcp` | on | none |
| `serena` | stdio `uvx -p 3.13 --from serena-agent==2.1.0 serena start-mcp-server --context claude-code --project-from-cwd`, dashboard off, telemetry off; project config `.serena/project.yml` (tool interface, no onboarding, no memories) | **off** | none |
| `playwright` | stdio `npx -y @playwright/mcp@0.0.84 --browser msedge --headless --isolated --output-dir artifacts/playwright` | **off** | none |
| `mlflow` | stdio `uv run --locked mlflow mcp run` (read tools only) | **off** | none |

`.claude/settings.json` denies the MLflow write tools and Playwright `browser_run_code_unsafe`.

**Per-clone setup (once).**
1. Open a session in the repo. In `/mcp`, select `serena`, `playwright` and `mlflow` and
   choose *Disable*. Claude Code stores this per project in `~/.claude.json`
   (`projects.<path>.disabledMcpServers`). Until then the three local servers start with
   every session.
2. Optional, for `github`: create a fine-grained PAT (repository `Nusultan11/SupportPilot`,
   or public repos only; permissions read-only: Metadata, Contents, Issues, Pull requests,
   Actions), then set it as a user env var and restart the app:
   ```powershell
   [Environment]::SetEnvironmentVariable("GITHUB_MCP_PAT", "<token>", "User")
   ```
   Without it `github` fails with HTTP 400/401 and the other servers are unaffected.
3. Optional keys (higher rate limits) go into local scope, never `.mcp.json`, e.g.
   `claude mcp add --scope local --transport http context7 https://mcp.context7.com/mcp --header "Authorization: Bearer <key>"`
   (a local entry with the same name overrides the project one). An empty `Bearer ` header
   breaks the connection, so the project config carries no optional key.

**On demand.** Enable a local server in `/mcp` for the task that needs it; per the Claude
Code docs it connects in the running session (not exercised here). Disable it in `/mcp` when done. Ending the session also ends the
server's process tree. No server is needed by `uv run pytest`, the CLI or any pipeline.

**Check without an agent** (`claude` = the CLI; the desktop copy lives under
`%LOCALAPPDATA%\Claude-Data\...\claude-code\<version>\`). A plain `claude mcp list` shows
"Pending approval" because the desktop app approves project servers itself:
```bash
claude --settings '{"enableAllProjectMcpServers":true}' mcp list   # health-checks enabled servers
claude mcp get github                                              # headers and status
```
Expected: `context7`, `exa`, `huggingface` "Connected"; `github` "Connected" when
`GITHUB_MCP_PAT` is set; `serena`, `playwright`, `mlflow` "Disabled for this project", not started.

**Diagnostics.**
| Symptom | Fix |
|---|---|
| `github`: "Authorization header is badly formatted" / 401, warning "Missing environment variables: GITHUB_MCP_PAT" | Set the PAT (step 2) and restart the app |
| `exa`: 403 | Keyless access needs a direct IPv4 connection; use a key in local scope |
| `serena` times out on first start | It installs Pyright via uv on first use; start it once in a terminal or set `MCP_TIMEOUT=60000` |
| `serena` offers onboarding or memories | Check `added_modes` in `.serena/project.yml`; project facts belong in docs, not `.serena/memories` |
| `playwright`: browser not found | It uses the installed Edge (`--browser msedge`); no `playwright install` needed |
| A local server keeps running | Disable it in `/mcp` or end the session; `Get-CimInstance Win32_Process` filtered on `serena\|pyright\|playwright` must be empty |

How it was verified (2026-11-12): Claude Code `mcp list` connected `context7`, `exa`,
`huggingface`, `mlflow` and (token in the process env only) `github`; it reported the
disabled servers without starting them. A fastmcp client started each server exactly as
declared and ran one safe call: `context7` `resolve-library-id` (polars), `exa`
`web_search_exa` (Qdrant hybrid search docs), `github` `list_commits` on this repo,
`serena` `find_referencing_symbols` for `find_project_root`, `playwright` `browser_navigate`
to an inline `data:` page. After each stdio probe no `serena`, `pyright` or `playwright`
process was left.

## 5. Code backup (git bundle): verified Stage 0, new location verified Stage 2

Location: `backups/code/` in the project root (git-ignored, ADR-013).

Create:
```bash
BK=/c/Users/nurs/SupportPilot/backups/code; mkdir -p "$BK"
TS=$(date -u +%Y%m%dT%H%M%SZ); F="$BK/supportpilot-$TS.bundle"
git -C /c/Users/nurs/SupportPilot bundle create "$F" --all
git bundle verify "$F" && (cd "$BK" && sha256sum "$(basename "$F")" > "$(basename "$F").sha256")
```

Restore into a new directory, never over the live repo:
```bash
cd /c/Users/nurs/SupportPilot/backups/code && sha256sum -c supportpilot-<TS>.bundle.sha256
git clone -b main supportpilot-<TS>.bundle /c/Users/nurs/SupportPilot-restore
```

Retention: keep the 10 newest bundles plus the stage-end bundles listed in PROGRESS.md, and
delete others by hand. `main` is also pushed to GitHub, which is an off-machine copy.

## 6. Data and model snapshots (verified Stage 1)

Root: `backup.root` = `backups` (= `<project>/backups/`). A root inside the repo is refused
unless git confirms it is ignored, so snapshots can never be committed to the public remote.
Layout: `<root>/<data|model>/<name>/<snapshot_id>/{archive.tar, manifest.json}`.

```bash
uv run supportpilot backup create data/raw/bitext --name bitext_raw --kind data [--note "..."] [--pin]
uv run supportpilot backup list [--kind data] [--name bitext_raw]
uv run supportpilot backup verify data/bitext_raw/<snapshot_id>          # deep: every file hash
uv run supportpilot backup verify data/bitext_raw/<snapshot_id> --quick  # archive hash only
uv run supportpilot backup restore data/bitext_raw/<snapshot_id> --target <new-or-empty-dir>
uv run supportpilot backup pin data/bitext_raw/<snapshot_id> [--unpin]  # stage-end snapshots
uv run supportpilot backup prune [--keep 3]        # dry run: lists what would be deleted
uv run supportpilot backup prune --keep 3 --yes    # deletes; pinned snapshots always stay
```

Behaviour:
- Identical content to the newest snapshot of the same name is reused, not duplicated.
- Restore refuses a non-empty target. Files are verified against the manifest before the
  directory appears.

Measured on 1 GiB, including ~0.8 s CLI start:

| Operation | Time |
|---|---|
| create | 3.8 s |
| verify (deep) | 2.6 s |
| verify (quick) | 1.6 s |
| restore | 3.8 s, byte-identical |

## 7. Source data (verified Stage 2)

The registry is `configs/sources.yaml` (DATA_CONTRACTS §2). Data lives in `data/` and is git-ignored.

```bash
uv run supportpilot data sources [--status selected]   # registry + acquired/converted state
uv run supportpilot data acquire --all                 # all status=selected; or name ids
uv run supportpilot data convert --all [--force]       # raw -> data/interim/records
uv run supportpilot data verify --all                  # deep sha256 check of raw + records; exit 1 on mismatch
```

Behaviour:
- **Re-runs:** acquisition skips verified revisions without any network call. Conversion skips
  outputs that are still current. `--force` rebuilds and produced byte-identical Parquet in the test.
- **Interrupted downloads:** run the same command again. Files already complete in
  `data/raw/<id>/.partial-<rev>/` are re-verified and kept, and incomplete ones resume.
  Verified with a real `kill -9` during a 70 MB download.
- **Corrupt raw revision:** `acquire` and `convert` stop with "immutable". Restore that source
  from its pinned snapshot (below). Never edit files under `data/raw`.
- **Gated sources** (`kazparc`, `flores_plus`, `kazqad`):
  1. Accept the terms on huggingface.co.
  2. Put `HF_TOKEN=...` in `.env` or the environment.
  3. Set the entry's `status: selected`.
  4. Add a converter for it.

  The token is never written to the registry or the logs. This path has not been tried (no account).

Timings (2026-11-11): first acquisition of the 5 sources took 31 s (151.5 MB). Conversion of
99,882 records took 9 s.

Backup after a new acquisition, one pinned snapshot per source:
```bash
uv run supportpilot backup create data/raw/<source_id> --name <source_id>_raw --kind data --pin
uv run supportpilot backup create data/interim/records --name raw_records --kind data
```

Restore a raw source:
```bash
uv run supportpilot backup restore data/<source_id>_raw/<snapshot_id> --target <empty-dir>
# then move <empty-dir>/<revision> to data/raw/<source_id>/<revision> and run `data verify`
```

## 7b. Stage 3: cleaning, dedup, gates, EDA (verified Stage 3)

```bash
uv run supportpilot data clean --all          # clean every selected source, build data/interim/canonical, run gates
uv run supportpilot data clean --all --force  # rebuild everything (same input -> byte-identical Parquet)
uv run supportpilot data clean bitext_retail_ecom   # only the per-source clean checkpoint, no build
uv run supportpilot data gates                # re-run the quality gates on the current build; exit 1 on failure
uv run supportpilot data eda                  # profile + report -> artifacts/reports/eda/ (downloads 2 Qwen tokenizer.json once)
uv run supportpilot data eda --no-tokens      # offline: skip token counts
```

Outputs and their meaning are in DATA_CONTRACTS §4–4d. The configuration lives in
`configs/base.yaml`:
- `clean:` has the thresholds (ADR-020/022);
- `eda:` has the review sample sizes and the pinned tokenizers;
- `configs/intent_map.yaml` is the draft v0 label map;
- `configs/placeholders.yaml` is the synthetic slot catalogue.

Behaviour:
- **Re-runs.** A step is skipped when its inputs, parameters and file hashes are unchanged:
  clean per source (input records hash + params), build (all clean hashes + params,
  including the intent-map sha). After a change in `clean.py` / `build.py` logic, bump
  `CLEAN_VERSION` / `BUILD_VERSION`, or run with `--force`.
- **Interruption.** Outputs are written to `.partial-*` and swapped in at the end.
  - Verified with `taskkill /F` twice: once during grouping and once while the partial
    canonical directory was being written.
  - Both times the previous build stayed intact, and `data verify --all` and `data gates`
    passed. A plain re-run skipped everything in 2.8 s.
  - Any leftover partial is removed on the next run.
- **Failed gates.** The new build is not swapped in. It is kept at
  `data/interim/canonical.failed/` (see its `quality_report.json`), the previous build stays
  in place, and the exit code is 1.
- **RAM.** The full forced build peaked at 2.84 GiB RSS. `data clean --all` warns when less
  than 3 GiB is free. Stop Docker/IDE first if RAM is low.

Timings (2026-11-11, full `--force`, 99,882 records):

| Step | Time |
|---|---|
| clean bitext_retail / bitext_cs / while-ai / support_json_ru / belebele | 23 / 11 / 78 / 0.7 / 2 s |
| build (dedup, 7.6M LSH candidates, groups, gates) | ~104 s |
| eda (incl. two tokenizers over all texts) | ~63 s |

Qualitative review:
- `data eda` writes deterministic samples to `artifacts/reports/eda/review/<source>.md`
  (100 random + up to 100 flagged, seed 20261111).
- The hand-written annotations live in `docs/reviews/stage3_review.yaml` and are rendered
  into the report.
- After a change that alters the samples (`review/samples.json`), re-check the annotations;
  the report shows them by sample id.

Backup after a rebuild:
```bash
uv run supportpilot backup create data/interim/clean --name clean_records --kind data
uv run supportpilot backup create data/interim/canonical --name canonical --kind data --pin --note "stage end"
uv run supportpilot backup create artifacts/reports/eda --name eda_report --kind data --pin
```

## 7c. Stage 4: multilingual SFT dataset and gold benchmark (verified Stage 4)

Inputs: the Stage 3 build (`data/interim/canonical/`, read-only), `configs/taxonomy.yaml`
(taxonomy v1), `configs/locale_kz.yaml`, `authored/` (scenarios, answer templates, training KB,
gold sources) and `kb_sources/demo_store/facts.yaml`. Settings: `configs/base.yaml` `stage4:`.

```bash
uv run supportpilot data taxonomy             # validate taxonomy v1, show how labels were split
uv run supportpilot data select               # v1 relabel + EN backbone + MT queue
uv run supportpilot data kb-train             # training KB variant (6 stores x 3 languages)
uv run supportpilot data mt-compare           # MT systems on a 100-item sample -> artifacts/reports/stage4/mt_compare/ (GPU)
uv run supportpilot data translate --lang kk  # MADLAD-400 en->kk + kk->en back-translation (GPU, resumable)
uv run supportpilot data translate --lang ru  # en->ru + Opus-MT ru->en back-translation
uv run supportpilot data qe-score             # QE + glossary -> data/interim/stage4/translated/
uv run supportpilot data sft                  # select/kb/qe if stale, then SFT -> data/processed/sft/<version>/
uv run supportpilot data gold                 # gold benchmark -> data/gold/<version>/ (needs the SFT build)
```

Behaviour:
- **Models.** `translate`, `mt-compare` and the semantic leakage check download pinned models
  into the HF cache on first use (MADLAD-400-3B 11.8 GB, Opus-MT 0.3 GB each, M2M100 1.9 GB,
  multilingual-e5-small 0.5 GB). Afterwards set `HF_HUB_OFFLINE=1`. `--no-semantic` on `sft` /
  `gold` skips e5 (the lexical and id checks still run).
- **GPU time.** MADLAD in nf4 uses about 3.4 GB VRAM and translates about 2.5 short texts/s on
  the RTX 3050 (batch 8, 2 beams). The 7,554-row queue per language takes about 50 min forward
  plus the back-translation. Close other GPU programs; `nvidia-smi` should show < 300 MiB used.
- **Interruption.** Translations are cached per 512-row shard
  (`data/interim/stage4/mt/<system>/<pair>/`). A killed run loses at most one shard; re-run the
  same command. A finished run re-translates nothing.
- **Re-runs.** `select`, `kb-train`, `qe-score`, `sft` and `gold` skip when inputs, parameters
  and file hashes are unchanged; `--force` rebuilds (`sft` reproduces the same
  `dataset_version` for the same inputs). Outputs are written to `.partial-*` and renamed;
  leftovers are removed by the next run.
- **Failed gates.** The SFT or gold build is kept at `<root>/current.failed/` with its
  `quality_report.json`, and the exit code is 1. A gold leakage hit lists the gold ids
  (`leakage_items`): preserve `authored/gold/` and diagnose copied SFT inputs. Rebuild SFT
  with the ADR-029 copy-exclusion policy, then rebuild gold; do not rewrite evaluation
  questions to accommodate training data.
- **Long runs that must survive a closed terminal or agent session:** start them detached, e.g.
  in PowerShell `Start-Process cmd -ArgumentList '/c uv run supportpilot data translate --lang kk > mt.log 2>&1' -WindowStyle Hidden`.

Backup after a stage build:
```bash
uv run supportpilot backup create data/interim/stage4 --name stage4_interim --kind data --pin
uv run supportpilot backup create data/processed/kb_train --name kb_train --kind data --pin
uv run supportpilot backup create data/processed/sft/<dataset_version> --name sft --kind data --pin
uv run supportpilot backup create data/gold/<benchmark_version> --name gold --kind data --pin
```

## 7d. Stage 5: demo-store knowledge base (verified Stage 5)

Inputs: `kb_sources/demo_store/` (`facts.yaml`, `catalog.yaml`, `docs/<doc_id>.yaml`),
`configs/locale_kz.yaml` (city / carrier / payment names), `authored/gold/` (evidence gate).
Settings: `configs/base.yaml` `kb:`. Output: `data/kb/demo_store/` (ADR-027, DATA_CONTRACTS §8).

```bash
uv run supportpilot kb render                  # specs + facts -> data/kb/demo_store/sources/ (30 files)
uv run supportpilot kb build                   # render, then extract, chunk, gate, activate a new version
uv run supportpilot kb build --force           # re-parse every document (same content -> "unchanged")
uv run supportpilot kb validate                # re-run all 13 gates on the current version (exit 1 on failure)
uv run supportpilot kb validate --version <kb_version>
uv run supportpilot kb status                  # current version, stored versions, sources_match_current
uv run supportpilot kb activate <kb_version>   # switch the current version (rollback)
```

Behaviour:
- **Updating the KB.** Edit a value in `facts.yaml` / `catalog.yaml` or a text in
  `docs/<doc_id>.yaml`, then `kb build`. Only the files whose bytes change are re-rendered and
  re-parsed; `versions/<new>/changes.json` lists the documents changed/added/removed and the
  chunk ids to delete and upsert (the Stage 6 reindex input). Deleting a spec removes its
  documents from the next version. Quote YAML values that contain commas (the loader rejects
  split values).
- **Identical content** (same chunks, same ground truth) creates no version: the build reports
  `unchanged`, or `activated_existing` when it matches an older stored version (a reverted
  edit). The tokenizer is the pinned Qwen3 `tokenizer.json` from the HF cache
  (`HF_HUB_OFFLINE=1` once cached).
- **Failed gates.** The version is kept as `versions/<kb_version>.failed/` with its
  `quality_report.json`, `current.json` is not changed, and the exit code is 1. A source file
  edited after rendering is refused (`changed after rendering`): run `kb render` again.
  Extraction errors (scanned PDF page, missing header row, no effective date) exit 2.
- **Timings (laptop):** render ≈ 2.4 s; full build ≈ 8 s including start-up and the tokenizer;
  `--force` re-parse of 30 documents 6.4 s, peak RSS 269 MiB; 2.4 MB on disk per version set.

Backup and restore drill:
```bash
uv run supportpilot backup create data/kb/demo_store --name kb_demo_store --kind data --pin
uv run supportpilot backup restore data/kb_demo_store/<snapshot_id> --target <empty-dir>/kb/demo_store
SUPPORTPILOT_PATHS__DATA_DIR=<empty-dir> uv run supportpilot kb validate   # gates on the restored copy
```
A version directory is self-contained (its own copy of the sources), so a restored version
validates without re-rendering.

## 7e. Multilingual data expansion and retrieval queries (2026-11-12)

The new sources use the same pinned acquire/convert/clean pipeline; source licence evidence
and attribution are in `configs/sources.yaml`. Existing raw data and MT caches can be reused.

```bash
uv run supportpilot data acquire mangi_kk_support_qa massive_en_ru
uv run supportpilot data convert mangi_kk_support_qa massive_en_ru
uv run supportpilot data clean --all
uv run supportpilot data verify --all
uv run supportpilot data native
uv run supportpilot kb build
uv run supportpilot data sft
uv run supportpilot data gold
uv run supportpilot data retrieval-set
```

`data sft` refreshes selection/native/training-KB/QE inputs, preserves assigned evaluation
coverage and excludes copied training inputs, including copies of gold questions. It does
not silently fill a missing MT cache: resume `data translate --lang ru|kk` as in §7c.
`--no-semantic` is available for offline diagnostics; it leaves the semantic check
unverified and is not the evidence for the completed data build.

Retrieval output: `data/benchmarks/retrieval/current.json` identifies the immutable directory
containing `queries.jsonl`, `qrels.tsv`, `stats.json`, `quality_report.json`, `manifest.json`.
Each query states its corpus language and KB version. Evaluate with that KB, not with
arbitrary latest chunks. After KB/SFT changes, rebuild: stale section references, corrupt
inputs or prohibited overlaps fail without changing the current pointer. No encoder/index
or RAG experiment is run by this command. Evaluation details: EVALUATION §2.

Snapshot the significant outputs after completion, using §6 verification and restore:

```bash
uv run supportpilot backup create data/interim/stage4 --name stage4_interim --kind data --pin
uv run supportpilot backup create data/processed/kb_train --name kb_train --kind data --pin
uv run supportpilot backup create data/processed/sft/<dataset_version> --name sft --kind data --pin
uv run supportpilot backup create data/kb/demo_store --name kb_demo_store --kind data --pin
uv run supportpilot backup create data/benchmarks/retrieval --name retrieval --kind data --pin
uv run supportpilot backup create data/gold/<benchmark_version> --name gold --kind data --pin
```

Processed datasets and backups stay git-ignored. Authored YAML sources are versioned in Git;
do not publish processed datasets under an invented project-wide licence. Review status is
`draft` for gold/retrieval; fluent human review remains pending. Input-language sample
evidence and the distinction from historical MT error estimates are in
`docs/reviews/data_expansion_review.yaml`.

## 8. Diagnostics

| Symptom | Check |
|---|---|
| `torch.cuda.is_available()` False | Does `nvidia-smi` work? Did `uv sync` install `+cu126` torch (`uv pip list \| grep torch`)? Was `--no-group gpu` used? |
| CUDA OOM on 4 GB | Batch size 1, gradient checkpointing, shorter `max_seq_length`, 4-bit; close GPU apps (`nvidia-smi`). The desktop already uses ~0.8 GiB |
| Docker "cannot connect to npipe" | Start Docker Desktop (`"C:\Program Files\Docker\Docker\Docker Desktop.exe"`), then wait for `docker info` |
| Qdrant calls take ~2 s each | URL uses `localhost`. Switch to `http://127.0.0.1:6333` |
| Embedded Qdrant dir cannot be deleted | Windows keeps `storage.sqlite` locked while the process lives. Exit the process first |
| `config error: ...` from the CLI | Unknown key or bad value in YAML/env. `extra="forbid"` names the field |
| pre-commit "mixed line ending" | A tool wrote CRLF. The hook fixes it, so re-stage and commit again |
| `data acquire`: "is gated" | Needs `HF_TOKEN` from an account that accepted the dataset's terms |
| `data acquire`: "fails verification ... immutable" | A raw file changed on disk. Restore from the pinned snapshot (§7) |
| `data convert`: "rows rejected (> 1.00%)" | Upstream format changed. Inspect `rejected` reasons, then fix the converter and bump its version |
| Console shows `?` instead of Kazakh letters | Windows code page. Set `PYTHONIOENCODING=utf-8` for ad-hoc scripts |
| `data clean --all`: "quality gates failed" | Open `data/interim/canonical.failed/quality_report.json`, fix the cause, re-run. The previous build is untouched |
| `data clean`: "not cleaned yet" / "fails verification" | Run `data clean --all` (rebuilds checkpoints); for records problems run `data verify --all` and re-convert |
| `data clean` re-runs although nothing changed | `configs/placeholders.yaml`, `intent_map.yaml`, `clean:` params or the lingua version changed. This is expected |
| `data eda`: tokenizer download fails or sha256 mismatch | Offline or upstream changed: use `--no-tokens`, or re-pin `eda.tokenizers` after checking the new revision |
| HF warning "cache-system uses symlinks" | Harmless on Windows. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1` |
| `data/interim/.partial-*` directories exist | A run was killed. The next run of the same step removes them |
| `data translate` very slow, `nvidia-smi` near 4 GB | VRAM spills into shared memory. Close GPU apps; keep MADLAD `batch_size` at 8 in `stage4.mt_systems` |
| `data sft`: "translations missing" | Run `data translate --lang <ru|kk>` until it reports `translated: 0` |
| `data gold` fails `leakage` | Keep gold unchanged. Rebuild SFT with copy exclusion; diagnose remaining SFT inputs using `current.failed/quality_report.json` |

## 9. Retrieval and experiments (verified Stage 6, ADR-030)

Start Docker Desktop, wait for `docker info`, then `docker compose up -d qdrant`.
`docker desktop start --detach` was also verified on this Windows installation.
The selected defaults are in `configs/base.yaml`: BGE-M3, word sparse, weighted .75,
20 candidates/channel, BGE rerank 20, output up to 10 and context 1024 Qwen tokens.
Model revisions/licences and pooling/prefixes are pinned in `configs/retrieval_models.yaml`.

Commands below were run in PowerShell (UTF-8 is needed for redirected KK text):

```powershell
$env:PYTHONUTF8='1'
uv sync --locked
uv run --locked supportpilot retrieval index
uv run --locked supportpilot retrieval query 'Алматыға жеткізу қанша тұрады?' --lang ru --budget 1024 --as-of 2026-11-12
uv run --locked supportpilot retrieval query 'Доставка' --lang kk --doc-id missing-document --as-of 2026-11-12
```

`--lang` filters the **corpus**, independently of query language (the first example is
KK input / RU evidence). Omit it to search all KB languages. `--doc-id`/`--doc-type`
are repeatable. `--as-of` is inclusive; serving defaults to today's date, experiments
pin `retrieval.evaluation_date`. CLI returns JSON evidence with original locators,
scores, KB/index versions, token budget and `abstain/reason`. Empty document filter
returned zero tokens/evidence and `no_retrieval_evidence`. `--budget 0` rejects all
evidence; fragments are never cut to fit. Retrieval does not generate store answers.

Library (the equivalent persistent-model API was used for all 681 online requests):

```python
from supportpilot.common.config import load_settings
from supportpilot.retrieval import SearchFilters, open_retriever

with open_retriever(load_settings()) as retrieval:
    context = retrieval.context(
        "Алматыға жеткізу қанша тұрады?",
        filters=SearchFilters(corpus_lang="ru"),
        budget=1024,
    )
```

Resume/index versions:

- Re-run `retrieval index` after interruption. Checksummed embeddings are committed in
  batches; Qdrant inventory supplies the remaining upserts/deletes. OS locks release on
  process death. Do not remove persistent `.lock` files or SQLite caches to recover.
- KB activation and retrieval synchronisation are separate. After `kb activate <version>`,
  run `retrieval index` before serving; queries fail closed until they agree. To roll back,
  activate the previous verified KB and run the same index command. Cached immutable
  snapshots avoid re-encoding unchanged text. The activate command itself is Stage 5;
  Stage 6 tested real parent/current index transitions without changing the active KB pointer.
- `retrieval index --kb-version <version>` materialises an explicit version; it does not
  activate the KB. Queries still require agreement with active KB. Never serve a mismatched
  collection or clear the live collection as a recovery shortcut.
- A matching parent validates `changes.json`; other transitions use exact inventories.
  Selected forward update: 90 deletes/171 upserts/198 vectors retained; rerun 0/0.
  Global sparse IDF is unsuitable after these transitions: preserve the implemented
  KB-scoped IDF query settings (requires the pinned Qdrant 1.20 feature).

Experiment stages were verified individually; all reuse existing checkpoints where valid:

```powershell
# After a successful first download; omit this on a cold cache:
$env:HF_HUB_OFFLINE='1'
uv run --locked supportpilot retrieval experiment --stage prepare
uv run --locked supportpilot retrieval experiment --stage compare
uv run --locked supportpilot retrieval experiment --stage batch
uv run --locked supportpilot retrieval experiment --stage rerank
uv run --locked supportpilot retrieval experiment --stage report
uv run --locked supportpilot retrieval experiment --stage profile
uv run --locked supportpilot retrieval experiment --stage drill
```

Cold execution downloads pinned public weights; it does not need remote inference or
paid API credentials. The gated Gemma candidate was only screened, not downloaded.
HF cache completeness is checked against actual weight files (BGE-M3 uses `.bin` at its
pinned revision); a config-only snapshot is not a usable offline model. Clear the
offline env flag for a missing download. Model changes produce a separate recipe/cache.

Reranking all 13,620 pairs is substantially slower with Qwen; interrupted committed
batches resume. Qwen batch 4 is retained: measured 4/8/16 pair throughput 4.87/4.88/4.41 s^-1.
Do not run concurrent model jobs during measurement. `profile` runs all 681 queries
through the public API without query/pair caches and measures the packed context.
All results remain conditional on draft qrels. A confidence-rule amendment followed
an initial holdout inspection; later holdout results are exploratory. No gold tuning.

Read `artifacts/reports/retrieval/comparison.md`, `summary.json`, per-configuration
`metrics.json`/`observations.json`, `online_context_metrics.json`, `performance.json`,
`context_audit.json`, `indexing_upload.json`, `error_analysis.md`, `mlflow_audit.json`.
The scripts `error_analysis.py`, `context_audit.py`, `measure_index_upload.py` in that
backed-up report directory reproduce the detailed diagnostic census and isolated upload
measurement; they are analysis artifacts, not runtime dependencies. MLflow experiment
`supportpilot/retrieval` contains 96 audited current comparison runs and nine retained
failed attempts, as well as preparation/probe/profile runs and superseded measurements.

Measured laptop profile: warm p50/p95 297/841 ms, 2.18 sequential requests/s; cold first
request 16.93 s. Python peak RSS 4.80 GiB, PyTorch allocated/reserved VRAM 2.145/2.174 GiB.
These exclude Docker/desktop memory. Document encoding 40.37 units/s; portable selected
index files 2,123,262 bytes, snapshot 7,687,680 bytes. Larger/multi-user workloads and
simultaneous LLM residency are unverified. Earlier short samples were faster; use the
681-query result for planning. Keep models alive between requests; CLI startup is cold.

**Evidence limit:** score gating is disabled by default because all tested positive
thresholds violate the cross-language recall guard. `evidence_threshold` optionally
rejects a whole query by maximum reranker score; `min_rerank_score` instead prunes
individual chunks. Do not transfer either threshold to a different model. When disabling
the reranker, also disable its thresholds. Default top-k returns candidates for all 48
unanswerable benchmark queries; it cannot prove a missing policy or live order fact.
Future routing/claim grounding is required before customer-facing answers. `draft`
judgements and a six-query RU–KK slice do not establish production readiness.

Back up `data/indexes/`, `artifacts/reports/retrieval/` and `mlruns/` with §6 after all
writers finish. Deep-verify each pinned snapshot. `experiment --stage drill` verifies
the saved snapshot hash, uploads the local file into a separate collection and compares
all payloads/named vectors plus six dense/sparse queries. Reruns verify the receipt;
they do not reset the serving collection. Restoring the backup archive to a new empty
directory also verifies files before publication. Stage-end IDs and code bundles are
in PROGRESS.md. Keep earlier failed/superseded artifacts and never prune pinned backups.

Stage-end restore receipts: 84 index/cache files (10 immutable indexes, six SQLite
integrity checks), 557 report files (556 manifest members), 1,926 MLflow files (SQLite
integrity plus FINISHED delivery run). Restores live under `artifacts/restore_drills/`;
snapshot IDs and the separate code bundle are in PROGRESS.md. The frozen report's
`manifest.json` is a standard `DatasetManifest` with `retrieval_report/1`; delivery
source fingerprints refer to implementation commit `326d223`. Publication receipts
are outside that frozen directory at `artifacts/reports/retrieval_delivery_receipt.json`.
