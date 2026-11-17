# RUNBOOK — verified commands

Run from the repository root (PowerShell or Git Bash). CLI prints JSON on stdout, logs on stderr.
Exit codes: 0 ok · 1 failed check · 2 usage/operation error.

## 1. Setup

```powershell
uv sync --locked
uv run pre-commit install
uv run supportpilot env --no-benchmark
```

- `uv sync` installs `dev` and `gpu` (Torch cu126) groups.
- `supportpilot env --save` also runs a GPU matmul and saves a report to `artifacts/reports/env/`.
- Settings: `configs/base.yaml`, `--config <file>` or `SUPPORTPILOT_CONFIG`.
  Override keys with `SUPPORTPILOT_<SECTION>__<KEY>`. Secrets go in `.env`.

## 2. Checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -q
```

- Full pytest runs `gpu`/`slow` tests when the hardware exists; a skip is unverified.
- Pre-commit runs only the fast subset (`-m "not slow and not gpu"`).
- If sandboxed `uv` cannot write its cache, use `.venv/Scripts/python.exe -m pytest|mypy|ruff`.
  Never rebuild or delete `.venv` as a workaround.

## 3. MLflow

```powershell
uv run supportpilot check mlflow
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5055
```

`mlruns/mlflow.db` and `mlruns/artifacts/` are one recovery unit; snapshot them together.

## 4. Data sources

```powershell
uv run supportpilot data sources [--status selected]
uv run supportpilot data acquire <source_id ...> | --all
uv run supportpilot data verify <source_id ...> | --all
uv run supportpilot data taxonomy
```

- `acquire` downloads pinned revisions to `data/raw/` and skips verified ones. Gated sources need `HF_TOKEN`.
- `verify` re-hashes raw files against their manifest.
- A new source: add it to `configs/sources.yaml` with licence evidence first.

## 5. Code backup

```powershell
$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$bundle = Join-Path (Get-Location) "backups/code/supportpilot-$stamp.bundle"
git bundle create $bundle --all
git bundle verify $bundle
Get-FileHash -LiteralPath $bundle -Algorithm SHA256
git clone -b main $bundle <new-empty-directory>
```

Record the SHA256. Check the clone's HEAD and clean status, then delete the clone.
Never restore over the live repository.

## 6. Data and model snapshots

```text
uv run supportpilot backup create <path> --name <slug> --kind data|model [--pin] --note "<purpose>"
uv run supportpilot backup list [--kind data] [--name <slug>]
uv run supportpilot backup verify <kind>/<name>/<snapshot_id> [--quick]
uv run supportpilot backup restore <kind>/<name>/<snapshot_id> --target <new-or-empty-dir>
uv run supportpilot backup pin <kind>/<name>/<snapshot_id> [--unpin]
uv run supportpilot backup prune [--keep N]
```

- Layout: `backups/<kind>/<name>/<snapshot_id>/{archive.tar, manifest.json}`.
- `verify` is deep by default; `--quick` checks archive bytes only.
- `restore` verifies every file and refuses a non-empty target.
- `prune` without `--yes` is a dry run. **Never prune, delete or unpin without owner permission.**
- Snapshot significant datasets, models, reports and MLflow; restore-test into a new directory.

## 7. Safe cleanup

1. Inventory tracked, untracked and ignored paths with sizes. Find active writers.
2. Delete only proven disposable caches and your own task's temporary work.
3. Resolve each recursive target; it must stay inside its directory and not be a link.
4. Never: `git clean -xfd`, deleting `backups/`, `.venv`, model caches or Docker volumes.
5. Report removed bytes/counts. Run the checks. Update the owning docs.

## 8. Agent tools

MCP is optional and development-only (ADR-012). `.mcp.json` holds project servers.
GitHub MCP reads `GITHUB_MCP_PAT` from the user environment. Local servers (Serena, MLflow)
stay off until the owner enables them.

Optional Shrimp + Structurizr (ADR-036), Node.js ≥ 20 and Docker Desktop:

```powershell
npm ci --prefix tools/mcp --ignore-scripts --no-fund --no-audit
node tools/mcp/configure.mjs
docker compose --profile engineering up -d structurizr
node tools/mcp/check.mjs
claude --mcp-config tools/mcp/claude.json
docker compose --profile engineering stop structurizr
```

- `configure.mjs` adds disabled entries to the ignored `.codex/config.toml`;
  enable per session with `codex -c 'mcp_servers.shrimp.enabled=true'`.
- Without tool reload: `node tools/mcp/client.mjs shrimp list_tasks <args.json> <receipt.json>`.
- Shrimp store: `artifacts/engineering/shrimp/tasks.json`. One writer only; close the client
  before snapshotting. Never reset tasks or auto-clear a lock; check the lock's PID first.
- Structurizr: validate/parse/inspect `docs/architecture/workspace.dsl` only.
- Resume: AGENTS → PROGRESS → ROADMAP stage → `list_tasks` / `get_task_detail`.
