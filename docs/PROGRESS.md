# PROGRESS: current state (single source of truth for status)

Last updated: 2026-11-11

## Current stage

**Stage 1: Architecture and engineering foundation. Status: COMPLETED.** Waiting for owner
confirmation to start Stage 2 (data acquisition and source registry).

| Stage | Status | Commits |
|---|---|---|
| 0 Audit and research | COMPLETED | `699dc23`, `cd50a6c`, `ced287e` |
| 1 Engineering foundation | COMPLETED | `6b17404` (code), plus the docs/progress commit after it |

## Stage 1: what exists now

| Component | Path | Notes |
|---|---|---|
| uv project | `pyproject.toml`, `uv.lock`, `.python-version` | Python 3.12. Groups: `dev`, `gpu` (torch 2.14.1+cu126) (ADR-015) |
| Settings | `common/config.py`, `configs/base.yaml` | defaults < YAML < .env < env < overrides; frozen; unknown keys rejected; `SecretStr` |
| Logging | `common/logging.py` | JSON/console; redacts secrets and PII; customer text hashed unless `log_text` (ADR-016) |
| IDs, hashing, IO | `common/ids.py`, `hashing.py`, `io.py`, `provenance.py` | DATA_CONTRACTS §1; streaming sha256; atomic writes; `code_commit` with `-dirty` |
| Contracts | `common/schemas.py`, `common/manifest.py` | §2 registry, §3 raw manifest, §4 canonical example, §9 dataset manifest; scan/verify |
| Tracking | `common/tracking.py` | MLflow on local SQLite; `RunSpec` enforces hypothesis, dataset_version and seed tags |
| Ops CLI | `cli.py`, `ops/{envcheck,backup,health}.py` | `supportpilot env`, `backup create/list/verify/restore/pin/prune`, `check mlflow/qdrant` |
| Services | `compose.yaml` | Qdrant v1.20.0, named volume, ports bound to 127.0.0.1 (ADR-014) |
| Quality gates | `.pre-commit-config.yaml`, `[tool.ruff/mypy/pytest]` | ruff, mypy `--strict` on src + tests, fast pytest subset, file hygiene, 5 MB limit |

Not created yet, on purpose: `supportpilot.data` and the other domain packages (from
Stage 2), Polars and PyArrow (from Stage 2–3), `configs/sources.yaml` (Stage 2).

## Verification evidence (2026-11-11)

- **Tests:** 108 passed, 0 skipped (`uv run pytest -q`, 15 s, with the Qdrant server up).
  Fast subset: 105 passed in 6 s. Coverage:
  - contracts, including invalid inputs (bad language, unknown intent, escalation without a
    reason, MT without lineage, unsafe paths, naive datetimes, duplicate sources, unknown licence);
  - config precedence and rejection of bad input; redaction;
  - the backup cycle, including corruption, tampering, non-empty targets, dedup, prune and pin;
  - CLI exit codes; MLflow; Qdrant in memory, embedded and server modes; the GPU matmul.
- **Lint and types:** `ruff check`, `ruff format --check`, `mypy --strict` (26 files) are all
  clean. `pre-commit run --all-files` passes, and the hook also ran on commit `6b17404`.
- **Clean clone:** `git clone` → `uv sync --locked` (20 s, warm uv cache) → ruff, mypy and
  108 tests passed; torch CUDA available inside the clone.
- **GPU:** `supportpilot env --save` → RTX 3050 Laptop, compute 8.6, 4.0 GiB (3.23 GiB free),
  bf16 supported, fp16 matmul correct at 7.64 TFLOPS. Report saved as
  `artifacts/reports/env/env-20261111T113143Z.json` (local). Values are in RUNBOOK §1.
- **Backup round trip with the real CLI and the real root:** snapshot
  `data/env_reports/20261111T113155Z-92ddf171` → deep verify ok → restore → `diff -r`
  identical → re-create reused the snapshot (dedup).
  - 1 GiB test in a scratch root: create 3.8 s, deep verify 2.6 s, restore 3.8 s,
    byte-identical (`cmp`).
- **MLflow:** `supportpilot check mlflow` → run written to `mlruns/mlflow.db` and read back
  (params, 3-step metric history, tags); `mlflow ui` served the experiment list (HTTP 200).
- **Qdrant:**
  - the round trip passed in all three modes;
  - benchmark at 50k × 1024-dim: server 1276 pts/s and p50 11 ms, embedded 198 pts/s and
    p50 604 ms;
  - the named volume survived `compose down/up`. Decision in ADR-014.

## Known limitations and open items

1. `uv sync --no-group gpu` (CPU-only/CI path) has not been run. It is planned for CI in Stage 10.
2. A cold-cache install downloads about 3 GB, and the venv is 4.6 GB. Only the warm-cache
   clean clone was measured.
3. RAM is the tightest resource: 15.4 GiB total, about 6.4 GiB free with the desktop plus
   Docker's VM (about 6 GB assigned). Stop Qdrant (`docker compose down`) during local GPU work.
4. Embedded Qdrant keeps its SQLite file locked on Windows until the process exits.
5. No CUDA toolkit (`nvcc`): source-built kernels still need Docker/WSL2 or the cloud (ADR-003).
6. The public repository has **no LICENSE file**. Choosing one is the owner's decision.
7. Still open from Stage 0: paid-API budget for KK/RU work, fluent KK/RU reviewers for the
   gold benchmark, Kaggle/Colab accounts from Stage 6–7.

## Git and backups

- Remote: `origin` = https://github.com/Nusultan11/SupportPilot (public). `main` is pushed after each stage.
- Code bundles in `C:\Users\nurs\SupportPilot-backups\code\`. Stage-end snapshots:
  - Stage 0: `supportpilot-20261111T104345Z.bundle`.
  - Stage 1: the newest bundle created after the Stage 1 docs commit (named in the Stage 1 report).
- Data snapshots in `C:\Users\nurs\SupportPilot-backups\data\`:
  `env_reports/20261111T113155Z-92ddf171` (the round-trip evidence).

## Notes for the next stage (Stage 2)

- Put the source registry in `configs/sources.yaml` and load it with
  `common.manifest.load_source_registry`. A source with licence `unknown` must be `blocked`
  (enforced by the schema).
- Write raw manifests with `RawManifest` + `scan_files`. Snapshot raw data after download:
  `supportpilot backup create data/raw/<source> --name <source>_raw --kind data --pin`.
- Pin HF downloads by commit sha (`revision=`). Add `huggingface_hub`/`datasets`, `polars` and
  `pyarrow` only in Stage 2, and record them in ADR-015.

## Next recommended action

Start Stage 2 (ROADMAP §Stage 2) after owner confirmation.
