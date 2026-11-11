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

## 3. Quality checks (verified Stage 1)

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy                   # strict; src + tests
uv run pytest -q              # full suite: 108 tests, ~15 s (with the Qdrant server up)
uv run pytest -q -m "not slow and not gpu and not qdrant_server"   # fast subset, ~6 s
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

## 5. Code backup (git bundle): verified Stage 0

Location: `C:\Users\nurs\SupportPilot-backups\code\`, outside the work tree.

Create:
```bash
BK=/c/Users/nurs/SupportPilot-backups/code; mkdir -p "$BK"
TS=$(date -u +%Y%m%dT%H%M%SZ); F="$BK/supportpilot-$TS.bundle"
git -C /c/Users/nurs/SupportPilot bundle create "$F" --all
git bundle verify "$F" && (cd "$BK" && sha256sum "$(basename "$F")" > "$(basename "$F").sha256")
```

Restore into a new directory, never over the live repo:
```bash
cd /c/Users/nurs/SupportPilot-backups/code && sha256sum -c supportpilot-<TS>.bundle.sha256
git clone -b main supportpilot-<TS>.bundle /c/Users/nurs/SupportPilot-restore
```

Retention: keep the 10 newest bundles plus the stage-end bundles listed in PROGRESS.md, and
delete others by hand. `main` is also pushed to GitHub, which is an off-machine copy.

## 6. Data and model snapshots (verified Stage 1)

Root: `backup.root` = `../SupportPilot-backups` (refused if inside the repo).
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

## 7. Diagnostics

| Symptom | Check |
|---|---|
| `torch.cuda.is_available()` False | Does `nvidia-smi` work? Did `uv sync` install `+cu126` torch (`uv pip list \| grep torch`)? Was `--no-group gpu` used? |
| CUDA OOM on 4 GB | Batch size 1, gradient checkpointing, shorter `max_seq_length`, 4-bit; close GPU apps (`nvidia-smi`). The desktop already uses ~0.8 GiB |
| Docker "cannot connect to npipe" | Start Docker Desktop (`"C:\Program Files\Docker\Docker\Docker Desktop.exe"`), then wait for `docker info` |
| Qdrant calls take ~2 s each | URL uses `localhost`. Switch to `http://127.0.0.1:6333` |
| Embedded Qdrant dir cannot be deleted | Windows keeps `storage.sqlite` locked while the process lives. Exit the process first |
| `config error: ...` from the CLI | Unknown key or bad value in YAML/env. `extra="forbid"` names the field |
| pre-commit "mixed line ending" | A tool wrote CRLF. The hook fixes it, so re-stage and commit again |
