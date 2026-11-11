# RUNBOOK — verified operational commands

Only commands that were actually run go here, each marked with the stage that verified it.
Shell: Git Bash on Windows unless stated otherwise.

## 1. Environment check (verified Stage 0)

```bash
nvidia-smi                                   # driver, CUDA runtime, VRAM
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv --version && git --version && docker --version
```

```powershell
Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors
[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,2)
Get-PSDrive C
```

Reference values (2026-11-11): Ryzen 7 5800HS (8C/16T), 15.4 GB RAM, RTX 3050 Laptop 4 GB
(driver 566.07, CUDA 12.7 runtime, no CUDA toolkit/nvcc), 734 GB free on C:, Python 3.12.3
(torch 2.12.0+cu126, CUDA available) and 3.13, uv 0.12.1, git 2.45.1, Docker 29.8.2 (WSL2,
~6 GB assigned to the Docker VM).

## 2. Project setup

Added in Stage 1 (`uv sync`, checks). Not available yet.

## 3. Code backup (git bundle) — verified Stage 0

Location: `C:\Users\nurs\SupportPilot-backups\code\` (outside the work tree).

Create:
```bash
BK=/c/Users/nurs/SupportPilot-backups/code; mkdir -p "$BK"
TS=$(date -u +%Y%m%dT%H%M%SZ); F="$BK/supportpilot-$TS.bundle"
git -C /c/Users/nurs/SupportPilot bundle create "$F" --all
git bundle verify "$F" && (cd "$BK" && sha256sum "$(basename "$F")" > "$(basename "$F").sha256")
```

Restore (to a new directory, never over the live repo):
```bash
cd /c/Users/nurs/SupportPilot-backups/code && sha256sum -c supportpilot-<TS>.bundle.sha256
git clone -b main supportpilot-<TS>.bundle /c/Users/nurs/SupportPilot-restore
git -C /c/Users/nurs/SupportPilot-restore log --oneline -3
```

Retention: keep the 10 newest bundles plus every bundle listed in PROGRESS.md as a stage-end snapshot.
Delete older bundles by hand after checking that list. There is no automatic deletion.

## 4. Data and model snapshots

Stage 1 adds the tool (tar + sha256 manifest, retention 3 per dataset/model + stage-end snapshots).

## 5. Diagnostics

| Symptom | Check |
|---|---|
| `torch.cuda.is_available()` False | `nvidia-smi` works? torch wheel built with `+cu12x`? Wrong venv? |
| CUDA OOM on 4 GB | batch 1, gradient checkpointing, shorter `max_seq_length`, 4-bit, close other GPU apps (`nvidia-smi` processes) |
| Docker not responding | Docker Desktop running; `wsl -l -v` shows `docker-desktop Running` |
