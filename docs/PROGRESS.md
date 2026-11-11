# PROGRESS — current state (single source of truth for status)

Last updated: 2026-11-11

## Current stage

**Stage 0: Project audit and research. Status: COMPLETED.** Waiting for owner confirmation to start Stage 1.

## Stage 0 results

Audit:
- On 2026-11-11 `C:\Users\nurs\SupportPilot` was **empty** (no files, no git). There was nothing to keep, fix or delete, and no prior data needed a backup.
- Hardware and software facts are in RUNBOOK §1. Key constraint: 4 GB VRAM. No CUDA toolkit (`nvcc`) is installed, so kernels that build from source (causal-conv1d) are unlikely to build natively → use Docker/WSL2 or cloud (ADR-003).

Delivered:
- `AGENTS.md` (canonical rules), `CLAUDE.md` (pointer), `README.md`, `.gitignore`, `.gitattributes`.
- `docs/`: PROJECT, ARCHITECTURE, ROADMAP, DATA_CONTRACTS, EVALUATION, DECISIONS (ADR-001…013), PROGRESS, RUNBOOK.
- Git repo initialised on `main`.

Research findings (sources in DECISIONS §References):
- No open Kazakh or Russian e-commerce support dataset with a usable licence was found. EN backbone: Bitext retail (44,884 rows, 46 intents) + Bitext customer support (26,872 rows, 27 intents), both CDLA-Sharing-1.0 (ADR-006).
- Bitext rows are paraphrase families of few scenarios. The realistic count of *distinct* quality examples is unknown until Stage 3 EDA; reaching 100k needs substantial RU/KK adaptation and new generation.
- Model shortlist: Qwen3-0.6B/1.7B, Qwen3.5-0.8B/2B, Gemma 4 E2B. None explicitly lists Kazakh, so KK zero-shot quality must be measured (ADR-008). Unsloth advises against QLoRA for Qwen3.5.
- Embeddings to compare: multilingual-e5, BGE-M3, Qwen3-Embedding-0.6B. Rerankers: bge-reranker-v2-m3, Qwen3-Reranker-0.6B. Published benchmarks disagree and none covers KK e-commerce, so our own retrieval set decides (Stage 6).

Verified checks:
- Backup and restore round trip PASSED on 2026-11-11 with the RUNBOOK §3 procedure: bundle verified, sha256 OK, clone HEAD = `699dc23`, `diff -r` showed an identical tree. The restore directory was removed afterwards.
- No code exists yet, so there are no tests, lint or type checks. They start in Stage 1.

## Git and backups

- Commits: `699dc23` docs: establish project instructions, architecture and roadmap. The follow-up commit records this progress entry.
- Backups (`C:\Users\nurs\SupportPilot-backups\code\`):
  - `supportpilot-20261111T104335Z.bundle` (sha256 `1ad543f0…8205`): the restore-tested snapshot of `699dc23`.
  - Stage-end snapshot: the newest `supportpilot-*.bundle` created after the progress commit. Keep it under the retention rule.

## Open issues and owner decisions

1. Paid LLM API budget for KK/RU translation, generation and judging? (DECISIONS §Open decisions)
2. Fluent KK and RU reviewers for the gold benchmark? Without them, gold is `llm_reviewed`, not "independently verified" (EVALUATION §1).
3. GitHub remote for off-machine backup and CI?
4. Kaggle/Colab accounts: needed from Stage 6–7 for GPU runs. Quotas are not guaranteed.

## Notes for the next stage (Stage 1)

- Create the uv project with Python 3.12. Install torch from the CUDA 12.x index (cu126 wheels are known to work with driver 566.07).
- Keep the package skeleton minimal: only `common/` and `data/` are needed in Stages 1–3.
- The data snapshot tool must write outside the repo (RUNBOOK §4) and have a restore test.

## Next recommended action

Start Stage 1 (ROADMAP §Stage 1) after owner confirmation.
