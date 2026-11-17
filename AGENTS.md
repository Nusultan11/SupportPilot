# AGENTS.md — SupportPilot agent rules

The only rules file. `CLAUDE.md` imports it. Docs, code and commits: English.
Talk to the owner: Russian.

## 1. Project

- Support assistant for a fictional Kazakhstan online store. Languages: RU, KK, EN, mixed RU–KK.
- Requirements: [PROJECT](docs/PROJECT.md). Current state and next action: [PROGRESS](docs/PROGRESS.md).

## 2. Hard rules

1. Never invent policies, prices, order state or customer data. No evidence → clarify, abstain or escalate.
2. Reply in the user's language. Measure KK and mixed RU–KK separately.
3. Synthetic customers and orders only. No real PII.
4. Secrets live only in the environment or the ignored `.env`.
5. Use a data source only as `configs/sources.yaml` allows (licence + `allowed_use`).
6. No synthetic or LLM-generated training data without an owner-approved ADR.
7. Keep train / validation / test isolated. Select on validation. Never tune on test.
8. Not measured = `NOT_EVALUATED`. Keep failed attempts on record.
9. Hardware: Windows 11, RTX 3050 Laptop 4 GB, ~15.4 GB RAM. Bigger jobs: authorized free cloud.
   Any cost needs owner approval.
10. Add a dependency, service or abstraction only for a measured need.

## 3. Before work

1. Read: AGENTS → PROGRESS → the current ROADMAP stage → only the code you need.
2. Check `git status` and the real files. If a doc is wrong, fix the doc.
3. Work only inside the authorized scope. Never start the next stage on your own.

## 4. During work

1. Plan → implement → verify → review the final diff.
2. Significant choice: compare alternatives, cost, risk and rollback. Record it as an ADR.
3. Python 3.12, uv, src layout, typed public functions, Pydantic at boundaries.
4. Config: YAML + `SUPPORTPILOT_<SECTION>__<KEY>`.
5. Business logic in domain modules. CLI/API/UI only orchestrate.
6. Published data, indexes and models: immutable, versioned, manifest + SHA256, atomic write.
7. Split by scenario or paraphrase family, never by random rows.
8. Tests: unit for transforms, integration for pipelines, schema/language/refusal/injection
   cases for LLM boundaries. No tests that only mirror a trivial edit.
9. No dead code, no root-level temp files, no notebooks as source, no `new/old/final/v2` copies.

## 5. Checks

All must pass before a commit. Commands: [RUNBOOK §2](docs/RUNBOOK.md#2-checks).
A skipped check is unverified, not passed. Pre-commit's fast subset does not replace them.

## 6. Where files go

| What | Where | In Git |
|---|---|---|
| Code / tests / configs / docs | `src/` · `tests/` · `configs/` · `docs/` | yes |
| Datasets | `data/` | no |
| Models | `artifacts/models/` | no |
| Cloud jobs / reports | `artifacts/kaggle/` · `artifacts/reports/` | no |
| Temporary work | `artifacts/tmp/<task>/`, delete when done | no |
| Experiment tracking | `mlruns/` | no |
| Backups | `backups/` | no |

Tests that create Git repositories use the system temp directory, never the checkout.

## 7. Deleting

1. Delete only proven disposable files: caches and your own temporary work.
2. **Never delete `backups/` without separate owner permission.** Retention settings are not permission.
3. Never run `git clean -xfd`. Never blindly delete `.venv`, model caches or Docker volumes.
4. Before a recursive delete, check that the target is inside its directory and is not a link.
5. Irreversible deletion of important data needs the owner's OK.

## 8. Documentation

| Question | Owner |
|---|---|
| State, blockers, next action | [PROGRESS](docs/PROGRESS.md) |
| Stages and acceptance | [ROADMAP](docs/ROADMAP.md) |
| Requirements | [PROJECT](docs/PROJECT.md) |
| Architecture | [ARCHITECTURE](docs/ARCHITECTURE.md) · [workspace.dsl](docs/architecture/workspace.dsl) |
| Schemas, IDs, invariants | [DATA_CONTRACTS](docs/DATA_CONTRACTS.md) |
| Metrics, experiment rules | [EVALUATION](docs/EVALUATION.md) |
| Decisions | [DECISIONS](docs/DECISIONS.md) |
| Commands | [RUNBOOK](docs/RUNBOOK.md) |

- Change behaviour → update its owning doc. Link; never copy rules, commands or results.
- PROGRESS is a short current state, not a log.
- A new document needs a clear reader and purpose.

## 9. Git

- Work on `main`. `origin` is public: https://github.com/Nusultan11/SupportPilot.
- Conventional commits: `docs:` `data:` `kb:` `rag:` `ml:` `eval:` `api:` `ui:` `ops:` `test:`.
- Before a commit: review staged paths and diff, secrets, the 5 MB file limit.
- Never push data, models, `mlruns/` or `backups/`.

## 10. Finish

1. Acceptance criteria met with evidence. Checks green.
2. Owning docs updated.
3. Significant results backed up and restore-tested ([RUNBOOK §5–6](docs/RUNBOOK.md#5-code-backup)).
4. Commit and push.
5. Report: status, what changed, evidence, what is unverified, next action.

A submitted or running cloud job is not a finished experiment. An API/auth error means status
UNKNOWN; recheck before retrying.

## 11. Ask the owner only about

Costs, missing accounts or credentials, irreversible deletion of important data, changes to goals.
Everything else inside an authorized task: decide and do.

## 12. Tools

1. Built-in file, search, Git and CLI tools first.
2. MCP is optional development tooling, never a runtime dependency.
3. External tools get only minimal public queries: no code, data, logs, secrets or PII.
4. GitHub and MLflow MCP are read-only. Local MCP servers are off until the owner enables them.
5. Shrimp (task tracking) and Structurizr (architecture model) are optional for large tasks.
   Setup and rules: [RUNBOOK §8](docs/RUNBOOK.md#8-agent-tools). Never use their destructive modes.
6. Subagents only for independent authorized work. Verify their results yourself.
