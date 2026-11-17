# DECISIONS — architecture decision records

Format: ID · decision · why · revisit when. Accepted decisions stay in force until amended.
Only decisions still in force are listed. ADRs for removed stages (KB, retrieval, LLM, service,
human corpus: ADR-006–008, 010, 014, 019–022, 024–045) are retired by ADR-046; their full text
is in Git history at commit `1561971`.

## ADR-001 — One agent rules file
`AGENTS.md` is the only rules file; `CLAUDE.md` imports it. README is navigation only.
Why: per-tool copies drift. Revisit: a tool that can follow neither entry point.

## ADR-002 — Documentation language
Docs, code, comments and commits in English; owner communication in Russian.
Revisit: owner changes preference.

## ADR-003 — Python 3.12, uv, Windows-native
Project venv + `uv.lock`. Windows GPU works; no conda/WSL duplicate. Linux-only work may use
authorized cloud/WSL/Docker. Revisit: a required dependency without Windows support.

## ADR-004 — Content-addressed manifests
Immutable raw revisions; manifests bind outputs to inputs, parameters, code and SHA256.
DVC/LakeFS add overhead without a sharing need. Revisit: multi-machine data sharing.

## ADR-005 — Parquet and Polars for pipelines
Lazy/streaming Polars + Parquet. pandas uses more RAM; Spark is unjustified; DuckDB only for
ad-hoc analysis. Add the dependency when the new data pipeline first needs it.
Revisit: a measured limitation.

## ADR-009 — Facts in KB/tools, behaviour in weights
Policies, catalogue and prices come from a versioned knowledge source; orders, stock and
payments from authorized tools. Training data uses changed fictional values.
Why: trained facts go stale and cannot be verified. Not optional.

## ADR-011 — Local MLflow
SQLite + local artifact directory (`mlruns/`). W&B needs an external account; CSV lacks
provenance. Revisit: a real shared-tracking need.

## ADR-012 — Optional development tools
MCP is development-only, never runtime. Built-in tools first. Remote servers get minimal public
queries. GitHub and MLflow MCP are read-only. Local stdio servers are off by default; the owner
enables them. Secrets stay in the environment. Setup: RUNBOOK §8.
Revisit: an actual tool gap or connection cost.

## ADR-013 — Main branch and ignored backups
Work and publish on `main`; `origin` is public. Backups live in ignored `backups/`.
Retention: latest 10 code bundles, latest 3 per snapshot name, plus pinned ones.
Retention is eligibility only; deletion needs separate owner permission.
Same-disk backups do not survive disk loss. Revisit: off-machine private storage.

## ADR-015 — Dependency groups and CUDA index
Runtime dependencies in `[project]`; `dev` (checks) and `gpu` (Torch, cu126 index) groups.
`uv.lock` owns exact versions. Do not silently change the driver/index. Add dependencies
only when code uses them. Revisit: driver change or measured incompatibility.

## ADR-016 — Redacted stdlib logging
JSON/console formatters redact secrets, e-mails, phones, long digit runs; customer text is
logged as length/hash by default. Revisit: a real observability requirement.

## ADR-017 — Tar snapshots with SHA256
Atomic uncompressed tar + per-file/archive/content hashes. Restore verifies into a new or empty
directory. Pinning protects stage evidence. Revisit: encryption or cross-machine storage.

## ADR-018 — Executable licence and usage policy
Schemas cap each licence's allowed use; `unknown`/`conflicting` block use; noncommercial
licences are eval-only. acquire/convert/analysis need `reference_only`, evaluation `eval_only`,
training `train`. Code enforces it (`data/policy.py`). Revisit: new licences or owner restrictions.

## ADR-019 — Verified acquisition
Pinned HF/GitHub files download resumably into `.partial-<revision>`, are checked against the
upstream LFS SHA256 / git blob SHA1, then published atomically. Revisit: non-HF sources or
very large (~10 GB+) shards.

## ADR-023 — Intent taxonomy v1
`configs/taxonomy.yaml` owns 35 intents, 5 actions, 4 mock tools and escalation reasons.
`common.schemas.INTENTS` must equal it (tested). Revisit: measured intent confusions or new
tool behaviour.

## ADR-036 — Optional Shrimp and Structurizr
Shrimp Task Manager 1.0.21 tracks internal tasks for large work through a local adapter
(`tools/mcp/`): fixed store in `artifacts/engineering/shrimp`, no destructive tools, bounded
input, single writer. A task score never equals stage acceptance. Structurizr MCP (digest-pinned,
loopback, read-only) validates the one model `docs/architecture/workspace.dsl`.
Revisit: upstream fixes or a multi-writer need.

## ADR-046 — Project reset (2026-11-17)
Owner decision: Data and ML pipelines restart from scratch.
Removed: knowledge base and its sources, all synthetic and LLM-authored data, translated/MT data,
SFT/gold/benchmark sets, retrieval, LLM, routing and service code, their tests, configs and reports.
Kept: engineering foundation (config, logging, manifests, backups, env/MLflow checks),
verified acquisition, source registry, taxonomy v1, real human-authored raw datasets,
`artifacts/`, `mlruns/` and all backups.
Why: synthetic data was not used going forward; the old pipelines depended on it.
Recovery: code in Git (`1561971`), data in `backups/data/`.
Revisit: never silently; new pipelines get new ADRs.
