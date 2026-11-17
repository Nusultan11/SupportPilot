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
enables them; on 2026-11-17 the owner enabled Playwright (pinned, headless, isolated),
Shrimp and Structurizr in `.mcp.json`. Secrets stay in the environment. Setup: RUNBOOK §8.
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

## ADR-047 — Stage 1 corpus from human-authored sources (2026-11-17)
Decision: `supportpilot data corpus build` turns the seven selected raw sources into one
versioned corpus (DATA_CONTRACTS §6–9): an intent dataset (`core` / `unverified` / `excluded`)
and a separate EN document-grounded dataset (doc2dial). No synthetic, LLM or MT text.
- Labels: an authored recipe maps each source label to a v1 intent; a row enters `core` only
  when the label alone decides it or a cue in the text confirms it, else `unverified`.
  Alternatives: trusting source labels blindly (prompt/scenario noise), LLM labelling
  (not independent, needs an ADR), human annotation (no reviewer or budget yet).
- Admission: `mangi_kk_support_qa` is not admitted (authorship undocumented); `xsid_kk` is
  validation/test only (its paper: evaluation data); translations are flagged
  (`human_localised`), never presented as native speech.
- Splits: union-find groups (dialogue, parallel id, near-duplicate cluster) split by a seeded
  hash (stable when sources are added); doc2dial's 383 document families use a size-aware
  deterministic assignment because hashing so few large groups skews the ratios.
- Near duplicates: char-4-gram MinHash/LSH confirmed by exact Jaccard (deterministic, no model)
  instead of all-pairs (quadratic) or embeddings (model dependency, threshold drift).
- Format: JSONL (≈50k rows, 77 MB) without Polars (ADR-005: no measured need). `numpy` (MinHash)
  and `pyarrow` (reading upstream Parquet) became direct dependencies; both were already locked.
Cost: none. Risk: cue rules are heuristics; label quality is NOT_EVALUATED by humans.
Rollback: the recipe is data; edit and rebuild (old versions stay immutable).
Revisit: independent review results, newly licensed KK/RU–KK data, or a purchased/commissioned set.

## ADR-048 — Stage 1 completion round (2026-11-17)
Decision: Stage 1 stays PARTIALLY COMPLETED; the corpus is corrected and republished as
`corpus-20261117-26d02054`; closing the remaining gap needs commissioned native-speaker data
and independent reviewers (owner decision, PROGRESS).
- Research: 8 more sources investigated beyond the HF Hub (registry section ADR-048). None is
  usable: oasst2 was measured (0 store requests in 5,710 RU prompts), others lack a data licence,
  are non-commercial, out of domain, LLM-translated, private or commercial.
- Labels: a seeded agent review (LLM, not independent) of 307 v1 rows found 36 wrong labels.
  Recipe fixes, each tied to a taxonomy v1 description: an address on file is `account_help`
  (profile address), `change_address` only with an order/shipping mention (ABCD account flow,
  MINDS-14); money back before shipping is `cancel_order`; bare "I have a shipping question" and
  "an issue with a return" become `unverified`; "who are you" is not a greeting.
  Alternatives: keep labels and report the error rate (knowingly wrong targets), LLM relabelling
  (needs an ADR, not independent), drop the sources (loses all RU in-domain rows).
- Code-switch detector: one in-clause particle (ма/ме/ба/бе) without other Kazakh evidence no
  longer counts (the only v1 RU–KK row was a Russian typo); a clause-final particle still does.
- Publishing: a damaged existing version with the same content hash stops the build with a clear
  error instead of a failed overwrite (found after an interrupted build).
Cost: none. Risk: cue rules stay heuristics; the audit sample is small and not independent.
Rollback: the recipe is data; `379aa455` stays published and backed up.
Revisit: commissioned or newly licensed KK / RU–KK / RU data, or independent review results.

## ADR-049 — Commissioned collection and independent review (2026-11-17)
Decision: close the Stage 1 gap (natural in-domain RU, KK, mixed RU–KK; independent human
review) with commissioned native-speaker writing and blind independent review, taken in
through `data collection` / `data review` (DATA_CONTRACTS §10, [COLLECTION](COLLECTION.md)).
Route proposed to the owner: direct fixed-price contracts with Kazakhstan-resident freelancers
on Upwork, maximum USD 1,628.84 including platform fees; nothing is spent before approval.
- Route alternatives (researched 2026-11-17): Prolific (no participants in Kazakhstan; 42.9 %
  corporate fee); Toloka self-service/experts (33 % commission plus LLM QA; Kazakh not in its
  published expert languages; crowd text-generation authorship hard to verify); Yandex Tasks
  (RUB billing, 22 % VAT, Russian legal entity, crowd authorship as Toloka); data vendors
  FutureBeeAI, Pangeanic, IndecalAI (no published price; quotes need the owner's contact
  details); KZ translation bureaus (translation, not natural requests). Upwork: KZ freelancers
  with KK/RU annotation profiles, published fees, §6.4 assigns all work-product rights on payment.
- Briefs are seeded structured facets over taxonomy v1 descriptions (agent-authored, no example
  sentences); writers phrase every request. Scenario (brief) is the split unit across slices.
- Registry: `fetch.type: delivery` pinned to the content sha256; licence id
  `commissioned-assignment` → `train` (contract assignment, not a public licence).
- Review: blind relabelling (the v1 `review/` sheets show the label, so they cannot measure
  agreement), seeded 20 % second reviewer, Cohen's kappa ≥ 0.6 per stratum, Wilson CIs,
  reviewer ≠ writer enforced.
Cost: ≤ USD 1,628.84 if approved. Risk: contractors may use generators despite attestations
(mitigated by screening, timing, duplicate/language flags, reviewer judgement, removal of a
writer's rows); small reviewer pool (2) limits kappa precision. Rollback: an unapproved or
failed collection leaves the corpus at `26d02054`; the code paths are inactive without a
registry entry. Revisit: owner decision, vendor quotes, or a newly licensed natural source.
