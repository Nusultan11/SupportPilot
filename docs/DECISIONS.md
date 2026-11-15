# DECISIONS — architecture decision records

Each ADR retains its ID, decision, reason/alternatives and revisit condition.
Accepted decisions remain in force until explicitly amended; historical measurements are
evidence for a choice, not current status. Current work: [PROGRESS](PROGRESS.md).
Full pre-maintenance narratives remain in Git at commit546f2df; raw evidence stays in manifests/MLflow.

## ADR-001 — Canonical agent instructions
Accepted 2026-11-11; clarified during owner-authorized maintenance 2026-11-15.
AGENTS.md is the only rules file; CLAUDE.md imports it with
`@AGENTS.md`. Per-tool copies drift; README is navigation, not another rulebook.
Revisit only for a tool unable to follow either entry point.
Contradictory historical status headings motivated explicit document ownership and a concise
current handoff (AGENTS §§3/7). Keep detailed history in Git/MLflow/verified artifacts rather
than a second archive document that duplicates the same evidence.

## ADR-002 — Documentation language
Accepted 2026-11-11. Docs/code/comments/commits use English; owner-facing communication
uses Russian. This matches the ecosystem and portfolio audience. Russian documentation
is the alternative if the owner changes preference.

## ADR-003 — Python 3.12, uv and Windows-native development
Accepted 2026-11-11. Use the project venv/lock; Windows GPU works, and no duplicate conda/
WSL environment is needed. Linux-only workloads may use authorized cloud/WSL/Docker.
Revisit when a required dependency lacks Windows support. Dependency/index policy: ADR-015.

## ADR-004 — Content-addressed manifests
Accepted 2026-11-11. Immutable raw revisions and manifests bind datasets/indexes to inputs,
parameters/code and hashes. Local snapshots provide recovery. DVC/LakeFS add coordination
overhead without a current sharing need. Revisit multi-machine/collaborative data distribution.

## ADR-005 — Parquet and Polars
Accepted 2026-11-11. Use lazy/streaming Polars and Parquet for pipelines; pandas is more
RAM-intensive, Spark unjustified. DuckDB is acceptable for ad-hoc analysis, not a second
pipeline engine. Revisit a measured limitation on larger workloads.

## ADR-006 — Licensed backbone and language adaptation
Accepted 2026-11-11; refined by ADR-024/025/028/029. Bitext EN supplies scenario wording,
not trusted policy answers; RU/KK use localized filtered MT plus native/authored gaps.
Templated rows do not imply distinct needs; volume is quality-bounded.
Source eligibility/licence evidence lives in configs/sources.yaml; gated/NC/conflicting sources
are not silently substituted. Published Bitext-derived data retains CDLA-Sharing obligations;
that does not automatically give all project data a redistribution licence.
Risks: MT semantics, RU leakage, unnatural templates. Revisit usable licensed native support
sources, fluent review or failure of language-quality gates.

## ADR-007 — Qdrant
Originally provisional (2026-11-11); operational/ranking choice established by ADR-014/030.
Metadata/date filters, dense+sparse search and incremental updates meet the retrieval contract.
FAISS lacks the metadata/update layer; pgvector adds a database; Chroma was not justified.
Revisit measured Qdrant overhead or new workload requirements.

## ADR-008 — Compact LLM audit
Originally provisional (2026-11-11); finalist choice is ADR-033.
Audit Qwen3-0.6B/1.7B, Qwen3.5-0.8B/2B, Gemma4 E2B and newer suitable compact models
by pinned licence/card, KK tokenization, real language/schema behaviour, memory and tooling.
Marketing parameter/language counts or community scores are screening leads, not evidence.
Local 4 GB supports smoke/resource probes; larger full runs use authorized free cloud compute.
Unsloth/alternative kernels are adopted only if measured useful, never a required assumption.
Revisit new development evidence; no post-holdout retuning.

## ADR-009 — Facts in KB/tools, behaviour in weights
Accepted 2026-11-11. Policies/catalogue/prices come from versioned KB; live orders/stock/
payments from authorized tools. SFT uses changed fictional values to teach reading evidence.
Training active facts would become stale and unverifiable. This boundary is not optional.

## ADR-010 — Structured responses
Accepted 2026-11-11; semantic acceptance clarified by ADR-033.
AssistantResponse JSON provides explicit intent/action/citations/escalation; free-text parsing
cannot reliably recover these fields. Validate schema/evidence/language separately and measure
constrained decoding compatibility. Valid JSON or number/citation matching is not grounding.

## ADR-011 — Local MLflow
Accepted 2026-11-11. Local SQLite + artifact directory preserve experiment metadata/history.
W&B needs an external account/data transfer; CSV lacks comparison/provenance support.
Revisit a genuine shared tracking requirement. Operational locations: RUNBOOK §4.

## ADR-012 — Optional development tools
Accepted 2026-11-11; amended 2026-11-12 at owner request.
Initially no skills/MCP were needed; repeated pipelines became code. Later owner-requested
Context7/Exa/GitHub/HuggingFace and local Serena/Playwright/MLflow were configured.
That amendment supersedes the original rejection of GitHub/Context7.

- MCP is development-only; no runtime/configuration dependency on a connected agent tool.
- Prefer built-in file/Git/CLI tools. Remote servers use minimal public queries;
  local stdio servers are off by default and enabled by the owner only when required.
- GitHub uses read-only toolsets; MLflow MCP is read-only. Code logs runs.
- Secrets stay in environment/local scope. Do not modify private client state to toggle servers.
- Serena uses tool interface/no-onboarding/no-memories; facts stay in owning project docs.
  Playwright uses installed Edge, isolated/headless, for future UI testing.
- Persistent setup is in .mcp.json/.serena/project.yml; client-specific setup is RUNBOOK §4a.
  A measured Claude Code setup is not a claim about every client's activation behaviour.

Alternatives rejected: always-on local processes, Docker GitHub MCP, private-state launcher,
Qdrant MCP with a different embedding protocol. Revisit actual tool gaps, connection cost,
client activation capabilities or a repeatable manual workflow that merits a skill.

## ADR-013 — Main branch and ignored backups
Accepted 2026-11-11, owner rule. Work/publish on main; origin is public.
Backups moved from the sibling SupportPilot-backups directory to project/backups that day;
all moved copies were reverified. Use code bundles and data/model snapshots, never public data.
Retention eligibility: latest10 code bundles/latest3 per snapshot name plus pinned stage evidence.
Deletion still requires separate owner permission; automatic eligibility is not authorization.
Same-disk backups are vulnerable to project/disk loss. Revisit off-machine private storage before
destructive operations or when expensive models/data warrant it.

## ADR-014 — Docker Qdrant; memory mode for tests
Accepted 2026-11-11. Qdrant1.20 server on localhost with a named Docker volume;
memory mode for tests; embedded mode only for small experiments.
Measured50k-point server p50/p95 11/61ms vs embedded604/716ms;
embedded Windows SQLite remains locked until process exit.
Named volume avoids non-POSIX Windows bind-mount issues; 127.0.0.1 avoids observed
IPv6 localhost delay. Qdrant Cloud adds account/data-transfer requirements.
Revisit Docker VM memory pressure with joint retrieval/generation; never delete its volume
as a cleanup/recovery shortcut.

## ADR-015 — Dependency groups and CUDA index
Accepted 2026-11-11, extended with each consuming pipeline.
Runtime dependencies, dev checks, gpu Torch and ml Transformers/PEFT/TRL groups are in
pyproject.toml; dev/gpu/ml install by default. uv.lock owns exact versions.
Torch uses the explicit cu126 index, verified against the laptop driver; do not silently
upgrade driver/index or install untracked packages. New pipeline dependencies are added when used.
Alternative: main-dependency CUDA wheels or unlocked installs harm lightweight/reproducible setup.
Revisit driver changes or a measured dependency compatibility requirement.

## ADR-016 — Redacted stdlib logging
Accepted 2026-11-11. JSON/console formatters redact sensitive keys, tokens, e-mails/phones/
long digit runs and default customer text to length/hash. Third-party HTTP logs stay WARNING.
structlog/loguru would add dependencies without solving redaction.
Revisit a real observability/OpenTelemetry requirement.

## ADR-017 — Tar snapshots with SHA256
Accepted 2026-11-11. Atomic uncompressed tar + per-file/archive/content hashes; identical newest
content deduplicates, restore verifies into a new/empty directory, pinning protects stage evidence.
Compression adds CPU cost for already-dense Parquet/weights. Plain copies lack integrity;
restic/borg/DVC add tooling. Revisit encryption, cross-machine storage or disk-budget pressure.
Disposal of a verified restore-drill copy is distinct from deleting its retained backup.

## ADR-018 — Executable licence/usage policy
Accepted 2026-11-11. Schemas cap each known licence's allowed use; unknown/conflicting licences
block use. Noncommercial licences are conservatively eval-only under possible commercial use.
Acquire/convert/analysis require reference_only; evaluation eval_only; training train.
Source loading enforces eligibility; documentation alone is insufficient.
Revisit owner-defined use restrictions, new licences or genuinely row-level rights.

## ADR-019 — Acquisition and raw conversion
Accepted 2026-11-11. Pinned HF files download resumably into .partial-revision, verify upstream
LFS SHA256/git blob SHA1, then publish atomically. Source converters stream original rows to
RawRecord Parquet with lineage/content hashes and explicit rejects.
Verified rerun/resume/forced-conversion parity supports this choice.
datasets.load_dataset hides originals/adds a cache; snapshot_download lacks our per-file
publication check; direct canonical conversion would mix acquisition and labelling.
Revisit non-HF sources or shard-level resume above roughly10GB.

## ADR-020 — Conservative cleaning and explicit pending output
Accepted 2026-11-11. Separate cleaned-row and canonical checkpoints preserve rejects/accounting.
Normalize encoding/spacing but retain typos/case/emoji/profanity with flags.
Mask PII/fill deterministic synthetic slots; never invent missing labels/policies.
Pending rows preserve unsupported source use/type/label reasons.
Lingua's Windows/offline support won over fragile fastText wheels; calibrated Belebele questions
2699/2700 and passages2700/2700, not a proof of short support-text accuracy.
Gates re-read outputs before publication. Garbled multi-script text is detected;
single-script nonsense and Latin KK remain limitations.
Revisit real RU/KK confusion evidence; detailed thresholds are in configs/base.yaml.

## ADR-021 — Stage 3 draft taxonomy
Accepted 2026-11-11; superseded for Stage 4 by ADR-023, retained for old builds.
intent_map.yaml maps source labels or an explicit null/reason; label/action conflicts remain
pending. Reference-dependent actions distinguish tool vs clarification; explicit complaints/
human requests escalate. Suspect labels are not guessed by a classifier.
Revisit only through the v1 taxonomy and versioned relabelling, not mutation of Stage 3 outputs.

## ADR-022 — Label-aware deduplication and grouping
Accepted 2026-11-11. Exact normalized template/final-text keys retain the most permissive
compatible source; conflicts stay explicit. Near candidates use char5-grams, seeded MinHash
128 permutations/32x4 LSH, verified exact Jaccard. Families join compatible pairs atJ>=.8
plus source alignments; scenarios use source label/action or explicit source scenarios.
J0.5–0.7 chained most Bitext rows into huge groups; even near text can have different intent.
AtJ>=.8 labels still require guarding; similarity never merges conflicting actions.
Bitext's few scenarios cannot support intent-complete scenario-disjoint heldout;
ADR-026/029 resolve this with authored evaluation situations.
Embedding clustering/datasketch/all-pairs were less appropriate for this bounded pipeline.
Revisit new multilingual data or demonstrated semantic leakage; preserve stable IDs.

## ADR-023 — Taxonomy v1
Accepted 2026-11-12. configs/taxonomy.yaml owns35 intents/10 groups/5 actions/4 mock tool names
and escalation conditions. Ordered explicit-text rules split ambiguous source labels;
unmatched/family-conflicting rows are excluded. No while-ai dialogue enters SFT because its
dialogue-level payment labels and defects do not support per-message targets.
INTENTS_V0 stays for old builds; tests equate v1 executable intent set and YAML.
Alternative in-place v0 rewriting would silently change historical identities.
Revisit measured intent-pair errors or newly authorized tool behaviour.

## ADR-024 — Localized MADLAD translation with QE
Accepted 2026-11-12. A 100-item comparison and LLM-only 50-item/system review found major
pre-QE RU/KK error rates: Opus 52/92%, M2M100 50/100%, MADLAD 40/34%, pivot KK 52%.
These are uncalibrated LLM review estimates, not fluent human measurements.
Use licensed MADLAD-400-3B NF4, beams2/batch8, localized fictional KZT/names/cities,
domain glossary repairs and strict rejection. Back-translate with Opus RU→EN/MADLAD KK→EN.
QE requires target language/Cyrillic, Latin-word<=34%, preserved names/numbers,
length ratio.45–2.6, no loops/Russian function-word leakage, chrF>=.30 and E5 cosine>=.80.
Keep raw MT/back-translation/QE/provenance and rejected rows; resume shard caches.
NLLB's NC rights/gated alternatives/paid APIs did not meet this task's constraints.
Revisit fluent reviewers, licensed native data or authorized paid generation.

## ADR-025 — Authored scenarios, altered KB and templated targets
Accepted 2026-11-12. Native/authored scenarios fill behavioural gaps; all are labelled
llm_generated with provenance. Alternate fictional KB values differ from active facts,
including outdated distractors. Render targets from cited evidence/tool fixtures and
language-specific agreement tables; never reuse source reference answers as factual truth.
Task distributions include grounded/no-evidence/conflict/injection/tool/clarify/escalate/OOS.
Free-form generation would introduce unverifiable targets; translated authored EN loses native
variation. Correlated templates and common authorship limit generalization; revisit human writers/
generator budget. Actual expanded counts belong to manifests, not this decision.

## ADR-026 — Scenario-disjoint evaluation
Accepted 2026-11-12; copy-exclusion direction refined by ADR-029.
Bitext originals/translations are train-only. Authored situations supply scenario-disjoint
validation/test; translations/families stay together. Gold is separately authored and decontaminated.
Quality gates can reduce planned volume; never upsample to force100k/90k–5k–5k.
Original70/15/15 action-stratified authored splitting was expanded to intent-complete trilingual
stratification in ADR-029. Gold's draft/common-author limits remain.
Revisit additional independent scenarios or validation/gold disagreement.

## ADR-027 — Deterministic rendered KB and measured parsers
Accepted 2026-11-12. Canonical facts/catalogue/specs render to six formats in three languages;
bytes are deterministic, and expected fact placements/register metadata support exact gates.
Choose pdfplumber for font-aware exact tables (row recall1.0 vs PyMuPDF.389/pypdf0);
python-docx retains styles, openpyxl preserves integer values. Full measurements:
`artifacts/reports/kb/` and the original ADR in Git.
Parsers read document signals, not expected answer text; scanned pages fail explicitly.
Chunk sections/FAQ pairs/records, split only over budget with headers retained.
All13 gates verify source/IDs/metadata/sections/facts/coverage/locators/completeness/
cross-language/language/tokens/PII/gold-reference resolution before activation.
Immutable self-contained versions contain changes/evidence maps; unchanged content creates no
new version. Failed gates never change current pointer.
Docling/OCR/AGPL alternatives add cost or rights complexity for these clean born-digital layouts.
Revisit scans/complex real documents or memory-bound corpora. No claim about arbitrary PDFs.

## ADR-028 — Native-language external examples
Accepted 2026-11-12. Pinned CC-BY mangi public sample and MASSIVE EN/RU enter a checkpointed
native step; questions alone, only reliable greeting/OOS labels. Full commercial mangi is unused.
Banking/telecom OOS requires explicit message cues; generic ambiguous text is excluded.
Native means no pipeline MT, not certified human authorship.
Registry policy/hash joins selection identity to prevent stale eligibility.
MT expansion alone or whole-domain automatic refusal would repeat scenarios/mislabel ambiguity.
Revisit new licensed native support sources and independently reviewed coverage.

## ADR-029 — Intent-complete heldout and independent retrieval data
Accepted 2026-11-12. Expand authored situations and stratify by intent with interleaved actions.
Keep assigned heldout questions; remove copied training inputs at existing thresholds instead
(amends ADR-026). Gold is an exclusion list, never a training authoring source.
Separate answer-wording RNG from evidence/tool-state RNG.
Expand demo documents/products without changing existing policies/gold abstention assumptions.
Author graded section qrels with explicit rationales; resolve only against exact KB versions.
Keyword qrels/gold paraphrases/slot-count inflation would bias evaluation.
Draft single-LLM authorship and missing fluent review limit independence; no dataset-wide
redistribution licence is introduced. Schema: DATA_CONTRACTS §14.

## ADR-030 — Measured hybrid retrieval and portable indexes
Accepted 2026-11-12. Use pinned BGE-M3 CLS1024/fp16, word sparse with KB-scoped Qdrant IDF,
min-max weighted dense.75,20 candidates/channel, BGE-reranker-v2-m3 over20,
top10 canonical output,1024 Qwen-token context. No diversification/score gate by default.
96 variants compared encoders/sparse/views/fusion/rerank/budgets. Development reranking
nDCG delta+.0339 CI[.0209,.0467] supported adoption; k20 superiority to k10 was inconclusive.
Qwen reranking was slower, sparse alone weak cross-lingually. Evidence:
`artifacts/reports/retrieval/`; pinned model cards/protocols in configs/retrieval_models.yaml.

Whole-query .02 initially met mono guards but cross recall fell.9744→.6667.
Amended cross/mixed guard rejects every positive threshold; default negatives therefore get
candidates (FPR1.0). Retain gate history; later inspected-holdout comparisons are exploratory.
Nonempty rankings never establish semantic answerability.

Use immutable local snapshots/checksummed caches, OS writer locks and fail-closed inventory
synchronization; validate parent changes or exact rollback diffs. Scope sparse IDF to live KB
because global IDF drifted after delete/reinsert. Recovery verifies all vectors/payloads and
query parity. Brief transition downtime is accepted instead of dual-collection complexity.
Revisit fresh reviewed evidence, unacceptable recall/latency or joint-serving resource pressure;
never mix KB versions or discard baseline backups.

## ADR-031 — Proof-aware scale protocol and isolation
Accepted 2026-11-13. Stage 6 volume/inspected holdout cannot establish scale or sufficiency.
Author need/family keys and AND-of-OR support groups; freeze family splits only after volume,
diversity, integrity/leakage review. Draft labels remain drafts; use scenario/family uncertainty.
Isolate namespaces/caches/collections from preserved baseline.
Separate100k random-vector load from semantic KB; measure upload/build/exact/HNSW/recovery.
Observed ANN Recall@10 ef32/64/128=.163/.294/.466 does not justify approximate default.
Generator identity includes text template; old interrupted/limited receipts are not rewritten.
Alternatives (translated row pooling, keyword proofs, random splits or old holdout reuse)
would introduce known bias. Revisit independently reviewed diverse confirmation data.

## ADR-032 — Explicit product profiles and dated policies
Accepted 2026-11-13 for controlled fictional evaluation.
200 explicit fictional products/20 groups/2,600 numeric attributes supply separate technical
documents, native units/names and stable SKUs; four dated policies provide temporal confusers.
Rule cloning or rewriting original catalogue would inflate diversity/change citations.
222 logical documents/666 versions/8,865 chunks pass gates and preserve original369 chunk IDs.
Related predicates/templates remain correlated (119 families); do not claim human dialogue diversity.
LF fingerprints and atomic candidate checkpoints prevent silent changed-input resume.
Keep retained ranking control/context1024: alternatives fail registered guards; exposed holdout
cannot tune packing/gating. Operationally use the expanded namespace with exercised baseline rollback.
Revisit fresh development/confirmation data for real conversation diversity and packing changes.

## ADR-033 — Sealed behaviour adapter and separate router
Accepted 2026-11-14 for guarded research/integration assessment, not autonomous factual/
critical-escalation use. Independent outcomes support this restriction, not a new selection.
Choose Qwen3-0.6B full pinned revision, all-linear rank16/alpha32/LR1e-4/512steps,
FP16 base/FP32 adapters, MT.5, schema mode, preregistered primary seed 20261113.
Validation score.83906 vs Qwen3.5-2B few-shot.51049 supports the engineering choice;
matched0.6B per-language CIs improve, but KK intent superiority over2B is inconclusive.
Eight80-step OFAT pilots are not a universal/full-factorial LoRA–QLoRA study.
Secondary NaN gradient warnings and UNKNOWN optimizer skips remain explicit.

Retain E5-small/C10 typed routing; measured LLM routers do not beat it per language.
Conditional draft action labels, uncalibrated probabilities and critical misses limit acceptance.
Seal prompts/demos/configuration/source/report hashes, encoder head and immutable inference
export before holdout; exclude training pickle from inference export, retain full checkpoints.
Semantic UNKNOWN defaults to localized fallback, not accepted factual generation or completed handoff.
Actual remote/local precision/resource scopes and per-language results:
[Stage 7 measured report](reviews/stage7_results.md).
Revisit only with new development evidence/fresh confirmation; heldout failures alter readiness,
not sealed parameters. Rollback preserves adapter=None/legacy boundary and earlier artifacts.
The sealed independent assessment improves raw structure but leaves KK gold macro-F1 about .224,
with no improvement over the 2B control, and misses 24/26 required KK safety escalations.
These draft-label diagnostics and absent calibrated semantics prevent autonomous acceptance.
Keep the selected research configuration unchanged; use the measured report for full uncertainty,
source-traced failure analysis and operational resource limits.

## ADR-034 — Authorized local delivery with an untrusted research generator

Accepted 2026-11-15 for Stage 8. Keep one local service and SQLite rather than adding CRM,
Redis or paid inference: there is no measured need for distributed operation. Use existing
encoder/retriever/generation boundaries; load the router on CPU and optionally the frozen
adapter in measured NF4 mode. Verify selection identity and sealed router/adapter bytes;
record Stage 8 source/lock identity separately from the immutable Stage 7 execution environment.
Do not rewrite its seal when adding serving dependencies.

Deliver deterministic authorized tool values; require explicit confirmation for a cancellation
request. Model confidence and action fields cannot authorize business writes. Preserve semantic
UNKNOWN-to-fallback, with actual transactional queue insertion before acknowledging recording.
Use an independent risk guard, bounded identity-owned process-local sessions, durable idempotent
receipts and a simple operator-only status transition. Persist allowlisted structured context
rather than raw messages, model candidates, arbitrary customer IDs or PII.

Always loading the generator adds memory/latency while no calibrated verifier can accept facts;
therefore default to evidence retrieval plus queued assistance and offer explicit research mode.
Returning model candidates based on schema/numbers would contradict ADR-033. Extractive excerpts
are labeled source material, not semantic answers. Regex coverage and draft fixtures do not
establish unrestricted multilingual critical-risk recall or fluent KK acceptance.

Serialize local requests and bound component waits with one worker; late read/inference results
cannot mutate tools or sessions. Hung drivers require restart; SQLite queue/state survives it.
Revisit distribution or independent semantic delivery only with new measured need and fresh
review/calibration evidence. Rollback: stop API/UI; existing offline libraries/artifacts and
Stage 7 selection remain intact. Queue/business DB stays preserved.

## ADR-035 — Narrow system guard correction without semantic promotion

Accepted 2026-11-15 on fresh Stage 9 development evidence, before confirmation. Add word
boundaries to English risk terms and prefix boundaries to inflected RU/KK roots. The existing
substring guard conflates ordinary product words with risks before authorized tools execute.
Keep the frozen encoder/model/prompt/KB/index unchanged. A learned semantic guard or new SFT
needs broader independently reviewed data; the small draft development cohort does not justify
training or recalibrating the sealed models. No model confidence authorizes facts or actions.

Compare paired Stage 8 vs candidate HTTP outcomes on identical draft needs; preserve family
sensitivity and distinguish raw routing from delivery. Confirmation does not establish overall
quality improvement. Retain the narrowly justified lexical correction without claiming a
multilingual semantic/risk solution; the measured report owns effects, critical misses and
readiness. Do not respond to exposed confirmation failures by extending the keyword list.

Reject deployment changes to context budget, cache, batching or precision from speed or
locator-hit proxies alone. Batched encoder throughput is not single-request latency, exact-
repeat cache hits do not model diverse traffic, and smaller precision does not guarantee speed
or schema quality. Keep generation off and semantic UNKNOWN-to-queue. The measured overload
limit is evidence for future design work, not permission to introduce distributed serving or
silently change Stage 8's serialized mutation boundary. No added dependency is needed.

Rollback restores the original risk pattern from the Stage 9 freeze; keep all attempts and
the selected research seal unchanged. Revisit with fresh development/confirmation families,
independent fluent review/calibrated claim support and measured end-to-end trade-offs.

## ADR-036 — Optional task orchestration and one local architecture model

Accepted 2026-11-15 for the owner's standalone engineering integration. Keep ROADMAP as stage
criteria, PROGRESS as verified state/authorization, and AGENTS as the only agent instructions.
Use upstream Shrimp Task Manager 1.0.21 for internal tasks with dependencies, verification and
continuation notes. A score>=80 completes only an internal task, never stage acceptance.
Preserve completed stages by references to existing evidence, not synthetic reruns or backdated
task completion. Small fixes need neither MCP. No GSD, separate Mermaid MCP or extra task system.

Use the official Structurizr MCP image pinned by digest with only DSL tools, loopback HTTP,
no host mounts, non-root/read-only execution and an opt-in Compose profile. Keep one versioned
DSL at docs/architecture/workspace.dsl and narrative interpretation in ARCHITECTURE. Existing
ADRs/review records stay authoritative; do not copy them into a second generated workspace.
Two external-document inspection findings are explicitly INFO, with owning paths in the model;
structural inspection failures remain errors. No Structurizr visualization/cloud service needed.

Measured upstream constraints justify the small local Shrimp MCP adapter: client roots otherwise
change DATA_DIR, direct JSON writes have no concurrency protection, and the tool catalogue includes
history resets/rule generation. Fix storage to ignored artifacts/engineering/shrimp, exclude
dangerous tools/modes, bound request bytes/task counts, serialize calls and hold an exclusive
process lock. No alternate task engine or runtime dependency is introduced. Upstream writes
remain nontransactional; stop writers and use verified project snapshots for recovery. Never
auto-clear a stale lock or corrupt state. Keep secrets out of the child environment and no GUI.

Pin the npm tree with package-lock; SDK 1.32.2 and compatible Hono/uuid overrides resolve measured
advisories. Upstream Zod 3 still has an oversized-collection DoS advisory (two npm high entries,
one affected dependency path); local trusted stdio and adapter input bounds limit exposure.
Do not claim a vulnerability-free upstream or force an incompatible Zod major migration.

Alternatives: direct upstream launch risks divergent stores/concurrent writes; always-on MCPs
add unnecessary processes/context; remote Structurizr would transmit the model; replacing working
runtime modules lacks measured need. Revisit upstream storage/SDK fixes or a measured multi-writer
requirement. Rollback disconnects these tools/stops only the engineering service; retain private
task state/backups and the DSL. Runtime, sealed inputs and historical assessments stay unchanged.
Reproduction/lifecycle and client-specific activation: RUNBOOK §4a.

## ADR-038 — Evidence-gated target architecture on existing boundaries

Accepted 2026-11-15 for the owner-authorized architecture and development-control redesign.
Preserve the modular Python monolith, thin HTTP/UI boundaries, domain libraries, Qdrant and
transactional synthetic SQLite Store. Separate online request authority, offline data/ML
qualification and deployment acceptance. The single DSL labels implemented, experimental and
planned capability; ARCHITECTURE interprets it, ROADMAP owns package/transition criteria and
PROGRESS owns verified state. Earlier ADRs, protocols, frozen experiments and contract versions
remain in force. This decision authorizes design documentation, not Stage 10 implementation.

Evidence: service code always withholds KB candidates despite the library's verifier protocol;
readiness only checks local worker/SQLite state; UI lists tickets but has no transition editor;
receipts grow without expiry; loaded retrievers retain their index artifact. Stage 9 measured
overload/quality limitations do not justify autonomous delivery or a multi-worker deployment.
The pending Stage 9 continuation is a separate unconfirmed working-tree increment, not a new
baseline quality claim. ARCHITECTURE §§1–8 and ROADMAP's A1–A8/9Q/10P mapping own the resulting
design and acceptance details, avoiding a second copied inventory here.

Choose independently calibrated claim verification AND an explicit fail-closed service delivery
path before factual answers; independent language/critical-risk evidence before quality promotion;
release identity binding before deployment; scoped diagnostics/retention/admission controls before
long-lived operation. Keep synchronous allowlisted tools and atomic queue/mutation/receipt commits.
Preserve existing API/storage contracts until a separately reviewed versioned change is needed.

Alternatives: schema/citation/score-only promotion is unsafe; always loading the generator spends
memory without deliverable facts; a microservice/agent framework rewrite duplicates working code;
multi-worker flags break process-local session/mutation assumptions; new databases/caches/queues
add migration/operations cost before a qualified workload exists. A single monolith limits capacity
and cannot forcibly cancel a hung inference thread, but its authority and recovery boundaries are
inspectable and already tested. Measure admission/process isolation first; choose further scaling
only on a declared bottleneck with quality parity, costs and rollback assessed.

Verification: source-to-model review, Structurizr validate/parse/inspect, linked package gates,
shared checks and protected-artifact/unrelated-change comparison. Engineering delivery cannot
waive independent semantic/fluent review or incomplete Stage 9 targets. Stage 10 still requires
separate owner authorization. Rollback restores only this documentation/DSL change; runtime,
contracts, frozen models/data/indexes, existing task history and backups remain unchanged.
Revisit on independently reviewed fresh quality evidence, a measured serving/recovery requirement,
or an explicit changed delivery goal, not exposed-holdout retuning or infrastructure fashion.

## Open decisions

Owner input is needed only if the work requires a paid MT/generation/judge budget or fluent
independent RU/KK reviewers. Their absence limits quality claims; it does not authorize
invented measurements or unrelated service dependencies.

## References

Historical source checks informed these ADRs. Current adopted versions/licences are pinned
in source/model registries; recheck primary sources before adopting a new version.

- [Shrimp upstream](https://github.com/cjo4m06/mcp-shrimp-task-manager) · [Structurizr MCP](https://docs.structurizr.com/ai/mcp) · [Inspections](https://docs.structurizr.com/workspaces/inspections)
- [Codex MCP configuration](https://developers.openai.com/codex/mcp/) · [Remaining upstream Zod advisory](https://github.com/advisories/GHSA-mrhx-c7xc-m3w5)

- [Qwen3 report](https://arxiv.org/abs/2505.09388) · [Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) · [2B](https://huggingface.co/Qwen/Qwen3.5-2B)
- [Unsloth Qwen3.5](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Gemma4 card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [Farabi community card](https://huggingface.co/nur-dev/farabi-1.7b) (screening lead)
- [SozKZ](https://arxiv.org/abs/2603.20854) · [KazMMLU](https://arxiv.org/abs/2502.12829)
- [Bitext retail](https://huggingface.co/datasets/bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset)
  · [customer support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [KazParC/Tilmash paper](https://arxiv.org/abs/2403.19399)
  · [KazParC](https://huggingface.co/datasets/issai/kazparc)
- [Qwen embeddings/reranker](https://arxiv.org/abs/2506.05176) · [MMTEB](https://arxiv.org/abs/2502.13595)
- [Qdrant hybrid](https://qdrant.tech/course/beginners/module-3/hybrid-search-in-qdrant/)
  · [Windows storage](https://qdrant.tech/documentation/common-errors/)
  · [1.20 release](https://qdrant.tech/blog/qdrant-1.20.x/)
  · [optimization](https://qdrant.tech/documentation/operations/optimize/)
  · [snapshots](https://qdrant.tech/documentation/operations/snapshots/)
- [CUDA wheel index](https://download.pytorch.org/whl/cu126)
- [Alpaca licence](https://github.com/tatsu-lab/stanford_alpaca)
  · [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus)
