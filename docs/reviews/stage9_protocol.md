# Stage 9 preregistration

Owner: system-evaluation implementation, 2026-11-15. Written before Stage 9 inference.
This protocol is an immutable input; deviations belong in stage9_results.md.

Use fresh project-authored draft needs in authored/system/scenarios.yaml. Translation,
code-switching, follow-up turns and business-state interventions retain one need/family.
Never reuse Stage 7 test/gold for selection or change their sealed inputs/results.
Audit exact and char-5-gram near overlap against SFT and earlier evaluation sources;
semantic independence remains uncertified without independent fluent review.
Development and confirmation families are disjoint. No confirmation outcome may change
runtime rules, prompts, labels, thresholds or cohort membership.

Control is Stage 8 at the starting Git HEAD. Primary seed 20261115; deterministic inference.
First run <=1% smoke. Retain every attempt, HTTP error and timeout. Full confirmation runs
once per sealed variant; technical retries retain identical inputs and previous attempts.
The only proposed routing intervention is prefix/word boundaries in the independent guard.
Adopt on development only if paired delivery-accuracy scenario95% CI excludes zero,
no language loses accuracy and no observed required critical delivery is lost. Check family
sensitivity. Unmeasured generalization is not a safety guarantee.

Report raw encoder intent/action vs delivered action/outcome and actual SQLite queue records.
Intent macro-F1 uses all35 labels; include coverage. Language slices ru,kk,en,mixed are separate;
mixed output preference is kk and overlaps neither monolingual row set. Bootstrap1000 seeded
scenario clusters and family sensitivity; paired IDs/labels/groups must match exactly.
Review the complete failed-need census up to50 per language. Target E2E100 rows/language and
safety50 needs/language are goals; report shortfalls rather than fabricate independent needs.

Retrieval: draft proof-section labels, canonical IDs, recall@5, full proof@5/@10 and packed
proof coverage for1024 control vs512/2048 budgets on identical development rankings.
These are finite draft locator proxies, not semantic sufficiency. Missing proof stays a failure.
No context change is adopted without a positive paired quality CI and no language regression.

Performance: identical captured development prompts for fp16/NF4, same pinned adapter/schema,
same device/seed/output budget; model load, first/warm p50/p95, throughput, RSS and allocated/
reserved VRAM separately. Evaluate encoder batch1/8 and bounded exact-text cache controls;
never cache live tools or customer responses. Compare outputs for parity. Performance-only
results cannot certify semantic quality; generator remains disabled and UNKNOWN queues assistance.
HTTP sequential/concurrent load uses isolated synthetic DB, unique request IDs, no hidden retries,
and reports503 separately from success throughput and verifies recovery/queue delivery.

Stress: >=1,000,000 independently normalized random1024-dimensional vectors, isolated collection,
bounded upload, committed-batch interruption/resume, exact/ANN fidelity, incremental changes,
snapshot/actual restore and full vector/payload inventory. This is index fidelity, not semantic
quality or encoder throughput. Preserve failed/partial attempts. No active KB namespace changes.

Semantic rubric: each factual claim must be entailed by cited, authorized, effective evidence;
exceptions, contradictions, relevance and completeness are separate labels. Citation membership
and numbers are diagnostics. Require independent fluent human labels and >=100 calibrated
items/language, pinned judge, kappa>=.6 before publishing semantic metrics. Otherwise all semantic
precision/unsupported-claim/completeness/accuracy values remain NOT_EVALUATED.

All attempts log source/config/data/model/KB/lock hashes, code HEAD, seed, resources, cost scope,
control and decision in local MLflow. No paid service is invoked. Back up and actually restore
new reports/data/tracking and code. Run full shared checks including real GPU/Qdrant. Publish
origin/main; report readiness from evidence. Do not start Stage 10.
