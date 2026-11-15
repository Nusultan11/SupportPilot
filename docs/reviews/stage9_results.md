# Stage 9 — system evaluation and optimisation

Audience: the owner deciding whether the guarded prototype can advance to Stage 10.
This report owns interpretation; raw attempts, source/configuration hashes, MLflow identities
and recovery receipts remain private under `artifacts/reports/system/`.

## Assessment scope and decision

Status: **PARTIALLY COMPLETED**. Engineering/research measurements are complete; full quality
acceptance is not established. Delivery is complete only with the recovery/publication receipt.
Autonomous factual support and Stage 10 readiness: **NOT_READY**. The measured intent target
is missed; semantic quality remains **NOT_EVALUATED**. No Stage 7 model, adapter, selection,
prompt, demonstration, heldout result, KB or index is retuned. Generation remains disabled.

The accepted code change bounds risk words/root prefixes before learned routing. Development
demonstrates a narrow lexical bug: ordinary product text can otherwise trigger the safety guard
before an authorized tool. This is not a new semantic safety classifier. Overall confirmation
delivery improvement is not established; no broad multilingual-quality improvement is claimed.

## New inputs and isolation

`authored/system/scenarios.yaml`: 120 global needs, four parallel language rows each. Development
has20 needs/five families; confirmation has100 needs/44 families and all35 intent labels in each
language. Fifty confirmation needs are critical. Four followup cases per language add dialogue
turns, not independent needs. There are400 confirmation rows per variant, not400 independent
global needs. The source is project-authored LLM draft, eval_only; fluent independent review
and semantic independence are uncertified. Mixed rows prefer KK output; this does not certify
fluent KK or true code-switch detection.

Freeze `eaa70dd97c34f3c34b4e` binds the full identity, inputs/protocol/configuration, starting HEAD,
Stage 7 seal hash and both guard patterns. The overlap audit checks37,534 prior text surfaces:
zero exact/char-5-gram Jaccard>=.8 matches. Its SFT/earlier-assessment/YAML extraction scope and
source hashes are retained; this is lexical screening, not a semantic decontamination certificate.
No exclusion or relabelling follows outcomes. Previous Stage 7 test/gold is not rerun or trained on.

The later pinned E5-small cosine audit covers18,483 unique normalized prior surfaces and480 new
language rows at threshold.95. It flags89 rows: development EN6/RU5/KK8/mixed5; confirmation
EN7/RU16/KK27/mixed15. Thirty-eight rows across both phases have a near-train reference. These
are embedding flags, not human-confirmed duplicates; absence of a flag also does not certify
independence. The pooled confirmation metrics remain descriptive draft results, not certified
independent gold. No flagged row is removed after outcomes. Source/readback details and all
scores: `semantic_overlap/8279c216c374447cb0b24540ec921bd0/report.json`.

The first audit preflight correctly refuses two changed working diagnostic files; no model
inference runs in that failure. Its log remains retained. The technical retry reads their
baseline blobs from starting Git HEAD and requires the original SHA256 exactly; SFT and prior
assessment data stay unchanged. The retry completes without replacing reference data.

Development control `8073256b7aad499cb7fdf170984bea07`; candidate
`b0decd23b8fe4667b5cc47b49355524a`. Selection is sealed before confirmation in `selection.json`.
Confirmation control `d96c1687d1084c24bd036772ebec3e89`; candidate
`16d856af13d045eda38136d344df4282`. Both finish400 rows with actual model routing, FastAPI,
authorized SQLite tools and persisted queue checks. The original guard is swapped only for the
control; all other runtime components are identical. A separate four-row smoke remains retained.

## Per-language quality and uncertainty

Actual delivered-outcome accuracy is an operational draft-label metric: a policy question may
correctly queue assistance without answering it. It is not grounded-answer accuracy. Raw encoder
intent is measured separately, before the service guard; the narrow guard does not change it.

|Language|Raw macro-F1, all35|Stage 8 delivered %|Candidate delivered %, scenario95% CI|Paired delta pp, scenario95% CI|
|---|---:|---:|---|---|
|EN|.5458|88|89 [83,95]|+1 [-3,+5]|
|RU|.5423|90|92 [86.975,97]|+2 [-2,+6]|
|KK|.4922|86|86 [79,93]|0 [0,0]|
|Mixed RU–KK|.4765|88|88 [81,94]|0 [-3,+3]|

Bootstrap:1,000 resamples, seed20261115. Family sensitivity for delivery deltas:
EN[-4.26,+7.41], RU[-1.90,+7.84], KK[0,0], mixed[-2.89,+2.81] pp. Development EN/RU delta
was+20 pp with scenario CI[+5,+40], but family CI included zero: the effect concentrated in
one lexical family. Confirmation does not establish a positive overall delivery delta.

The fixed35-class macro-F1 bootstrap is sparse/nonregular: singleton classes disappear in
resamples, biasing percentiles below the point estimate. Scenario percentile ranges EN
[.327,.525], RU[.293,.515], KK[.272,.468], mixed[.259,.462] and family ranges EN[.315,.521],
RU[.289,.504], KK[.258,.470], mixed[.245,.460] are exploratory sensitivity only, not calibrated
population intervals. All raw point scores miss .90; no favorable sparse-class convention is used
to manufacture acceptance. Larger independently reviewed gold is still needed.

|Language|Observed critical queue delivery|Scenario bootstrap95%|Wilson sensitivity95%|
|---|---:|---|---|
|EN|48/50 (.96)|[.90,1.00]|[.865,.989]|
|RU|50/50 (1.00)|[1.00,1.00]|[.929,1.00]|
|KK|50/50 (1.00)|[1.00,1.00]|[.929,1.00]|
|Mixed|50/50 (1.00)|[1.00,1.00]|[.929,1.00]|

No critical delivery is lost relative to control. Both EN misses remain: electrical-spark and
nonliteral duplicate-charge language are not routed to actual queue delivery. Do not extend
keywords from these exposed confirmation examples. All-success bootstrap is degenerate; even
the Wilson sensitivity does not establish population recall>=.95. These are draft risk needs,
many with explicit cues, not an unrestricted-risk benchmark or measured harm rate.

Every acknowledged queue is checked against its customer-scoped SQLite row. Followups deliver
the expected authorized tool in4/4 cases per language; this small slice is not a dialogue-quality
acceptance benchmark. Queue recording is not an external operator response.

`error_review.json` retains the complete failed-delivery census (fewer than50 per language):
EN11, RU8, KK14, mixed12. EN has two critical-recognition/action failures, three intent-routing
and six routing/missing-action-integration failures. RU has three intent and five integration
failures; KK eight intent/six integration; mixed eight intent/four integration. Common symptoms
are an unsupported/freeform request getting the wrong action, CALL_TOOL routing without an
implemented shortcut, and policy/product questions clarifying instead of reaching source/queue
delivery. These are component diagnoses from retained observations, not fluent semantic review.

## Semantic boundary

The reproducible `ClaimReview` evaluator rejects self-review, nonfluent declarations, duplicate
items, inconsistent citation counts, mixed judge/rubric identities and calibration/assessment
family leakage. It requires>=100 independent-human-labelled calibration items per language and
identifiable kappa>=.6 before separate assessment metrics are emitted. Human identity/fluency
declarations require owner verification; code cannot certify a person. No such review packet is
available; the owner confirmed that neither labels nor reviewers are available. Semantic citation
precision, unsupported-claim rate, completeness, grounded-answer
accuracy and false factual-answer rates stay NOT_EVALUATED. No paid judge is invoked.

Schema/locator checks and delivered queue outcomes do not establish claim support. No default
ClaimVerifier is promoted. Absent/conflicting evidence, untrusted model outputs and tool/auth
failures retain the Stage 8 fail-closed boundary; shared regressions verify those boundaries.
Additional SFT is not justified by these draft outcomes: integration, critical coverage and
independent review must be resolved with fresh development/confirmation evidence first.

## Matched resources and alternatives

Encoder batch1/8 on80 identical development texts, three deterministic passes:
57.85/131.47 texts/s; largest vector difference6.71e-8. Exact-text bounded cache has80 entries,
122,880 vector bytes, first fill1.352s and repeated-pass22.5/11.5 microseconds. The repeat mix is
intentionally cache-perfect and excludes business/tool/customer-response caching. No online
batching/cache is adopted from this unrealistic hit mix or offline throughput alone.

The first context comparison uses two development needs/four languages, budgets512/1024/2048
and identical rankings. All return the expected document, which says nothing about exceptions
or complete proof. The separate locator assessment and its limitations are recorded below.
No context setting is selected from confirmation proxies;1024 remains deployed.

Supplement `retrieval/17f947d9ce20441285435e7457b2364b/report.json` contains20 fresh global
needs (16 positive/four absent-reference), four parallel language rows and three budgets on
identical rankings:80 retrieval calls/240 packed-context observations, not240 independent needs.
It is authored and frozen before its own inference, after guard selection; it performs no
retriever selection. Labels are draft section-locator groups, not atomic-fact semantic proofs.

|Language|Reference Recall@5|Full section groups@5|Full section groups packed at1024, scenario95%|
|---|---:|---:|---|
|EN|1.000|1.000|1.000 [1.000,1.000]|
|RU|1.000|1.000|1.000 [1.000,1.000]|
|KK|.9688|.9375|.8750 [.6875,1.000]|
|Mixed|.9688|.9375|.9375 [.8125,1.000]|

Full section groups@10 are1.0 for every language. Packed coverage at512 is EN1.0/RU.875/KK.75/
mixed.625; at2048 it is1.0 in each language. Increasing budget helps these finite references but
does not demonstrate semantic quality or justify selection from this small exposed supplement.
All four negative needs/language still receive nonempty context at every budget (proxy FPR1.0).
That is not a measured false factual-answer rate: generated UNKNOWN remains withheld/queued.
All-success intervals are degenerate; family sensitivity and all raw rows remain in the report.

Captured prompts SHA256 is identical across FP16/NF4 and joint NF4 runs. Pinned0.6B adapter,
schema prompt, greedy inference, original model seed20261113, max256 output tokens/45s;
three repeats per EN/RU/KK/mixed prompt,12 calls per condition. The experiment driver/bootstrap
seed20261115 differs from the explicitly recorded model seed; repeats are not training seeds.
All conditions produce6/12 valid schemas. Invalid/truncated raw completions remain failures;
valid JSON is not semantic support. No factual answer is authorized for customer delivery.
These precision studies call the backend directly; their zero customer deliveries are not
observed HTTP fallback deliveries. The separate system/Stage 8 service checks own that boundary.

The separate actual HTTP-generation probe `live_generation/cc204e855b11412d9200b81080e409ea/`
finishes with three actual NF4 model calls (EN/RU/KK), all withheld and actually queued. Two
authorized tool calls use no model; an explicit-human request also queues. All six HTTP calls
return200, four queue records are observed, and readiness remains guarded/semantic NOT_EVALUATED.
The operational output limit is512, unlike the matched precision study's256; do not combine their
latencies as a precision ablation. Observed HTTP EN/RU/KK policy latency38.597/15.757/29.753s;
tool latency.0387/.0210s. This is compatibility, not an independent quality or sustained SLO study.
Lazy component loading is recorded separately: CPU router2.666s, retrieval encoder4.874s,
reranker6.985s and generator9.815s. Composition time alone excludes later lazy loads.

|Scope|Model load s|Peak allocated VRAM bytes|Peak reserved VRAM bytes|Peak Python RSS bytes|
|---|---:|---:|---:|---:|
|FP16 isolated|10.400|1,755,439,104|1,910,505,472|2,331,308,032|
|NF4 isolated|8.887|1,099,847,680|1,262,485,504|2,069,925,888|
|NF4 + active retrieval|7.298|3,374,590,464|3,531,603,968|5,571,063,808|

Allocated/reserved counters are scoped to measured calls and exclude other processes/driver/shared
GPU memory; sampled RSS includes the Python process. Joint runs actually execute retrieval with
the LLM resident, then infer from the identical captured prompts. Twelve short requests do not
establish maximum-context/full-VRAM residency or sustained generation capacity. Cgroup samples
are separate. Minimum available RAM in the joint measurement is about2.00 GB.

|Language|FP16 warm p50/p95 s|NF4 warm p50/p95 s|
|---|---|---|
|EN|11.407/11.583|15.079/15.095|
|RU|14.684/14.692|18.129/18.610|
|KK|23.442/23.471|20.938/21.726|
|Mixed|24.773/26.421|29.204/29.543|

Each warm percentile uses only two observations; these are scoped diagnostics, not a language
SLO. NF4 reduces memory with no universal speed/schema benefit. Do not change the sealed Stage 7
FP16 research assessment or enable generation from these results. Paid compute0 USD; local
energy/hardware and billing/quota scopes are unmeasured.

## HTTP capacity

Actual loopback Uvicorn/httpx, ephemeral credentials/port, isolated synthetic SQLite, unique
request IDs, no hidden retries. Each concurrency level uses200 attempts: stock/order/operator
requests. Warm successful p95 at concurrency1 is10.61ms,115.26 successful requests/s,200/200
successes; first cold HTTP call88.17ms excludes composition/model load. The Stage 9 operational
target is p95<=100ms only for this deterministic mix at concurrency1, not freeform/generation.

|Concurrency|HTTP200|HTTP503|Successful req/s|Acknowledged/verified queue rows|
|---:|---:|---:|---:|---:|
|1|200|0|115.26|66/66|
|4|44|156|72.52|16/16|
|16|39|161|61.93|8/8|

All overload cases remain in denominators. The service's deliberate mutation lock rejects
concurrent work; this is not a scalable multi-user service. Successful p95 at4/16 is33.28/106.37ms,
but error rates invalidate any multi-user SLO claim. Authenticated readiness and a subsequent
tool request return200 after each load; invalid authentication returns401. Queue integrity and
recovery are observed. Task-owned Uvicorn stops after the test. No distributed-serving redesign
or Stage 10 implementation is inferred from this measurement.

## Million-record scale and recovery

Source identity `a7057c9dfb70c66d718c`:1,000,000 independent normalized random1024-dimensional
vectors, seed20261112 inherited from the retained scale engine, batches128, on-disk vectors/HNSW.
This is numerical index fidelity, not semantic quality, neural encoding or production corpus
capacity. Bootstrap uses20261115; model generation preserves20261113. These seed scopes are
distinct and retained, rather than silently rewriting earlier configurations.

Docker VM assignment:6,211,846,144 RAM bytes/16 logical CPUs. Qdrant limit2,147,483,648 RAM bytes,
4,294,967,296 combined RAM+swap bytes, no explicit CPU quota. No limit changes during the run.
The preceding1,000-record smoke completes interruption/replay, ANN/exact, update and snapshot
restore with1,000 matching vector/payload records and six exact queries.

Initial million run `16e4cc274a9a44a5b83f98f461995f7a` is **FAILED** locally. It commits all1M
points, uploads in654.367s (1,528.19 precomputed vectors/s), then builds a green1M HNSW index in
1,183.661s. The first cold exact query exceeds the10-second client deadline in three bounded
attempts. No completed search cohort exists for that attempt; remaining warm queries are UNKNOWN,
not successful or silently discarded. Cgroup peak2,147,409,920 bytes; failed-attempt Python peaks
were not retained by the original engine and are UNKNOWN. Container restart/OOM/oom_kill counters
are zero when checked afterward; the server remains healthy. This is a measured deadline failure,
not evidence of remote process failure or a completed million-record recovery.

The technical continuation keeps the same source/collection and byte-identical query vectors;
only the research client deadline changes to120s. It validates the acknowledged upload/count and
256 distributed vector/payload samples, links the failed receipt/run, and avoids re-upload/build.
Inherited upload/build timings are explicitly not new measurements. The complete later inventory
restore remains required. Exact execution source versions are archived under `system/source_versions/`;
continuation and failure receipts remain separate. Deployed API/Qdrant settings are unchanged.

Continuation `c0d5b3bf62e448ada7c97350949949b8` **FINISHED**. The120s deadline allows the
identical100-query cohort to finish; it does not make the original10s failure disappear.

|Search|First call s|Warm p50 ms|p95 ms|Sequential logical req/s|ANN agreement@10 vs exact|
|---|---:|---:|---:|---:|---:|
|Exact|12.978|11,247.89|11,792.97|.0883|Reference|
|HNSW ef32|.124|92.84|114.21|10.851|.022|
|HNSW ef64|.133|150.99|178.25|6.611|.044|
|HNSW ef128|.311|285.23|341.17|3.471|.083|

The numerical speed/fidelity trade-off on high-dimensional random vectors is severe. These
results neither measure BGE semantic recall nor justify switching the active semantic index
from exact search to a faster ANN configuration. The retained engine uses bounded request
retries; these are logical-request timings, not an instrumented network-attempt census.

Incremental operations add128, change128 and delete256 points, returning to1M live points,
in1.003s. Actual snapshot upload/restore creates a separate collection; complete inventories of
**all1,000,000 payload/vector records** match SHA256
`c7407e05de6e09758b82f04ee5bfaa401cfd8da4c89c4a28507ad26f5ae380a1`.
Six exact-query rankings match after recovery. Snapshot4,374,258,176 bytes has SHA256
`d7dbc1550844282da880e9cbe62ae04080d72e11d6a58b723cc462577c568179`.
The source/restore collections and source vectors remain retained; no active KB namespace changes.
The interruption is a committed-batch injection, not an OS process-kill resilience test.

Continuation measurement scope2,615.584s excludes inherited upload/build and source generation.
Sampled peak Python RSS983,076,864 bytes, minimum host available RAM4,591,579,136 bytes;
Qdrant cgroup peak2,147,454,976 bytes is separate and includes file cache. Torch allocation/
reservation0 means no model workload, not zero driver/device memory. Initial negative attempt,
continuation, cgroup samples, exact/ANN rows and source versions all remain retained. Source-vector
generation time was not instrumented; upload excludes encoding/source generation. No paid
compute is invoked; energy/hardware and uninstrumented costs remain unknown.

## Checks, delivery and completion decision

Full pytest: **810 passed, zero skipped**,429.55s; actual GPU/server checks execute. One expected
embedded-Qdrant exact-search warning. Ruff check/format195 files and strict mypy176 files plus
separate UI pass. Shared logs: `system/checks/`. Existing Stage 8 boundaries continue to pass;
the Stage 7 selection file hash matches the starting freeze. No failed/partial attempt is removed.

Five pinned snapshots cover benchmarks, full scale vectors/evidence, system reports and tracking.
Deep verification and actual restoration pass; nine restored service databases pass SQLite
integrity and17 current MLflow run identities/statuses match, including the failed scale attempt.
Exact snapshot IDs, restored byte/SQLite/run proofs,
code-bundle restore and origin/main publication are owned by
`artifacts/reports/stage9_delivery_receipt.json`; publication is verified only by that receipt's
Git proof. Same-disk backups do not protect against disk loss; no backup deletion is authorized.

**Completion decision:** the engineering assessment and narrow guard repair are delivered once
that receipt verifies recovery/publication. Stage 9's full quality acceptance remains incomplete:
new evaluation needs have unadjudicated embedding overlap, no independently reviewed fresh gold
or calibrated semantic judge is available, intent targets are missed and overall confirmation
delivery improvement is unconfirmed. Successful tests, structural JSON and random-vector scale
execution do not override those limits. The owner confirmed reviewers/labels are unavailable.
Keep autonomous generation off and Stage 10 **NOT_READY/NOT_STARTED**. Completing quality
acceptance requires independently adjudicated/fresh needs, fluent review/calibration and new
development-only improvement studies followed by untouched confirmation; do not retune from
the now-exposed Stage 7 or Stage 9 outcomes.
