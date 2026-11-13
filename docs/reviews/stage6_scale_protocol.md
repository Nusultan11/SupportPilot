# Stage 6.1 preregistered protocol

This protocol is written before Stage 6.1 inference. Configuration is in
`configs/retrieval_scale.yaml`. Stage 6 artifacts remain immutable; their inspected
holdout cannot confirm a new result. Stage 7 must not start in this work cycle.

## Counting and evidence

Count documents by document ID, language versions by `(doc_id, lang)`, and facts by
canonical semantic key, never by translations, placements, identifiers or text hashes.
Report atomic facts separately from distinct predicates and reasoning families.
Slot-filled repetitions must not be advertised as independent scenario diversity.
An information need identifies a subject, requested predicate and conditions; translations,
typos and paraphrases are variants of that need. Group related needs by `family_id` before
splitting. Keep source/author and draft review status on every judgement.

Qrels specify relevance; a separate list of required evidence groups specifies sufficiency.
All groups must be covered, with at least one member from each group. Alternative sources
are OR within a group; jointly required facts are AND across groups. Background relevance
does not satisfy a required group. Missing/live/ambiguous/contradictory/expired evidence
receives its own negative category. Neither scores nor a nonempty context prove answerability.

## Isolation and selection

Freeze source hashes, KB/SFT/gold versions, qrels, family split and protocol hash before
model comparison. The freeze is content addressed and never overwritten. Selection sees
development results only. Save the chosen configuration and development-only rationale
before permitting holdout evaluation. A changed protocol or changed inputs requires a new
experiment identity and cannot relabel an inspected holdout as untouched.

Use BGE-M3 weighted .75 + BGE rerank20 as the control. Compare dense, word sparse,
RRF and weighted fusion, candidates 20/40, rerank 10/20, and context 512/1024/2048.
The frozen configuration also includes E5-base dense and hybrid + BGE rerank20 as a
second embedding backend; pinned model identities remain those of Stage 6.
Choose using minimum monolingual Recall@5; no language may lose more than .005 and
cross-language/RU-KK loss is bounded by .01. A paired CI containing zero is inconclusive.
Keep score gating disabled unless development satisfies those guards and improves FPR.
Do not amend the rule after seeing holdout.

Report Recall@5/MRR@10/nDCG@10, all-required-group recall@5/@10, packed evidence
coverage and sufficiency, unanswerable FPR, answerable FNR and abstention accuracy.
Report RU/KK/EN, six directional pairs, RU-KK, rare documents, multi-document,
multi-fragment, temporal and hard negative categories separately, including denominators.
Bootstrap by scenario, with 1,000 draws and paired deltas. Family-cluster sensitivity is
needed where templated wording produces correlated needs. Empty cohorts are UNKNOWN.

## Scale and recovery

Use a separate, explicitly synthetic 100,000-vector/chunk workload. Compare exact search
against HNSW ef 32/64/128 on the same vectors and query IDs. Random-vector agreement
measures ANN fidelity, not semantic retrieval quality. Compare on-disk and memory costs
only when actually measured. Record upload vs vector generation vs total indexing time;
do not count precomputed-vector upload as model encoding throughput.

Measure cold/warm latency, p50/p95, sequential throughput, process RSS, machine available
RAM, GPU allocated/reserved bytes and server storage. Verify updates, deletes, idempotent
replay, interrupted/resumed batches, full inventory and snapshot upload to another collection.
Never delete baseline collections, prior artifacts or backups. Preserve failed attempts in
MLflow. Report Docker/process and model/server resource boundaries explicitly.

## Alternatives researched

Exact dense search is the reference for ANN fidelity; HNSW reduces search work at the
cost of a tunable recall/latency tradeoff. On-disk vectors avoid budgeting 100k x 1024 x
4 bytes entirely as resident Python arrays. Qdrant supports memory-mapped vector storage
and local snapshot recovery; use the pinned existing 1.20 server rather than add a service.
Sources: https://qdrant.tech/documentation/operations/optimize/ and
https://qdrant.tech/documentation/operations/snapshots/ (checked 2026-11-12).

Draft LLM authorship and automatic integrity checks do not establish fluent KK quality,
exhaustive qrels, human relevance or production readiness. An unmet volume or diversity gate
keeps Stage 6.1 PARTIALLY COMPLETED even when engineering tests pass.
