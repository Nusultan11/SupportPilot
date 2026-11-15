# Stage 9 quality continuation preregistration

Audience: the owner assessing the next guarded-service quality increment. Authored
2026-11-15 before new model inference. This protocol is immutable after freeze.
Earlier Stage 7/9 outcomes diagnose integration limitations only; do not rerun or
select using them. No training, model/index change, semantic acceptance or Stage 10.

Control: exact orchestrator source from Git 4b082d371b42feb4e91f85375ad36c0cf8c7ba98,
loaded from a checksummed private archive with identical frozen encoder, taxonomy,
retriever and synthetic SQLite fixtures. Candidate: service action integration and
contextual safety interpretation, one combined intervention. Do not attribute its
effect to individual rules. Retain raw encoder decisions separately from effective
service decisions, HTTP delivery, tool facts and persisted queue records.

Inputs: authored/system/quality.yaml, project-authored LLM draft, eval_only, no
external source text. Four parallel languages share one need. Development and
confirmation use distinct concrete subneeds/family IDs; translations, references,
followups and business-state changes never add independent needs. Lexical overlap
screen against prior SFT/assessment surfaces and across splits before inference;
semantic independence and fluent/native quality remain NOT_CERTIFIED. No independent
human review exists. Do not exclude/relabel after outcomes. All 35 intents must
occur in both splits and each language; risk/conflict/false-positive/missing-slot
and dialogue supplements have explicit expectations, including permitted tool facts.

Primary seed 20261115; deterministic inference. A <=1% smoke precedes full runs.
Select on development only: positive paired actual-delivery delta CI in at least
one language, no language point loss and no lost critical handoff. Also require
no observed unauthorized write or fact disclosure. Record family sensitivity and
all35 intent macro-F1; neither target .90 intent nor .95 critical recall is waived.
Seal all runtime/config/input hashes and selection before confirmation; evaluate
each variant once, retaining a pre-run marker and partial/failing observations.
After confirmation, no runtime/label/threshold edits informed by those outcomes.

Report per EN/RU/KK/mixed: numerator/denominator for intent/action/delivery, critical
queue recall (and reason correctness separately), false critical escalations on ordinary needs,
authorized tool success and complete dialogue success. Distinguish benign semantic
fallback queues from critical escalation. Require expected tool facts, not merely
delivery=tool. Use 1000 paired scenario bootstrap resamples and family sensitivity
for every metric including nonlinear all35 macro-F1. Sparse absolute macro-F1 CIs
are exploratory; all-success recall also has Wilson sensitivity. Empty slices are
UNKNOWN. Failed/missing HTTP/route/turn outcomes remain failures. Review all failing
needs up to50/language. Report p50/p95, errors, process RSS, load/wall time and cost
scope; seed repeats do not add evidence for this deterministic runtime.

Use actual local frozen encoder/retrieval and FastAPI/SQLite; generation stays off.
Semantic facts remain UNKNOWN-to-queue. Log every attempt/provenance/source hashes
in MLflow. Preserve original protocols/inputs/results; fresh files and private
artifacts/reports/system/quality own this extension. Full checks, deep snapshot
verification, actual restore and origin/main publication are required for delivery.
