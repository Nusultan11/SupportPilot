# EVALUATION — methodology and acceptance

This document owns metric definitions and experiment rules. Dataset schemas/identities:
[DATA_CONTRACTS](DATA_CONTRACTS.md). Current datasets/status: [PROGRESS](PROGRESS.md).
Detailed preregistrations under `docs/reviews/` are immutable experimental inputs;
result reports describe observations, not new tuning rules.

## 1. Evaluation datasets

| Set | Use | Planning target |
|---|---|---|
| SFT validation | Prompt/model/adapter selection and early stopping | ~5k rows, quality bounded |
| SFT test | Once per frozen finalist; final in-distribution report | ~5k rows, quality bounded |
| Gold | Separately authored final assessment; never training/tuning | >=300 items/language |
| Retrieval | Graded canonical chunk relevance; mono/cross/mixed queries | >=150 queries/language |
| E2E | Multi-turn authorized tool states/outcomes (Stage 8) | >=100 scenarios/language |
| Safety | Injection, PII, fabricated live state, conflict, ambiguity/OOS | >=50 needs/language |

Group translations/paraphrases by scenario/family. Report independent needs separately
from rows and overlapping language slices. Version/freeze inputs and decontaminate against
training; semantic near-copy flags remain explicit diagnostics, not certificates of absence.

Gold items contain turns, language, intent/action/escalation, required facts/tool fixtures,
allowed citation references, prohibited claims and review provenance.
Review levels: draft, llm_reviewed, human_reviewed, verified.
Only author + independent fluent human review supports verified; RU/KK require fluent reviewers.
One LLM's authorship/self-check is never independent human verification.

## 2. Metrics and denominators

Report per language plus mixed/cross/directional/task/difficulty slices when present.
Do not average away KK failures. Empty/unsupported cohorts are null/UNKNOWN, not zero.

| Area | Decision metric | Target |
|---|---|---|
| Data | Leakage / post-cleaning validity | 0 / >=99.5% |
| Intent | Macro-F1 across 35 intents on gold | >=.90 |
| Retrieval | Recall@5 | >=.85 per language |
| Grounding | Semantic citation precision | >=.95 |
| Critical escalation | Recall on required critical escalation | >=.95 |
| Generation/language | Grounded-answer accuracy / reply-language compliance | Establish after Stage 7 baselines |
| Serving | End-to-end p95 on target hardware | Establish in Stage 9 |

Targets are goals; missed targets require analysis and a readiness consequence.
Diagnostic metrics include schema validity, intent/action accuracy, confusion matrices,
escalation precision, ranking, proof/packing coverage, refusal/abstention, execution errors,
p50/p95, throughput, RSS/VRAM/storage and measured compute cost.

Definitions:
- Citation precision: cited chunks that semantically support the associated claims / all citations.
- Unsupported-claim rate: answers containing >=1 unsupported factual claim / all answers.
- Grounded-answer accuracy: answers correct for the query and fully supported by sufficient evidence.
- False-answer rate: answered unanswerable items / unanswerable items, under an explicit semantic rubric.
- False refusal/FNR: rejected answerable items / answerable items.
- Abstention accuracy: correct answer/abstain decisions / all answerability-labelled items.
- Language compliance: answer in requested language; script/LID is automatic evidence,
  not a substitute for fluent review or semantic correctness.
- Invalid/missing/raw execution outputs remain failures in the relevant denominators;
  never silently filter them to improve a rate.

Separate raw model output, schema/diagnostic validation, guarded routing, semantic support
and actual delivered responses. Analytic fallback implications are not observed API delivery.
Citation membership, numeric matching and nonempty retrieved context do not prove grounding.
Label task/action-based abstention and missing-proof checks as proxies with explicit scope.

## 3. Judges and review

Deterministic schema/reference/number/date/LID checks run first.
Semantic judges use a fixed rubric and pinned model/version, calibrated on >=100 human-labelled
items/language. Report agreement; do not use a language judge with kappa<.6.
Without calibration/independent fluent review, semantic quality remains NOT_EVALUATED.
A paid judge needs owner cost approval. Record LLM authorship/review without inventing
independence, certainty or native fluency.

## 4. Experiment and uncertainty protocol

Log every attempt in local MLflow: hypothesis, named control, changed parameters, data/KB/model
versions/revisions, seed, code/runtime/lock identity, per-language metrics, resource/cost scope,
conclusion and adopt/reject/follow-up decision.

1. Start with a <=1%/<=15-minute smoke; it proves execution, not quality.
2. Change one factor against a named control unless a registered sweep specifies otherwise.
3. Select on development only. Seal finalists before test/gold and evaluate each once.
   A technical retry preserves inputs/configuration and all prior failures; it cannot refresh holdout.
4. Use >=3 seeds when compute permits; report sample mean/std and omissions.
   Fix the primary seed before outcomes; no favorable-seed substitution.
5. Use1,000 seeded scenario-cluster bootstrap resamples and95% percentile CIs;
   add family-cluster sensitivity for related needs. Paired comparisons require identical IDs,
   labels and group assignments. A delta CI containing zero gives no evidence of improvement.
   One cluster cannot identify between-cluster generalization uncertainty.
6. Retain failed/negative/timeout/interrupted attempts and partial observations with reasons.
7. When targets are missed, inspect up to50 distinct failed needs/language.
   If fewer exist, report the entire census/count; never duplicate errors to reach50.

## 5. Planned comparisons

| Area | Comparisons |
|---|---|
| MT | Licensed EN→RU/KK systems, quality/review vs resource cost |
| Retrieval | Dense/sparse/hybrid, encoders, views/windows/overlap, fusion/top-k/rerank, budgets |
| LLM | Zero-/few-shot/schema/constrained; LoRA/QLoRA rank8/16/32, LR5e-5/1e-4/2e-4, targets, sequence/packing, encoder vs LLM routing |
| Serving | Baseline vs adapter, precision/quantization, cache/batching/context, isolated vs joint retrieval |

A screening/smoke or OFAT subset is not a full factorial study.
Compare tokenization on identical text, precision on matched captured prompts and resources on
the same hardware/scope before claiming a causal improvement.

## 6. Original retrieval protocol

Original benchmark and input identities are recorded in the local retrieval manifests.
Query corpus_lang filters are independent of question language.
For answerable queries use grade>0 for binary Recall/MRR, gain2**grade-1 for nDCG;
report grade>=2 sensitivity and multi-fragment coverage. Unjudged chunks count non-relevant
under finite draft qrels, not an exhaustive human relevance catalogue.
Empty-qrels queries stay outside ranking denominators; report their FPR separately.

Seed20261112 assigns scenarios to holdout by sha256(seed|scenario_id)[:8]%5==0.
Development selection starts with minimum-language recall and adopts reranking only if
paired Recall/nDCG improvement excludes zero with no mono recall loss>.005.
The original mono-only score gate was amended after it lost cross recall; the amended
guard also limits cross/mixed recall loss to.01. Because holdout had already been inspected,
subsequent historical holdout/CIs are exploratory, not untouched confirmation.
Gold never tunes retrieval. The gate history and failed attempts remain in local artifacts.

Measure whole packed context as well as rankings: metadata consumes the token budget,
canonical chunks/proofs can be lost in packing, and related text does not establish answerability.
Public online measurements bypass query/pair caches and separate model loading/first/warm calls.
Source and diagnostic evidence: `artifacts/reports/retrieval/`, ADR-030.

## 7. Scale retrieval contract

[Stage 6.1 preregistration](reviews/stage6_scale_protocol.md) and
`configs/retrieval_scale.yaml` define immutable source/volume/coverage gates, family split,
selection and heldout procedure. Count document IDs/language versions/registry keys/atomic
facts/needs/families separately; profile templates remain correlated synthetic evidence.

Required proof groups are AND across groups, OR within alternatives, with exact canonical
fact placement and effective-date support. Background qrels cannot satisfy a proof.
Separate missing fact/exception, partial, contradictory, ambiguous, historical and live-tool needs.
Report full evidence@5/@10, packed group coverage/sufficiency and packing loss.
FPR uses negatives, FNR answerable queries, abstention accuracy both. One-family/no-negative
slices have UNKNOWN uncertainty/rates. Supplemental wording/typo variants reuse exposed
development needs and add zero independent confirmation needs.

Freeze verifies source/code/tokenizer/qrel/split hashes before selection.
Changed protocols or inspected needs cannot become fresh holdout by renaming.
Preserve baseline namespace/results; isolate new semantic indexes, pair caches and random load.
Load data uses independently normalized random vectors: ANN agreement vs exact is numerical
index fidelity, not semantic quality or neural encoding throughput.
Verify all payloads/vectors and query parity after actual recovery; distinguish batch-injected
interruption from OS process death and re-encoding drift from exact stored-vector restoration.

## 8. LLM assessment contract

[Stage 7 preregistration](reviews/stage7_protocol.md) defines candidate audit, named-control
sweeps, training/resume, routing, sealed finalist selection and independent assessment.
Use altered training-KB facts, completion-only supervised targets, explicit length exclusions,
shared prompt formatting and train-only demonstrations.
Bind checkpoint resume to input/configuration/runtime identity; audit finite weights,
logged nonfinite gradients and any uninstrumented optimizer skips separately.

Final test/gold/safety observations require exact-version terminal completion, source/payload
identity and every output checksum before import. Store importer metadata separately from
remote execution provenance. Preserve raw completions, errors, timing/tokens and stable IDs.
Gold accepted references are allowed citations, not AND-proof labels; report reference coverage
without inventing gold sufficiency. Exposed development proof interventions and one-need
conflict fixtures supplement safety, not fresh independent confirmation.

Readiness considers absolute KK/safety/escalation quality and resource limitations, not only
a relative validation score. Semantic UNKNOWN must not authorize factual generation.

## 9. Resource and cost scopes

Report model/adapter files separately from runtime allocation and actual disk allocation.
Separate Python RSS, Torch allocated/reserved peaks, Docker/cgroup memory, other processes
and uninstrumented shared-GPU memory. Three local calls do not establish sustained SLO,
maximum context residency or all-case feasibility.

Record model loading, cold/warm latency, per-language p50/p95/tokens/s, upload/index/encode
work and sequential vs concurrent throughput. Distinguish process/device wall-hours,
allocated session time, billed hours and weekly account quota; unknown scopes stay UNKNOWN.
Free use means no paid service invoked, not zero local electricity or measured billing.
Do not run unrelated concurrent local model jobs during controlled performance measurements.
