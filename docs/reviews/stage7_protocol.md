# Stage 7 preregistered execution protocol

Owner authorization: Stage 7 task attachment and Kaggle CLI execution, 2026-11-13.
This file defines the procedure, not execution status (see PROGRESS.md).

## Inputs and isolation

Preserve Stage 6.1 serving and frozen artifacts. Use the existing SFT training KB
variant for optimization. No demo-store policies, test, gold or retrieval holdout
observations enter training, early stopping, prompt selection or parameter selection.
Fingerprint source files, resolved model revisions, configuration, code and lockfiles
before each run. Use scenario/family identities for split audits and confidence intervals.
Never silently truncate an assistant target or evidence; record overlength failures.

## Execution order

1. Hardware-only private Kaggle preflight, no project data or credentials in uploads.
2. Audit ADR-008 candidates and current compact multilingual alternatives against primary
   model cards. Pin full revision and licence before downloading weights. Compare KK
   tokens per word on identical development text, JSON validity and measured memory.
3. Local and remote pipeline smoke, <=1% training data and <=15 minutes. Loss and adapter
   loading must be observed. Failed, interrupted and unavailable attempts stay in receipts.
4. Zero-shot/few-shot baselines on validation. Few-shot demonstrations come only from
   train, with family overlap checks. Evaluate raw and validated/fallback output separately.
5. Named-control development experiments: ranks 8/16/32, LR 5e-5/1e-4/2e-4, target
   modules, sequence length and packing. Do not silently claim a full factorial sweep.
   Stop configurations that fail resource or schema gates; retain their failure evidence.
6. Compare an encoder classifier and the LLM router on all 35 labels. Report action
   accuracy, language-specific macro-F1, latency, throughput and measured GPU-hours.
7. Seal selected model/prompt/adapter/configuration and their hashes before loading test
   or gold. Test once per finalist; >=3 seeds when compute allows, with explicit limits.
8. Final safety/evidence study, performance, error census, restore drills and full checks.
   Commit and publish implementation/docs only; never upload local data to public Git.

## Evidence and metrics

Continuation plan before full-validation results and actual training are inspected:
`configs/llm_experiment.yaml` fixes candidate screening, OFAT pilot parameters, the
validation tuning subset, language/metric weights, KK guard, tie-breaks, final seal and
compute-dependent seed scope. The six-model 24-item audit is explicitly a development
screen, not final selection. Full validation compares the resource-feasible 0.6B control
and the stronger KK-language audit candidate Qwen3.5-2B; all six audit failures/results
remain in the report. Language conditioning uses the dataset language field, so reply
language results do not certify upstream language detection.

The encoder timing from the first run measured only the linear head; retain it as such.
The replacement report adds measured warm single-request encoding. No cached-feature
matrix multiplication timing is reported as end-to-end encoder latency.

Report RU, KK, EN and RU-KK independently, denominators and UNKNOWN for unobserved
slices. Raw JSON validity, intent accuracy/macro-F1, routing actions, escalation,
abstention, false-answer rate and false refusals are distinct from semantic quality.
Citation membership and digit matching are diagnostic checks, not semantic support.
Grounded answer accuracy, citation precision and unsupported-claim rate require explicit
claim/support judgments; unavailable calibrated judgments remain NOT_EVALUATED.
LLM authorship, draft labels and missing independent fluent KK review limit conclusions.

Pair predictions by stable item identity, resample 1,000 times with seed 20261113 at
scenario level, report 95% paired-bootstrap intervals and family-cluster sensitivity.
One family cannot establish between-family uncertainty. No improvement claim when
the delta interval covers zero.

Study complete, partial, irrelevant, contradictory and absent evidence separately.
Use fresh development evidence to change packing/gating; the inspected Stage 6.1
holdout remains exposed. Attribute missing required support to retrieval vs packing;
attribute unsupported generation separately. Candidate presence never proves sufficiency.
Safety requires >=50 scenarios per language covering injection, PII, live-state
fabrication, conflicts, ambiguity, out-of-scope and mandatory escalation.

## Delivery gates

Verify terminal Kaggle status for an explicit kernel version, download its outputs,
check hashes and source/config identity, import complete experiment receipts to MLflow.
Submission or live logs do not prove completion. Measure RTX 3050 inference separately
from joint retrieval residency; untested coexistence stays unverified. No paid resources.
Archive adapters, checkpoints, receipts and model card with data/base revisions; test
resume and actual backup restoration. Stage 8 begins only after owner confirmation.
