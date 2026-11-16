# Stage 9 local-only quality experiment

Audience: the owner reviewing the remaining Stage 9 local research and its limits.
Preregistered 2026-11-16 before inference on these new inputs. No external reviewers,
helpers, contact, paid resources or Stage 10. Earlier experiment seals/results stay immutable.

## Router comparison

Control: frozen E5-small revision614241f622f53c4eeff9890bdc4f31cfecc418b3 and original
C10 head. The train-only reproduction is byte-value identical in weights. Intervention:
class_weight=balanced, C10/seed20261113/training rows/encoder unchanged. Training used
16,576 distinct utterance/intent pairs;114 mixed examples; class counts30..2096. This
imbalance motivates a hypothesis, not a causal conclusion. Both trained heads already exist
under private system/qualification/heads; they are not selected for serving.

New project-authored LLM draft needs: authored/system/local_quality.yaml. Four parallel
language forms share a need; one concrete subneed per split/class,35 needs per split.
This small screening cohort is not300-item/language gold,100 independent E2E needs or
50-need critical coverage. Distinct family labels cannot certify semantic independence.
Freeze source/protocol/head/runtime hashes before any inference. Audit normalized exact,
char5-gram Jaccard>=.8 and E5 cosine>=.95 against retained prior input surfaces and across
splits. Report every flag; no outcome-informed exclusion, relabelling or independence claim.
Do not reuse prior prediction outcomes to select anything.

<=1% execution smoke precedes full development. Use the same real frozen encoder,
FastAPI/SQLite fixtures and service49ba7db for both heads; raw encoder intent and service
intent/action/actual tool facts/queue/delivery remain separate. Generation stays off.
Mixed requests explicitly prefer KK; this does not qualify mixed detection/fluency.

Select only on new development: at least one positive paired delivery95% CI, no language
point loss, no individually lost required critical queue and no observed unauthorized
tool action. Raw all35 macro-F1 is reported but cannot override safety/delivery regression.
No reselection from confirmation. Seal the decision, both heads and exact runtime before
once-only confirmation of the paired comparison, including a rejected candidate for a
terminal negative assessment. A rejected candidate remains a research artifact.

Primary bootstrap seed20261116,1,000 scenario clusters plus family sensitivity. Retain raw
HTTP/errors, denominators, confusion/error census, resource/cost scope and MLflow runs.
Sparse all35 bootstrap percentiles are exploratory; critical all-success slices additionally
report Wilson intervals. Failures stay in denominators. No favorable-seed substitution;
deterministic repeats are not independent training evidence.

## Local semantic diagnostic

Eight authored fictional needs crossed with full, counterfactual, partial, irrelevant,
absent and conflicting evidence, four languages:192 diagnostic rows, eight independent
authored needs, not192 independent scenarios. Draft expected labels are self-authored
controlled-fixture hypotheses. They never become independent-human calibration or gold.
Compare a pinned local0.6B model's support judgement with always-unknown control; freeze
all prompts/labels before inference. One-row execution smoke (<1%) precedes the full run.
Use greedy NF4 inference, seed20261113, max64 output tokens,15s backend deadline. Retain
all invalid/truncated/error/timeout outputs as failed judgements. Report fixture agreement,
false-supported verdicts, abstention and automatic language diagnostics separately.
No automatic score is semantic citation precision or independent claim support. This
diagnostic cannot install a qualified verifier or enable customer factual answers.

## Acceptance and owner review

Preserve existing targets: intent>=.90, retrieval Recall@5>=.85, critical recall>=.95,
semantic citation precision>=.95 where methodologically measurable. Human semantic/fluency
qualification stays NOT_EVALUATED. Missing numerical grounded/language targets are not
chosen after observed confirmation. The owner receives the concrete measured package,
negative results and limitations before final acceptance. A generic owner confirmation
is acceptance of a declared delivery scope, not item-level human labels or proof of
passing a missed target. No COMPLETED status is inferred ahead of that acceptance.

Full applicable shared checks, backups/deep verification/actual restores, reviewed public
diff and origin/main publication complete the engineering increment. Do not rerun the
already terminal million-record workload or precision study without a new measured need.
