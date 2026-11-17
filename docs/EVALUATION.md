# EVALUATION — metrics and experiment rules

## 1. Datasets

| Set | Use |
|---|---|
| Validation | Model/prompt/parameter selection, early stopping |
| Test | Once per frozen finalist |
| Gold | Independently authored final assessment; never for training or tuning |
| Safety | Injection, PII, fabricated live state, conflicts, ambiguity, out-of-scope |

- Split by scenario/paraphrase family. Freeze and decontaminate against training.
- Count independent needs separately from rows and language slices.
- Stage 1 corpus (DATA_CONTRACTS §7): use `core` splits only; `unverified` rows are never
  targets. Report coverage with every metric: KK rows are translated out-of-scope requests only.
- Gold review levels: `draft` · `llm_reviewed` · `human_reviewed` · `verified`.
  `verified` needs an independent fluent human reviewer (RU/KK fluent for those languages).
  An LLM's own check is never independent review.
- Independent label review (ADR-049, `data review`): blind relabelling by a fluent reviewer who
  did not write the rows; a seeded ≥ 10 % (plan: 20 %) is relabelled by a second reviewer.
  Report per stratum: agreement with the corpus label (Wilson 95% CI), Cohen's kappa
  (< 0.6 → labels of that stratum are not `human_reviewed`), language-OK and natural rates,
  every disagreement. A stratum without a filled sheet stays `NOT_EVALUATED`.

## 2. Metrics

Report every metric per language: EN, RU, KK, mixed RU–KK. Always state the denominator.
Typical metrics: intent/action accuracy and macro-F1, escalation recall, abstention correctness,
citation support, language match, latency p50/p95. Define new metrics here before use.

## 3. Judges

1. Deterministic checks first: schema, references, numbers, dates, language ID.
2. A semantic judge uses a fixed rubric and a pinned model, calibrated on ≥ 100 human-labelled
   items per language. Kappa < 0.6 → do not use it for that language.
3. Without calibration or independent review, semantic quality is `NOT_EVALUATED`.
4. JSON validity, citation membership and number matching do not prove semantic support.
5. A paid judge needs owner approval.

## 4. Experiment protocol

Log every attempt in MLflow: hypothesis, control, changed parameters, data/model versions,
seed, code commit, per-language metrics, resources/cost, conclusion and decision.

1. Smoke first (≤ 1% of data / ≤ 15 min). It proves execution, not quality.
2. Change one factor against a named control.
3. Select on validation only. Seal finalists, then evaluate test/gold once.
4. ≥ 3 seeds when compute allows. Fix the primary seed before seeing results.
5. Uncertainty: 1,000 cluster-bootstrap resamples, 95% percentile CI.
   A delta CI containing zero is no evidence of improvement.
6. Keep failed, negative, timed-out and partial attempts with reasons.
7. Missed target → inspect up to 50 distinct failures per language.
