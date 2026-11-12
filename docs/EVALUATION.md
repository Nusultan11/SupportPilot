# EVALUATION — datasets, metrics, experiment protocol

## 1. Evaluation datasets

| Set | Built in | Use | Size target |
|---|---|---|---|
| validation | Stage 4 | model selection and early stopping | ~5k SFT examples |
| test | Stage 4 | final in-distribution report, used once per candidate | ~5k |
| **gold benchmark** | Stage 4 | independent, never used to train or tune | ≥ 300 verified scenarios per language |
| retrieval set | Stage 5–6 | queries with graded relevant `chunk_id`s, per language and cross-lingual | ≥ 150 queries per language |
| e2e scenarios | Stage 8 | multi-turn simulated conversations with tool states and expected outcomes | ≥ 100 per language |
| safety set | Stage 7–8 | prompt injection, PII probing, out-of-scope, conflicting context | ≥ 50 per language |

Gold benchmark rules:
- It is authored separately from the training pipeline: new phrasings, real-style typos, code-switching, ambiguous and adversarial cases.
- Each item has: lang, message(s), expected intent, action, escalate, required facts, the KB/tool state it depends on, and accepted citation ids.
- Verification: two reviews per item (author + independent reviewer). Kazakh and Russian items need a fluent human reviewer. LLM-only review is labelled `llm_reviewed` and reported separately; it is never described as "independently verified".
- It is versioned like datasets. Leakage checks against train run on every build.

## 2. Metrics

The independent retrieval set is now authored under `authored/retrieval/` and built with
`supportpilot data retrieval-set` (DATA_CONTRACTS §14). It contains 175 original information
needs in each language, plus 150 cross-lingual and 6 code-switched query variants. All review
statuses remain `draft`; automated checks and an author self-check are not human verification.

Retrieval protocol: filter the candidate corpus to each query's `corpus_lang`; compute
Recall@5, MRR@10 and nDCG@10 for answerable queries, by query language and corpus language.
For binary Recall/MRR use grade > 0; for nDCG use gain `2**grade - 1`. Report the grade >= 2
slice as a sensitivity check. Multi-fragment recall counts all positively judged chunks.
Unjudged chunks count as non-relevant, but qrels are a finite LLM-authored judgement set,
not exhaustive human assessment. Inspect judgements when a new KB version changes content.
Unanswerable questions have empty qrels: report their abstention/false-positive results
separately, never include them in Recall/MRR/nDCG denominators. Report code-switch and
cross-lingual slices separately and the semantic-near-SFT subset separately. Bootstrap by
`scenario_id` so translations and cross-language variants are not independent observations.

The data preparation task does not measure retrieval scores or select a model. That is
Stage 6 work. Keep gold reserved for final evaluation as specified below.

**Decision metrics.** These choose between options. Each is reported per language plus macro-average across languages.

| Area | Primary metric | Target (goal, not guarantee) |
|---|---|---|
| Data | leakage rate / post-cleaning validity | 0 / ≥ 99.5% |
| Intent routing | macro-F1 (gold) | ≥ 0.90 |
| Retrieval | Recall@5 (retrieval set) | ≥ 0.85 |
| Grounding | citation precision | ≥ 0.95 |
| Escalation | recall on critical scenarios | ≥ 0.95 |
| Generation | grounded answer accuracy (gold) | set after baselines in Stage 7 |
| Language | reply-language compliance | set after baselines |
| Latency | p95 end-to-end on target hardware | set in Stage 9 |

**Diagnostic metrics** (explain, do not decide): per-class P/R, confusion matrix, Recall@10,
MRR@10, nDCG@10, cross-lingual retrieval gap, faithfulness, unsupported-claim rate, answer
relevance, abstention accuracy, escalation precision, JSON validity, duplicate rate, label
agreement (Cohen's κ on a double-labelled sample), translation error rate (human-annotated
sample), p50 latency, throughput, RAM/VRAM, indexing speed, storage, training GPU-hours.

Definitions:
- **Citation precision** = cited chunks that actually support the claim / all cited chunks.
- **Unsupported claim rate** = answers containing ≥ 1 factual claim (number, date, price,
  status, policy condition) that is absent from the evidence / all answers.
- **Abstention accuracy** = correct abstain/no-abstain decisions on the abstention items
  (answerable and unanswerable mixed).
- **Language compliance** = replies in the expected language. Script and LID checks run first;
  KK vs RU uses KK-specific letters plus LID confidence, with a human sample audit.

## 3. Judges

- Deterministic checks first: schema, citation ids, number and date matching, LID.
- Semantic judgements (faithfulness, relevance, accuracy) use an LLM judge with a fixed
  rubric and a pinned model and version. The judge is calibrated against ≥ 100 human-labelled
  items per language. Agreement is reported, and the judge is not used for a language where
  κ < 0.6.
- A paid judge API needs user approval (cost). Otherwise use the strongest local or free model
  available, and say so in the report.

## 4. Experiment protocol

Every run is logged to MLflow (local backend, see RUNBOOK) with:
`hypothesis`, `control_run_id`, `changed_params`, `dataset_version`, `kb_version`,
`base_model@revision`, `seed`, `env` (python, torch, CUDA, GPU, package lock hash),
`metrics` per language, `cost` (GPU-hours, platform), `conclusion`, `decision` (adopt / reject / follow-up).

Rules:
1. Smoke test first: ≤ 1% of the data, ≤ 15 min, checks that the pipeline runs and loss decreases.
2. Change one factor at a time against a named control, unless the run is a planned sweep.
3. Do not choose a final configuration on test or gold. Test is run once per finalist. Gold is
   used for the final report and for go/no-go decisions.
4. For finalists, use ≥ 3 seeds where compute allows, and report mean ± std.
5. Report 95% bootstrap CIs (1,000 resamples) for decision metrics. Use paired bootstrap for
   A/B comparisons on the same items. A difference whose CI covers 0 is "no evidence of improvement".
6. Failed or negative experiments are logged and kept in the report.
7. Error analysis is mandatory when a target is missed: sample ≥ 50 errors per language,
   categorise them, and propose the top fixes by expected impact.

## 5. Planned comparisons

| Stage | Comparison |
|---|---|
| 4 | MT systems for EN→RU/KK on a human-checked sample (quality vs cost) |
| 6 | dense vs sparse vs hybrid; ≥ 3 embedding models; chunking strategy × size × overlap; top-k; RRF vs weighted fusion; rerank vs none; mono- vs cross-lingual |
| 7 | zero-shot vs few-shot vs LoRA/QLoRA per candidate model; rank {8, 16, 32}; LR {5e-5, 1e-4, 2e-4}; target modules; sequence length / packing; encoder classifier vs LLM router |
| 8–9 | end-to-end with and without fine-tuned model; quantised vs bf16 inference; latency/VRAM trade-offs |

## 6. Stage 6 executed retrieval protocol (2026-11-12)

The immutable inputs are `retrieval-20261112-ecc28bf4` (681 queries, 175 needs,
633 answerable/48 unanswerable) and `kb_demo_store-20261112-d58e3f93` (369 chunks).
All relevance judgements remain `draft`; reported quality is conditional on those
judgements. Gold is not loaded for model/parameter selection or retrieval evaluation.
The Stage 4 SFT validation/test sets are not used for retrieval tuning.

`sha256(seed|scenario_id)[:8] % 5 == 0` defines holdout; seed 20261112 gives 33 needs /
160 queries, versus 142 / 521 development. Translations and code-switched variants of
one need stay together. Ranking selection uses development metrics only: start from
the best minimum-language recall baseline; adopt a reranker only when its paired
development Recall or nDCG CI excludes zero and no mono language loses more than .005
Recall. Prefer the baseline otherwise. There is no stochastic fitting; one deterministic
inference configuration/seed was measured, not three independent training runs.

The matrix comprises 74 embedding/sparse/fusion/top-k/view configurations, seven
rerank/control variants and 15 whole-query score gates. Four encoders and two rerankers
use pinned weights and documented pooling/prefix protocols. Sparse Qdrant BM25-style
fixed-pivot normalization and BM25S are separate baselines, not numerically identical
implementations. Chunk-view comparisons use E5-base as the controlled encoder; this
does not establish a universal chunking optimum for every encoder. Context budgets
512/1024/2048 and section diversification are compared separately. Canonical qrels
never change: an adjacent view with two members consumes two canonical ranking slots.

**Protocol amendment:** the initial gate rule required only RU/KK/EN mono Recall >= .85.
It selected .02 but development cross-language Recall fell from .9744 to .6667. That
rule was rejected as unsuitable for the task. The amended rule additionally allows at
most .01 development Recall loss on cross-language and RU–KK slices. No positive gate
survives; the default gate is disabled. The earlier holdout had already been inspected:
its subsequent results and paired CIs are **exploratory**, not an untouched confirmation
set. `gate_selection_history.json` retains the amendment and initial result. No qrels,
query texts, split seed or gold labels were changed. A fresh independently reviewed
cohort is needed before a generalization or deployment claim.

Full/development/holdout Recall@5, MRR@10, nDCG@10 use 1,000 scenario-cluster bootstrap
resamples and 95% percentile CIs. Paired deltas reuse the same needs; a CI containing
zero provides no evidence of improvement. Mono language, query language, six directional
language pairs, cross-language, RU–KK, multi-fragment and semantic-near-SFT results are
separate. Empty-qrels queries have undefined ranking metrics, not zero recall; their
abstention/FPR have separate grouped CIs. Rejected answerable queries score zero.
Tiny slices (six mixed queries, six answerable semantic-near queries) cannot establish
language robustness. This is evidence retrieval, not generated-answer quality.

The final public API is also run online for all 681 queries, without query/pair caches,
with model loading separated from warm latency. Evaluate the *packed context* as well
as top-k rankings; metadata counts in the Qwen tokenizer budget. Store raw latency,
resource samples, model-load time, document encode speed, artifact/snapshot sizes and
incremental/no-work times. RSS is the Python process; GPU figures are PyTorch peaks,
not whole-machine RAM/VRAM. Docker and other applications are outside process RSS.
Concurrent LLM serving, sustained multi-user throughput and production scale are untested.

Artifacts: `artifacts/reports/retrieval/{comparison.md,summary.json,paired_comparisons.json,
context_comparison.json,online_context_metrics.json,performance.json,errors.json,
error_analysis.md,index_drill.json}`; complete CIs/observations per configuration under
`comparisons/`. MLflow runs include failed attempts and operational measurements.
When a model misses .85, analyse up to 50 distinct failed needs per language; if fewer
exist, report the entire available census and its size, without duplicating errors.
No automatic relevance-score threshold proves that a live order/status or missing
policy exception exists in the KB. Stage 7/8 grounding and routing remain necessary.
