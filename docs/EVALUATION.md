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
