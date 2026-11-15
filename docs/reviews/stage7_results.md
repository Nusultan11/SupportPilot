# Stage 7 measured comparison

Execution status is maintained only in PROGRESS.md. This report describes measurement scopes and results.

Readiness: **NOT_READY for autonomous factual support or critical escalation**. Structural gains and verified experiment execution do not certify semantic quality.

## Validation

Frozen SFT validation: 313 rows (EN 97 / RU 103 / KK 113), including 22 overlapping RU–KK rows. Language is supplied to the model; this is not upstream language-ID accuracy. All values below are raw output diagnostics, not delivered fallback quality or semantic correctness.

|Configuration|Language|n|Schema %|Intent %|Action %|Language compliance %|
|---|---|---:|---:|---:|---:|---:|
|Qwen3-0.6B schema|en|97|41.24|4.12|18.56|39.18|
|Qwen3-0.6B schema|ru|103|32.04|6.80|8.74|30.10|
|Qwen3-0.6B schema|kk|113|31.86|6.19|7.96|30.09|
|Qwen3-0.6B schema|ru_kk|22|27.27|0.00|4.55|22.73|
|Qwen3.5-2B few-shot|en|97|63.92|28.87|45.36|61.86|
|Qwen3.5-2B few-shot|ru|103|67.96|29.13|53.40|66.99|
|Qwen3.5-2B few-shot|kk|113|58.41|30.97|46.02|54.87|
|Qwen3.5-2B few-shot|ru_kk|22|54.55|27.27|40.91|50.00|
|Qwen3-0.6B adapter schema|en|97|98.97|80.41|85.57|98.97|
|Qwen3-0.6B adapter schema|ru|103|98.06|78.64|86.41|98.06|
|Qwen3-0.6B adapter schema|kk|113|95.58|51.33|72.57|92.92|
|Qwen3-0.6B adapter schema|ru_kk|22|100.00|59.09|68.18|95.45|

Selection used the preregistered KK-heavy diagnostic score: adapter schema .83906 vs selected 2B base .51049. The adapter few-shot score .50402 fails the KK guard. Primary seed 20261113 was fixed before results; no favorable secondary seed was substituted.

Paired deltas below are percentage points, 1,000 seeded bootstrap resamples. The intervention control is the same 0.6B base/configuration/prompt/validation rows. Schema, prompts, inputs, Python/libraries and Kaggle image match; base-path source methods, adapter-integrity and unused constrained-code branches were audited separately.

|Language|Metric|Delta pp|Scenario 95% CI|Family sensitivity 95% CI|
|---|---|---:|---|---|
|en|schema_validity|57.73|[42.35, 73.12]|[44.66, 70.53]|
|en|intent_accuracy|76.29|[65.52, 86.21]|[64.65, 86.52]|
|en|action_accuracy|67.01|[49.99, 81.38]|[54.84, 78.02]|
|en|language_compliance|59.79|[45.65, 73.91]|[46.39, 71.91]|
|ru|schema_validity|66.02|[54.44, 77.23]|[54.64, 77.32]|
|ru|intent_accuracy|71.84|[58.33, 85.19]|[58.09, 83.81]|
|ru|action_accuracy|77.67|[64.15, 87.72]|[67.56, 87.62]|
|ru|language_compliance|67.96|[57.45, 78.23]|[56.56, 78.66]|
|kk|schema_validity|63.72|[49.12, 76.19]|[52.21, 75.22]|
|kk|intent_accuracy|45.13|[29.59, 59.21]|[33.04, 56.10]|
|kk|action_accuracy|64.60|[47.95, 78.69]|[52.38, 76.15]|
|kk|language_compliance|62.83|[49.16, 74.79]|[51.49, 74.34]|
|ru_kk|schema_validity|72.73|[44.43, 95.46]|[45.83, 100.00]|
|ru_kk|intent_accuracy|59.09|[30.42, 90.49]|[31.82, 90.91]|
|ru_kk|action_accuracy|63.64|[33.33, 92.01]|[31.82, 95.00]|
|ru_kk|language_compliance|72.73|[50.00, 92.01]|[50.00, 91.68]|

The stronger 2B deployment comparison changes the base model and prompt, so it is not a pure SFT effect. KK intent delta is +20.35pp but its primary scenario CI includes zero [0.00,38.85]; superiority on that metric is inconclusive. Other diagnostics and complete paired results are retained in `artifacts/reports/llm/selected_training.json`.

## Training and numerical scope

All-linear rank 16 / alpha 32, LR 1e-4, fp16 LoRA with FP32 trainable parameters, max_length 3072, gradient accumulation 4, no unsafe packing, MT weight0.5. Training includes 11,699 examples; 5,158 exclusions have reasons and targets are never truncated. Training uses the alternate KB, not active-store facts.

The primary seed actually stops at 256 and resumes checkpoint256 to 512 with optimizer/scheduler files and complete step history. Primary reference loss 2.23033→.05699; zero logged nonfinite gradient events, peak allocated2,825,739,776 bytes. Reference loss is a fixed training-sample diagnostic, not independent quality.

Secondary seeds 20261114/15 both reach 512 and each logs one NaN grad_norm (steps 387/124). All final safetensors are finite; optimizer skip counts were not directly instrumented and remain UNKNOWN. Their validation stability is reported with three-seed mean/sample standard deviation; these warnings are retained. Reference samples differ across seeds, so their absolute reference losses are not a controlled cross-seed comparison.

Raw adapters/checkpoints and the full218-file output inventory are verified against source/payload/output hashes, imported to MLflow and restore-verified. The inference export preserves exact weight/config/tokenizer bytes and excludes training_args.bin with recorded provenance.

## Routing and decoding

|Router|EN macro-F1|RU macro-F1|KK macro-F1|RU–KK macro-F1|
|---|---:|---:|---:|---:|
|Encoder E5-small C10|.53411|.53250|.38675|.16190|
|Qwen3.5-2B|.39324|.40980|.29828|.08102|
|Selected adapter, strict RouteDecision|.43987|.31286|.12123|.06571|

The encoder remains the frozen routing component. Current-message/empty-session inputs omit evidence/history sometimes needed by draft action labels. Raw route JSON, deterministic policy guards and safe fallback escalation have different evaluation scopes; none establishes autonomous critical-escalation readiness. Probabilities are not calibrated.

Actual local NF4 constrained-decoding probes on three validation rows yield1/3 valid AssistantResponse vs0/3 schema-prompt control. RU/KK constrained outputs remain invalid JSON; the sole valid EN output fails evidence diagnostics. This three-row compatibility diagnostic does not change the sealed configuration and does not establish multilingual JSON guarantees. Initial legacy smoke0/6 and all failed attempts remain preserved.

## Independent assessment boundary

Selection seal `570521fe81adaf13af962728b66f76da4f98e05aed8c47a47c891402699fa935` binds the adapter/schema primary seed, Qwen3.5-2B/few-shot control, validation evidence, source/configuration/schema/demonstrations, adapter manifests and encoder head before test/gold are opened. Final held-out predictions must never inform retuning.

Gold allowed citation sections do not define an AND-proof. Report reference coverage separately; gold retrieval/packing sufficiency remains null. Explicit support-group interventions reuse 18 exposed Stage 6.1 development needs, not fresh holdout. The supplemental conflicting-policy fixture has one artificial need and nine translated/intervened rows; one-cluster intervals cannot identify generalization uncertainty.

Grounded-answer accuracy, semantic citation precision, unsupported-claim and false-answer rates remain NOT_EVALUATED without calibrated independent judgments. Abstention/false-refusal values are labeled behavioral proxies. Gold/safety are LLM-authored drafts; independent fluent KK review is absent. Default generation returns safe escalation whenever semantic support is UNKNOWN; a valid schema or citation membership does not authorize customer-facing factual answers.

The frozen scorer's abstention proxy requires missing_information or needs_human **and no citations**. A legitimate refusal citing conflicting/partial evidence can fail that narrow proxy. Before inspecting final LLM completions, supplemental declared-action counts were specified irrespective of citations, including cited non-answer declarations. They supplement the original metrics without changing selection or scorer math and still do not establish semantic refusal/false-answer accuracy (`postprocess_scope.json`). Offline fallback-required/analytic delivered-abstention fields are policy implications, not observed Stage 8 API delivery; actual boundary evidence comes from local probes and integration tests.

The baseline/sweep/routing/selected archives, predictions, CIs, numerical audit, error census, MLflow runs and restored backups are local ignored artifacts. Public Git contains implementation, authored diagnostic source and aggregate documentation only. No Stage 8 implementation is introduced.

## Independent frozen encoder assessment

The selected C10 head is not refitted. Test contains 90/95/101 EN/RU/KK rows with all 35 intents per language; its 16 mixed rows overlap the language slices. Gold contains 312 rows per language and 59 overlapping mixed rows. Inputs remain current-message/empty-session with supplied language; action and escalation labels are conditional draft diagnostics.

|Cohort|EN macro-F1|RU macro-F1|KK macro-F1|RU–KK macro-F1|
|---|---:|---:|---:|---:|
|Test|.71047|.63844|.34668|.07143|
|Gold|.54107|.57066|.43001|.32943|
|Safety|.32298|.27856|.20944|UNKNOWN: no mixed rows|

Safety misses 22/26, 22/26 and 23/26 required escalations for EN/RU/KK. These observations prevent an autonomous-routing acceptance claim. The encoder's warm test p50 is 7.2–7.3 ms locally; remote LLM timings use different hardware and are not a same-device speedup comparison. Raw predictions and MLflow receipt: `artifacts/reports/llm/routing/heldout/`.

## Selected-adapter local resources

Actual local NF4 probes use the immutable 512-step adapter/schema prompt and the current retrieval pipeline. Each scope executes one query per language; both have zero execution errors and three observed model calls. In the joint scope, retrieval executes after the LLM is loaded, so both sets of models are resident during the actual request.

|Scope|Peak allocated VRAM bytes|Physical VRAM bytes|Observed calls|Delivered fallback|
|---|---:|---:|---:|---:|
|LLM isolated during generation|1,104,847,360|4,294,443,008|3|3|
|LLM plus active retrieval|3,376,806,912|4,294,443,008|3|3|

EN/KK raw responses fail schema; RU remains semantic UNKNOWN. All delivered responses are localized operator-assistance fallbacks. These three operational cases do not establish semantic quality, maximum-context residency or sustained serving SLO. Python RSS includes prior retrieval preparation even in the isolated scope. Local NF4 and remote fp16 are distinct conditions.

The adapter contains 40,422,168 weight bytes; its complete inference bundle is 51,853,699 bytes. The logical unquantized base weight file is 1,503,300,328 bytes. These are file sizes, not NF4 runtime allocation or allocated disk. The selected inference backup is deep-verified, restored, hash-identical and actually loaded as PeftModelForCausalLM in NF4. Full training checkpoints remain separate for resume.

Final LLM inputs are independently frozen: test 286, gold 936 (79 semantic-near-train), safety 150, sufficiency 144 and conflict 9. All 5,219 exported passages/ranking IDs match the immutable active KB/index; state files predate export. Gold reference coverage still does not establish sufficiency. Full identities and audit scopes are in `heldout_retrieval_identity.json` and `final_cohort_census.json`; no final LLM quality is inferred from submission or queue status.

An additional local fp16 diagnostic executes the same three operational queries per scope with the frozen model/schema. Both isolation and joint retrieval produce3/3 valid AssistantResponse objects without execution errors, all falling back on semantic UNKNOWN. Initial setup failed before LLM generation because Docker/Qdrant was stopped; that FAILED run with zero model observations is retained separately from the successful retries.

|Additional fp16 scope|Peak allocated bytes|Peak reserved bytes|Valid schema|Generation time range|
|---|---:|---:|---:|---|
|Isolated generation|1,757,544,448|1,828,716,544|3/3|10.08–20.97s|
|Active retrieval plus LLM|4,031,543,296|4,357,881,856|3/3|23.36–41.49s|

Reserved memory exceeds the physical4,294,443,008-byte budget; these successful requests do not prove entirely VRAM-resident serving. Allocation counters exclude other processes/driver memory, and shared-memory spill was not instrumented. The NF4 and fp16 runs share queries/formatter/immutable corpus but original NF4 prompt bytes were not captured, so this is an exploratory compatibility observation, not a causal quantization ablation or a full-validation precision guarantee. No frozen benchmark parameter changes. Audit: `artifacts/reports/llm/local_precision_diagnostics.json`.

The 2B FP16 control has a separate operational study on three already-exposed longest
eligible validation prompts (one per language), without test/gold. Two attempts failed at
the unchanged 6-GiB available-RAM guard before model loading. After temporarily stopping
this project's Qdrant/Docker services, all three calls completed with valid JSON:
EN/RU/KK generation24.42/27.81/33.07s and peak allocated4.839/4.857/4.903GB versus
physical4.294GB. Shared-memory residency was not instrumented; full-VRAM/full-suite feasibility
is NOT_ESTABLISHED. Services were restored healthy with unchanged collection names.
Failures, successful calls and service restoration remain in
`artifacts/reports/llm/local_control_fp16/` and three separate MLflow runs.
These resource probes do not change selection or establish semantic quality.

## Verified independent assessment

The explicit final kernel /2 completed; both workers exited 0. All 12 FINISHED tasks, 3,622 observations and source/payload hashes were verified before analysis. All 64 root-manifest members and both 30-member worker checksum manifests match. The original queued/cancelled /1 and empty archives remain separate. Readback/analysis receipts are in artifacts/reports/llm/assessment_verification.json and artifacts/kaggle/stage7_final_evaluation/output/2/.

This compares the sealed 0.6B adapter/schema deployment with 2B/few-shot. Both base model and prompt differ: it is not a pure SFT effect. Matched 0.6B development comparisons above remain the causal intervention control. No model, prompt, seed, router or metric was selected again from these outcomes.

|Cohort|Language|n|Base JSON %|Adapter JSON %|Base intent %|Adapter intent %|Base action %|Adapter action %|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|test|en|90|58.89|98.89|36.67|74.44|48.89|88.89|
|test|ru|95|67.37|98.95|34.74|71.58|54.74|85.26|
|test|kk|101|66.34|99.01|30.69|47.52|52.48|81.19|
|gold|en|312|80.77|91.99|35.26|52.24|54.49|61.22|
|gold|ru|312|79.81|92.95|33.01|44.55|52.56|62.82|
|gold|kk|312|78.53|94.55|27.56|35.90|49.68|62.82|
|safety|en|50|58.00|100.00|18.00|48.00|20.00|40.00|
|safety|ru|50|40.00|98.00|16.00|50.00|10.00|40.00|
|safety|kk|50|44.00|100.00|22.00|30.00|12.00|26.00|

Selected adapter RU–KK test n=16: JSON100%, intent50%, action81.25%; gold n=59: JSON94.92%, intent49.15%, action61.02%. Mixed rows overlap language slices. Safety has no mixed rows; mixed safety quality is UNKNOWN.

### Paired uncertainty

Intent deltas are percentage points,1,000 scenario-cluster resamples with family sensitivity.

|Cohort|Language|Intent delta pp|Scenario 95% CI pp|Family sensitivity 95% CI pp|
|---|---|---:|---|---|
|test|en|37.78|[21.11, 52.81]|[22.91, 50.00]|
|test|ru|36.84|[22.22, 53.20]|[25.25, 50.55]|
|test|kk|16.83|[0.00, 33.35]|[3.37, 29.47]|
|gold|en|16.99|[10.26, 24.04]|[1.94, 27.97]|
|gold|ru|11.54|[5.45, 18.27]|[-5.51, 26.18]|
|gold|kk|8.33|[1.92, 14.74]|[-10.58, 23.78]|

Schema improvement excludes zero per language under both resampling units on test/gold. KK test intent superiority is inconclusive under the primary scenario CI; gold RU/KK intent improvements are not robust to the coarse family sensitivity. Gold has only 9 category families, not independently reviewed paraphrase families.

KK gold intent macro-F1 is.22428 for the adapter vs.22443 for the 2B control; its paired scenario CI[-.05187,.06260] includes zero. Sparse 35-class cluster percentile F1 bootstrap can omit entire rare intent clusters: RU test point delta.33943 lies outside its scenario interval[.14246,.31239]. Keep the frozen values; these F1 intervals are exploratory calibration evidence, not a guarantee. Schema/accuracy intervals and raw points are unchanged.

Local plot/provenance: artifacts/reports/llm/independent_comparison.png/.svg and independent_comparison_manifest.json. Full paired metrics, mixed slices and families are in final_evaluation.json.

### Safety, evidence and failure attribution

|Cohort|Language|Required escalations|Base raw misses|Adapter raw misses|
|---|---|---:|---:|---:|
|test|en|10|6|6|
|test|ru|9|8|6|
|test|kk|10|8|9|
|gold|en|42|31|40|
|gold|ru|42|39|38|
|gold|kk|42|36|41|
|safety|en|26|17|16|
|safety|ru|26|20|18|
|safety|kk|26|22|24|

These are draft-label raw escalation failures, not measured harm rates: a refusal instead of the required escalation also counts as a miss. On safety, the adapter declares a non-answer action on 50/50 EN, 49/50 RU and 50/50 KK rows, with zero answer_kb declarations. This does not establish correct refusal: one inspected RU candidate requests a username/password. Safety abstention_accuracy has denominator 0 under the frozen task-label proxy, so remains UNKNOWN rather than 0 or 100%.

Sufficiency interventions reuse 18 exposed development needs (144 rows). Complete context has 9 rows/language, partial 3, and packed/absent/irrelevant12 each. In the natural packed subset, 3/12 per language have known proof lost after sufficient retrieval. Loss counters for absent/irrelevant/partial interventions are deliberately removed evidence, not an estimate of natural packing failure. Gold proof sufficiency remains unknown; allowed-reference coverage is reported separately.

All 36 absent-evidence adapter cases declare a non-answer. In partial/irrelevant contexts, however, no valid adapter completion declares a non-answer: EN partial 2/3 and irrelevant 10/12 answer_kb (others invalid); RU 3/3 and 12/12; KK 3/3 and 12/12. These declarations/known-proof diagnostics are not semantic false-answer rates. Deterministic citation/number errors and contract failures are retained separately; semantic generation errors remain NOT_EVALUATED.

The one-need conflict fixture produces valid adapter JSON in all 9 rows. Complete 11-day evidence yields 11-day answers, but equal-current 11/27-day conflict yields factual answer_kb in all 3 languages, inventing/merging conditions rather than rejecting the conflict. Absent contexts produce non-answer declarations; one KK raw candidate promises an operator handoff that has not occurred. This is one qualitative controlled need, not 9 independent scenarios; between-family uncertainty is UNKNOWN.

A targeted qualitative triage read 50 distinct failed needs/language, prioritizing required-escalation failures, with representative source checks. It is an LLM self-review, not random representative semantic scoring or fluent human review. IDs, sample texts, selection scope and findings: error_review_samples.json/error_review.json. Common patterns: explicit human requests map to account/contact templates; complaints/injury/fraud map to ordinary policy answers; the latest turn is overridden by earlier topic cues; missing/irrelevant evidence prompts generic order clarification. Some deterministic checks pass despite these failures.

Representative source-traced cases include EN g_es_safety_battery_en (packed stop-use/operator instructions ignored for charging advice), RU 2e822aa69114edf7 (credential solicitation with no evidence), RU/KK g_es_repeat_failure_* (unverified dialogue status promoted to current state), and KK g_x_es_safety_car_seat_kk (catalogue confusers instead of risk escalation). Retrieval/packing gaps, generation/intent errors and validation failures have distinct scopes; no semantic rate is inferred from these examples.

Gold flagged-near-train rows: EN 16/RU 21/KK 42, total 79. On unflagged rows, adapter intent accuracy is EN 52.36% (n=296), RU 43.30% (n=291), KK 33.70% (n=270); flagged values are 50%, 61.90%, 50%. The 857 unflagged rows are not certified contamination-free. Conditional tool fixtures and per-task slices remain separate in final_evaluation.json.

### Frozen routing assessment

|Language|Encoder macro-F1|2B LLM macro-F1|Adapter LLM macro-F1|2B raw→guarded misses|Adapter raw→guarded misses|
|---|---:|---:|---:|---|---|
|en|0.71047|0.49817|0.34808|10→4|9→5|
|ru|0.63844|0.42079|0.30204|9→3|8→1|
|kk|0.34668|0.32489|0.08483|10→4|10→5|
|ru_kk|0.07143|0.12857|0.04286|1→0|1→0|

The encoder remains selected. Raw and guarded LLM F1 are identical here; guards/fallback alter escalation outcomes. The LLM guard/fallback cannot be interpreted as calibrated model judgement. Current-message/empty-session draft action labels can require omitted history/evidence; supplied language does not test language detection. All 35 intents occur per test language; no routing configuration was refitted.

### Resources, provenance and readiness

Selected-adapter test p50/p95 generation seconds EN 7.91/9.71, RU 8.30/10.51, KK 11.84/15.91; gold EN 8.79/12.02, RU 9.76/14.64, KK 13.87/22.63. Generation throughput is about 12.4–12.5 tokens/s; loading and failed calls are separate. On gold, peak allocated/reserved CUDA bytes are 1,860,946,944/2,004,877,312 for the adapter versus 5,047,095,296/6,845,104,128 for the 2B control. These are Torch/process scopes, not whole-machine usage or local joint-serving guarantees.

Remote metadata requests NvidiaTeslaT4 shape with two successful CUDA workers; a separate final device-name sample was not recorded. Remote Python 3.13.16/Torch 2.11+cu130 differs from local Python 3.12/Torch 2.14.1+cu126. The pinned image/source and actual package versions are preserved in the exact-version archive. All 3,050 inference rows have no execution exception; both routing tasks completed 286 calls each without a propagated generation exception. Token caps remain failures/diagnostics, not discarded rows.

Final observed assigned-device process wall time 8.16972h; total measured across 68 retained attempts 17.71093h, with 25 older attempts lacking device-hour observations. Active-compute/session-allocated/billed hours remain UNKNOWN; paid compute invoked 0 USD, with local power/hardware and uninstrumented bootstrap/idle allocation excluded. Weekly account quota is not this cost metric. See cost_accounting.json.

Every final MLflow run has a verified assessment_identity.json with real cohort/input/selection/evidence fingerprints. Generic legacy importer parent-SFT/training-variant defaults are explicitly disambiguated; original receipts/metrics are unchanged. Validation reports, immutable adapter card/manifests and the selection seal still verify.

**Readiness:** research/inference/training/routing libraries are implemented and measured, but autonomous factual support and critical escalation are NOT_ACCEPTED. KK intent/generalization, raw escalation failures, missing calibrated claim verification/independent fluent review, and local joint FP16 residency prevent acceptance. UNKNOWN-to-localized-fallback remains mandatory; actual local boundary probes, not analytic heldout policy implications, establish that path. Do not switch to the baseline or retune from these results.

Future work requires new development/confirmation evidence: review/calibrate claim sufficiency; contrast escalation/privacy/human-request cases across languages; validate latest-turn handling and proof packing; measure sustained local authorized service. Keep exposed gold/test reserved as observed evidence. Stage 8 implementation requires separate owner authorization. Readiness decision and mutable external card: readiness.json/model_card.md.
