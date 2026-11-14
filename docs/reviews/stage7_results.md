# Stage 7 measured comparison

Execution status is maintained only in PROGRESS.md. This report describes measurement scopes and results.

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
