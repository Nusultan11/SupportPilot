# PROGRESS: current state (single source of truth for status)

Last updated: 2026-11-14 (Stage 7 selected-training execution)

## Current stage

**Stage 7: IN PROGRESS — verified baselines, OFAT pilots and routing; 512-step training running.**
Owner authorized the complete Stage 7 task and Kaggle CLI execution on 2026-11-13.
Stage 6.1 remains completed with its quality limitations below. Stage 8 is not started.

### Stage 7 execution checkpoint

#### Active continuation (2026-11-14)

- Gold-scope checkpoint **0b326bc** committed; all hooks pass. Local actual NF4
  constrained-decoding compatibility probe on three fixed validation items: ordinary
  schema prompt0/3 valid, constrained1/3 valid (EN). RU contains invalid control text;
  KK is incomplete JSON. No execution errors; EN constrained output still fails
  evidence diagnostics (unsupported number/answer without evidence). Do not treat
  enforcer support as a multilingual JSON or semantic guarantee. Frozen selection
  protocol remains unchanged. Receipts, raw outputs and MLflow runs under
  `artifacts/reports/llm/local_constraints/`; summary `constraint_compatibility.json`.
  Initial setup attempt failed before inference because RunSpec params were not scalar;
  failure log/receipt retained, config serialization fixed, two actual runs FINISHED.
- Validation automatic error census generated for all four baselines, per language/mixed,
  contract error types, intent confusions and up to50 distinct failed needs:
  `artifacts/reports/llm/validation_error_census.json`. No semantic/human-review claim.

- Published inference-export checkpoint **09df539c4d0de5d85d2a4f0f1c5df30579a47d26**;
  selected-training version1 still RUNNING on continuation. Before opening gold,
  corrected category aliases (`grounded`/`abstain`) and separated accepted-citation
  coverage from proof sufficiency: gold allowed references are not AND-proof groups.
  Explicit required-proof loss has a separate abstention scope; semantic quality remains
  NOT_EVALUATED. Three synthetic regressions plus selection/runner tests17 pass;
  Ruff and strict mypy pass. Full GPU/server suite **465 passed, zero skipped**,391.33s,
  one expected Qdrant warning; `artifacts/reports/llm/pytest_gold_scope.log`.
  No held-out results read. Supplemental conflict fixture frozen before final selection:
  one artificial policy need, three translations x complete/conflict/absent =9 rows,
  `data/benchmarks/llm_context_conflicts/`; no new independent-need claim.

- Prompt-boundary checkpoint **c25b4a80035cabbb624bc7ea07521e51b28cc726** published;
  all commit hooks pass, remote main verified. Full suite after that change:
  **460 passed, zero skipped**,378.10s, expected Qdrant warning, log
  `artifacts/reports/llm/pytest_prompt_boundary.log`.
- Real PEFT bundles include README.md and training_args.bin, which the previous
  inference upload allowlist rejected. Added immutable inference export: exact
  safetensors/config/tokenizer bytes, copied or supplied card, origin/exclusion hashes,
  no training pickle. Full training archive remains unchanged for resume. Packaging
  accepts model-card Markdown. Two regressions and existing related tests7 passed;
  Ruff/format and strict mypy pass. Export of the verified80-step all-linear pilot
  retains identical weight SHA256 and reloads as PeftModelForCausalLM on local NF4.
  MLflow receipt `artifacts/reports/llm/adapter_export_reload.json`; this is loader
  verification only, not512-step or quality acceptance. Full rerun after export support
  remains pending.
- Isolation scope: real test/gold artifacts have not entered Stage7 model runners,
  training or selection. Existing Stage4 hygiene/build tests do parse authored gold
  source YAML with fake ML components; no blind-to-author source claim is made.
  Scope audit `artifacts/reports/llm/isolation_scope.json`. Actual final inference/export
  remains gated by the selection seal; no holdout performance has been used for tuning.

- Runtime prompt parity: `build_messages`/`generate_grounded` now accept explicit
  schema/few-shot mode and frozen demonstrations through the shared formatter used
  by SFT/evaluation. Omitted mode preserves the legacy boundary and earlier smoke
  behavior. Two focused tests verify shared schema prefix, evidence isolation,
  few-shot forwarding/missing-demo failure and semantic UNKNOWN fallback; focused
  generation+runner suite17 passed. This does not change training or benchmark prompts
  in the already submitted selected-training package. Full rerun after this additive
  boundary change remains pending.
- Contract/integration checkpoint **417a7f4a3709c260627d17311182f0fbf568e80d** published
  and remote main verified; all commit hooks pass. Selected training remains RUNNING.

- Selected-training private Dataset ready and explicit kernel version1 RUNNING verified
  after the owner's direct upload/execution/retry approval. Remote source matches the
  submitted bootstrap; Docker image digest matches baseline. Inputs, prompts and
  AssistantResponse schema bytes match baseline; backend changes are adapter-integrity
  guards and inactive constrained-decoding compatibility, audited separately.
- Added2 runner integration tests: raw schema failure vs unsupported numeric candidate,
  retained raw observations/semantic UNKNOWN, byte-identical cached predictions, changed
  input and corrupted-prefix rejection. Both pass. Safety source header now records
  source_id/origin/eval-only use/licence note; all150 rows byte-identical after annotation.
  LLM/routing/artifact contracts and verified execution/recovery commands documented.
- Full current suite **458 passed, zero skipped**,395.07s, one expected local-Qdrant
  warning. Log `artifacts/reports/llm/pytest_server_completion.log`. Initial restricted
  run455 passed/3 skipped is retained in `pytest_continuation.log`: Docker was stopped
  and sandbox TCP was blocked. Docker/Qdrant restarted healthy; unrestricted localhost
  health200 and the full actual GPU/server rerun pass. Ruff/format and strict mypy pass
  (156 source files). This verifies engineering checks, not semantic quality or training.

- Published metadata checkpoint **98f6f5c630e2cd1778bf93aecf0d5ff0e3e4aa81**;
  remote main verified, all commit hooks pass. Owner explicitly authorized the prepared
  selected-training private upload, free execution and necessary fixes/retries on Nov14.
  Private Dataset created and status ready verified. Kernel submission started with
  NvidiaTeslaT4: `nurs2005/supportpilot-stage7-selected-training-run/1` RUNNING;
  terminal execution/results not yet verified. Do not repack this immutable
  payload into the same destination.
- Verified/restored backups: sweep375 files
  `data/stage7_sweep/20261114T024629Z-47e0f5f6`; prepared payload
  `data/stage7_selected_training/20261114T025543Z-9f7cc66e`; reports
  `data/stage7_reports/20261114T025543Z-76e1b565`; tracking2169 files
  `data/stage7_tracking/20261114T025550Z-8e29c53c` (restored SQLite integrity ok).
  Ledger `artifacts/reports/llm/pending_training_backups.json`.
  Code bundle `backups/code/supportpilot-20261114T025553Z.bundle` verified with SHA256.

- Sweep `nurs2005/supportpilot-stage7-sweep-run/1` COMPLETE. Every output byte,
  payload identity and remote kernel source verified. Download was interrupted after
  slow sequential IO, then four disjoint groups fetched124 missing/corrupt files;
  all group return codes0 and all SHA256 values verified. No experiment was rerun.
- Eighteen sweep attempts imported to MLflow:16 FINISHED,2 FAILED (unsafe eager
  packing and dependent adapter evaluation). Eight configurations trained80 steps.
  Preregistered weighted-score selection chooses **all_linear .70395**, control .59921;
  LR-high .69378, rank32 .64133. Pilot report/CIs: `artifacts/reports/llm/sweep.json`.
  All-linear includes11,699 training rows, excludes5,158 with explicit reasons;
  reference loss2.23033 ->1.17521, zero nonfinite gradient steps,
  peak allocated VRAM2,826,027,520 bytes. Tuning cohort124 (EN35/RU40/KK49),
  including22 mixed; do not compare its unpaired rates with full313-row baselines.
- Prepared private package `artifacts/kaggle/stage7_selected_training`:512-step
  all-linear rank16/LR1e-4, three seeds20261113/14/15, primary checkpoint256/resume512,
  schema/few-shot validation and adapter routing; only train/validation inputs.
  Automatic review rejected upload because prior permission named only the routing
  Dataset. Concrete permission requested for
  `nurs2005/supportpilot-stage7-selected-training`, including fixes/retries; owner approved,
  Dataset ready and kernel version1 RUNNING verified.
  No final selection seal. Test/gold remain closed; safety/sufficiency not evaluated.
- Published checkpoint **c7b9349c450646668ea2f1d88c557247b9871dd9**, remote main
  verified. Full pytest on that commit **455 passed, zero skipped**,420.12s,
  expected Qdrant warning; log `artifacts/reports/llm/pytest_current.log`.
  Subsequent MLflow execution/importer metadata separation passes focused regression,
  Ruff/format and strict mypy. Prior21 remote run tags corrected with before/after
  audit `artifacts/reports/llm/remote_metadata_repairs.json`; predictions, metrics and
  run statuses unchanged. Remote Git commit UNKNOWN; payload hashes authoritative.
- Restore-verified continuation snapshots:
  `data/stage7_routing/20261113T145422Z-746a5ded`,
  `data/stage7_reports/20261113T145422Z-26b1d904`.
  Code bundle `backups/code/supportpilot-20261113T151119Z.bundle` verified with SHA256.

#### Earlier continuation (2026-11-13)

- Routing2 COMPLETE; both313-row runs FINISHED, source/payload/output hashes verified
  and imported to MLflow. Comparison/paired scenario+family CIs:
  `artifacts/reports/llm/routing_comparison.json`. Intent macro-F1 EN/RU/KK/mixed:
  encoder .53411/.53250/.38675/.16190; Qwen3.5 .39324/.40980/.29828/.08102;
  Qwen3 .13314/.08116/.03678/.01905. Qwen3.5-minus-encoder scenario CI excludes
  zero for EN/mixed, includes zero for RU/KK. Encoder warm p50 ~8.3ms vs Qwen3.5
  ~3.0–3.5s (different local/Kaggle devices; no same-hardware speedup claim).
  Critical misses EN/RU/KK: encoder10/13,10/12,19/21; Qwen3.5 5/13,2/12,10/21.
  Neither router accepted for autonomous escalation decisions. Encoder predicted
  escalation reconstructed exactly from stored binary correctness and expected boolean,
  since its original report omitted that field; derivation recorded in comparison.
- Full pytest after routing import fix: **454 passed, zero skipped**, expected Qdrant
  warning,411.59s, log `artifacts/reports/llm/pytest_completion.log`. Subsequently
  added immutable packaging and missing-observation reporting guards: focused tests
  pass, full-stage verification still pending. Packaging refuses a changed payload in
  an existing destination and preserves identical ZIP bytes on replay.

- Published engineering checkpoint **fe50e7c2cf8c25eaab9c71c834e8fbd41e80b3cc**;
  remote main hash verified. All commit hooks pass. Code bundle
  `backups/code/supportpilot-stage7-checkpoint.bundle` verified; SHA256 sidecar retained.
- Baselines explicit version **1 COMPLETE**, source/payload/output hashes verified;
  four FINISHED runs imported to MLflow. Each configuration has all313 validation rows.
  Derived report/CIs: `artifacts/reports/llm/baselines.json`. Preregistered selection
  among bases favors Qwen3.5-2B few-shot (weighted diagnostic score .51049).
  EN/RU/KK raw schema validity .63918/.67961/.58407; intent accuracy
  .28866/.29126/.30973. Mixed22 schema validity .54545. Semantic quality remains
  NOT_EVALUATED; these results do not establish deployment acceptance.
  Baseline pinned deep-verified snapshot
  `data/stage7_baselines/20261113T141825Z-746892e2`.
- Routing explicit version1 submitted successfully after baseline freed a GPU slot.
  Routing1 COMPLETE but both tasks FAILED on an eager encoder import requiring
  qdrant_client. Encoder imports are now lazy; isolated subprocess regression passes.
  Both failures imported to MLflow. Original submission.json was overwritten during
  repackaging before import: evidence loss of original upload-ZIP SHA256 (UNKNOWN).
  Before any Dataset update, its unchanged private version1 archive was downloaded;
  every source/config/data byte matches the remote execution manifest. Recovery receipt:
  `artifacts/kaggle/stage7_routing/attempts/1/submission.json`.
  Automatic review rejected the updated-code upload despite the initial upload permission;
  owner explicitly authorized updates/retries in this chat. Dataset updated and routing
  version2 submitted successfully; RUNNING at latest check.
  Sweep1 remains RUNNING. No final model/adapter selection seal.
  Local Qwen3.5-2B NF4 executed one EN/RU/KK query each in isolation and with retrieval:
  no execution errors. Isolated allocated peak max2,364,117,504 bytes; generation
  8.14–9.25s. Joint allocated max4,636,914,688 / reserved max5,247,074,304 bytes;
  generation29.94–35.73s. Joint allocation exceeds physical4GB budget: do not claim
  fully VRAM-resident serving or SLO acceptance. Receipts under
  `artifacts/reports/llm/local_selected_base/`; isolated success under
  `20317c294533ff5186bd/`. Initial sandbox network failure, interrupted stalled Xet
  download and fingerprint-mismatch attempt retained; HTTP download then succeeded.
  Safety150 rows and sufficiency144 rows/18 related needs are frozen, not evaluated.

- Continuation verification: private training-smoke explicit version **3 COMPLETE**;
  downloaded result inventory, every checksum and submitted/remote kernel source match.
  Four attempt receipts imported to MLflow, all FINISHED (ledger
  `artifacts/kaggle/stage7_training_smoke/output/3/mlflow_import.json`).
  Checkpoint4/resume8 and adapter reload succeeded on 86 included training rows;
  step8 reference loss 2.23033 -> 2.07624, zero nonfinite gradient steps,
  peak allocated VRAM 2,701,483,520 bytes. This is engineering smoke, not quality acceptance.
  Versions1/2 remain retained failures (optional torchao/enforcer compatibility, then
  TRL rejecting DataParallel); version3 isolates one visible GPU per subprocess.
- Baselines version1 and OFAT sweep version1 remain RUNNING at continuation verification.
  No final selection/test/gold opening or completion claim. Owner explicitly authorized
  the concrete private routing upload in this chat. Dataset
  `nurs2005/supportpilot-stage7-routing` creation accepted. Kernel submission rejected
  by Kaggle's maximum two concurrent batch-GPU sessions; retry after an existing run ends.
  Automatic-review rejection before that authorization is preserved in the chat.
- Final selection seals now bind adapter manifest hashes and validate adapter bytes
  on reopening, rejecting weight replacement even with a refreshed manifest.
  Regression test passes. Ruff/format and strict mypy pass (153 source files).
  Full pytest: **450 passed, zero skipped**, one expected local-Qdrant warning,
  567.49 seconds. The subsequently added seal regression passes separately; targeted
  Stage7 suite **28 passed**. Restore-verified pinned continuation snapshots:
  smoke `data/stage7_training_smoke/20261113T134907Z-ba83e021`, reports
  `data/stage7_reports/20261113T134911Z-e596078d`; full snapshot ledger
  `artifacts/kaggle/stage7_checkpoint_backups.json`. Stage7 remains incomplete.

- Real Transformers backend, schema/few-shot prompts, constrained JSON support via
  lm-format-enforcer, completion-only TRL/PEFT training, input-bound resume and portable
  private Kaggle packaging are implemented in the working tree. PEFT 0.21.3, TRL 1.17.1,
  lm-format-enforcer 0.11.3 and scikit-learn 1.9.1 are locked. Full-stage checks pending.
- Development freeze `data/benchmarks/llm/9cbe1277cdde32b15f74` contains existing train
  and all 313 validation examples, scenario/family disjointness verified. It does not load
  test/gold. Three short train-only demonstrations are frozen. All labels remain draft.
- Public primary model cards/configs and full revisions of six Apache-2.0 candidates
  are retained in `artifacts/reports/llm/audit`; registry `configs/llm_models.yaml`.
  Farabi card access was denied anonymously; no claim about its current suitability.
- Private `nurs2005/supportpilot-stage7-audit-run/2` COMPLETE. All six models executed
  24 validation probes (eight tasks per language). Source/config/data payload and output
  hashes verified; nine attempts imported to MLflow, six FINISHED and three FAILED
  (two training attempts, dependent adapter load). Ledger:
  `artifacts/kaggle/stage7_audit/output/2/mlflow_import.json`.
  Audit is preliminary development evidence, not the full baseline comparison.
- Preserved failures: CLI relative-path Dataset upload; kernel/Dataset duplicate title
  rejected with HTTP409; audit version1 ERROR before inference because Kaggle expands ZIP
  inputs automatically. Version2 trained nothing: TRL1.17.1 rejects `warmup_ratio`.
  Replaced with supported `warmup_steps`, verified against the installed signature.
- Training smoke history is preserved by explicit version; see verified version3 above.
- Local RTX3050 NF4 Qwen3-0.6B inference and joint expanded retrieval each ran three
  real RU/KK/EN queries. Receipt `artifacts/reports/llm/local_resources.json` FINISHED;
  isolated peak allocated VRAM max869,581,824; joint max3,265,154,560 bytes.
  This establishes those probes only, not load/SLO acceptance or arbitrary model coexistence.
- Encoder classifier: pinned E5-small plus multinomial linear head, C=.1/1/10 on
  train/validation, selected C10. MLflow `27e59010e3344c72ab131ec99971b6c9` FINISHED.
  Validation macro-F1 EN .53411, RU .53250, KK .38675, mixed .16190 (10 intents in
  mixed, 35 per monolingual slice). Raw action accuracy EN .64948, RU .50485, KK .37168.
  Warm per-request encoder+head timing now included; earlier head-only timing report and
  run `ffcfdb2053c3465aaa195c7d09e22315` retained, never use it as end-to-end inference.
- Added `RouteDecision` and 50 authored semantic safety cases x3 languages (150 rows),
  `authored/safety/scenarios.yaml`, frozen `data/benchmarks/llm_safety/`. All LLM-authored
  draft; not training or tuning inputs. Safety inference and final evaluation not run.
- Current focused tests: 19 passed. Ruff and strict mypy for new modules pass after
  formatting. Full pytest/commit/push, model selection, actual training, final test/gold,
  CIs, model card and stage-end backups remain pending. No Stage8 work.

#### Earlier checkpoint at published commit 179030d

- Protocol: `docs/reviews/stage7_protocol.md`. Preserve existing SFT
  `sft-20261112-b0388c64`, training KB, gold and Stage 6.1 frozen artifacts.
- Kaggle CLI 2.2.5 authentication verified. Private hardware-only kernel
  `nurs2005/supportpilot-stage7-preflight/1` reached COMPLETE. Two Tesla T4 devices,
  each 15,636,037,632 bytes; CUDA matmul succeeded on each. Python 3.13.16,
  Torch 2.11.0+cu130, transformers 5.20.2, PEFT 0.21.2, accelerate 1.15.0.
  TRL, bitsandbytes and MLflow are absent. This is not the locked local Python 3.12
  environment, not proof of training compatibility or available future GPU quota.
- Downloaded explicit-version output and source/metadata under
  `artifacts/kaggle/stage7_preflight/`. `preflight.json` SHA256
  `bc149c243ab405630b49e4831422222b62524e69b2c77555da423fc9453a54c6`
  matches its downloaded sidecar. Remote metadata confirms private, no inputs,
  internet disabled and a pinned Docker image. No project data/credentials uploaded.
- Initial `llm/generation.py` boundary checks AssistantResponse schema, citation
  membership, unsupported numbers, response language and required escalation; retains
  raw completions and fallback reasons. Semantic support requires an explicit verifier
  verdict, default UNKNOWN/fallback. **No calibrated semantic verifier or real model
  backend is implemented yet; this is not an accepted working generation pipeline.**
- Initial targeted tests: 11 passed. Repository-wide Ruff, format and strict mypy
  pass (142 formatted files, 128 typed source files). Full pytest **431 passed,
  zero skipped**, one expected Qdrant local-mode warning, 407.89 seconds, including
  actual GPU/server tests. These verify the current boundary, not Stage 7 completion.
  Initial uv commands failed on sandbox cache access, escalated rerun worked;
  initial Ruff attempts found Cyrillic-string and compound-assertion lint errors.
- Model selection, actual baselines, training/resume, classifier comparison, safety set,
  paired CIs, RTX 3050 inference and joint retrieval residency are NOT_RUN.
  Semantic quality remains NOT_EVALUATED. No adapter selected or Stage 8 readiness claim.
- Preflight MLflow run `eecf03b90ab24b5b965e96c0d3f9d5bb` FINISHED. Downloaded
  source matches submitted source after documented LF/outer-whitespace normalization;
  source SHA256 `c17a379642d621b28473a38b5fbdee7dc924757c8fbfad7c2add7df67fef3c6e`.
  Snapshot `data/stage7_preflight/20261113T103127Z-f16280cc` is pinned,
  deep-verified and restored with all hashes verified.
- Private inference smoke `nurs2005/supportpilot-stage7-model-smoke/1` submitted:
  Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, fp16, six
  validation prompts (two per language), 512 max generated tokens, 900-second limit.
  Validation parquet SHA256 `1a989f060f9654f9a37850f24996b73c8e5eb692e3fa8721a1a78cb577479ff7`.
  No assistant targets enter prompts. Version 1 reached COMPLETE, receipt FINISHED,
  six predictions downloaded, all output hashes and normalized remote source verified.
  JSON parseability 2/2 per language; AssistantResponse validity **0/2 per language**
  (0/6 overall). Labels, enum values, field types and citations fail. This is a small
  negative prompt/pipeline diagnostic, not a complete baseline or model-selection result.
  T4 process peak allocated VRAM 1,659,389,440 bytes; loading 15.42 seconds, total
  script 70.81 seconds. These do not establish local RTX 3050 feasibility or session cost.
  MLflow `42cfea134e694e8c879e4681314fbc68` FINISHED; pinned snapshot
  `data/stage7_model_smoke/20261113T103707Z-ea042930` deep-verified and restored.
  First local import failed on Windows cp1251 decoding before MLflow opened; explicit
  UTF-8 fixed the importer. Raw outputs retained; semantic quality NOT_EVALUATED.
- Next: prepare pinned compatible remote training environment, audit model candidates,
  implement real backend/experiment runners and compare schema-aware zero/few-shot
  prompts on validation. Then training/resume, routing, safety and final evaluations.
  Use only validation for development. Test/gold have not been loaded in this work cycle.

## Stage 6.1 verified results

- Expanded active KB `kb_demo_store-20261113-f3b204a6`: **222 documents / 666 language
  versions / 8,865 chunks**, 2,600 explicit numeric product attributes. All13 KB gates
  pass, all369 original chunk IDs retained; rebuild unchanged. Serving namespace
  `scale_6fab32c16b3a6e05b29f`, control parameters unchanged. Real RU/KK/EN serving,
  empty-context abstention and baseline activation/rollback pass; final active KB expanded.
  Evidence: `artifacts/reports/retrieval_scale/6fab32c16b3a6e05b29f/serving/`.
- Freeze `data/benchmarks/retrieval_scale/6fab32c16b3a6e05b29f`: **3,570 needs /
  12,593 variants / 119 families**, 2,982 dev /588 holdout needs, 1,506 cross variants,
  377 mixed needs,830 unanswerable needs. Exact/lexical/ngram and semantic near-SFT
  leakage checks pass. 3,520 profile-derived plus50 manually authored needs remain
  correlated specifications, LLM-authored draft judgements, not human-certified dialogue
  diversity. No independent fluent KK review. See source registry/review.
- **17 development configurations; retained control** BGE-M3 dense .75 + word sparse,
  candidate20, BGE rerank20,top10,context1024,gate disabled. Alternatives fail cross guard
  or do not establish ranking gain. Selection sealed before single fresh family-disjoint
  holdout. Dev `1bdd4dd13dac4d85afdf8f9872210da8`, holdout
  `8cb7cbf9088949f98f239397a09a6830`, both FINISHED.
- **Holdout Recall@5 /95% scenario CI**: RU .90155 [.88083,.92098],
  KK .93264 [.91580,.94822], EN .97927 [.96891,.98834], cross .89506
  [.81482,.96914], mixed1.0 [1,1]. Overall recall .93466,MRR .97510,nDCG .95036,
  full evidence@10 .98678, packed sufficiency **.92438**. Multi-document packed
  sufficiency **.70297 [.64356,.75908]**, full@10 .96700.
- **Semantic abstention unresolved:** FPR **1.0 on606 holdout negatives /202 needs**,
  answerable FNR0. Gates trade cross recall for insufficient rejection. Cross/mixed have
  no holdout negatives: FPR UNKNOWN. Blind negatives cover partial evidence and missing
  exceptions; live-API/conflict/clarification/expired categories are development-only.
  This evaluates retrieval, not generated answers.
- Error census:161 misses@5,18 incomplete proof@10,103 incomplete packed contexts,
  **85 complete top10 proofs lost in packing** (overlapping labels). Source-bound audit
  retains121 failed needs; representative LLM qualitative review is
  `docs/reviews/stage6_scale_results.md`, not121 human judgements.
- Preplanned post-selection development-only wording/typo diagnostic:
  `data/benchmarks/retrieval_diagnostics/1ad2bbc278800f1f260c`,620 variants /80 existing
  needs /five families, **zero new independent needs**. Mono recall1.0, ENnDCG .96309
  [.93541,.98616],cross recall .83333 [.73333,.91687]. Different cross cohort, no paired
  drop claim, no tuning/fresh holdout claim. Run`fece36d08d3c41d98dcda4c217e1e2da` FINISHED.
- Same681 historical exposed queries: baseline p50/p95 **663.15/903.48ms**, expanded
  **687.57/889.45ms**, QPS1.458/1.447. Paired recall deltas RU/EN0,
  KK-.00613 [-.01840,0],cross-.02174 [-.04710,0], not confirmed improvement. Balanced80
  new dev probe p50/p95 **579.72/629.21ms**,QPS1.758 (different mix). No production SLO.
  Peak Python RSS5,839,515,648 bytes; Torch allocated/reserved2,301,659,136/2,334,130,176.
  Sampled Qdrant cgroup current2,144,235,520 includes file cache, not whole-VM RSS.
- Full uncached8,865-unit encoding each: BGE55.61 /E5 164.10units/s, warm weights.
  Minimum cosine .999901/.999968, max absolute drift .001618/.001104 from numerical/
  batch padding effects. Actual allocated collection disk baseline11,112,448 /
  expanded97,677,312 /100k489,938,944 bytes; preserve apparent `du -sb` separately.
  All80 cached/live probes have identical top10 and packed evidence.
- Semantic recovery: all8,865 dense/sparse vectors+payloads verified, max dense delta0;
  six serving evidence parity probes, sync12.945s without inference,upserts8865/deletes0.
  Profile`1cf54a1cd49e442b95d5b3c40dd353cb` and recovery
  `1cfcec4b69a34e2681ea9920fa4f489f` FINISHED.
- **100k isolated random vectors**, not semantic/neural-encoding evidence: upload62.338s,
  1,604.17points/s,HNSW61.567s,exact p50/p95 20.19/35.69ms. ef32/64/128 fidelity
  .163/.294/.466: keep exact default. Snapshot489,516,032 bytes; all100k payload/vector
  digests agree after restore/restart. Run`064d70bed19640d3be2ac759db82e11f` FINISHED.
  Real128-vector update/rollback .132913/.104280s, full digest unchanged,
  run`7e634079e5be40f19d3ebf1b00a9cc61`. Generator/2 reproduces vectors with separate
  identity. Current1000smoke`722dd27c44bc4d3d9d7b970326cd5174` verifies finalized local/
  MLflow receipt byte identity; do not rewrite the original100k receipt limitation.
- Preserved failures: numeric builds99 signed-temp then204 punctuation/money placements;
  fixed with regressions. First failed full report overwritten before capture: explicit
  evidence loss. Old freeze`e5d93adef95a2d2a128e`, run`0b22092bfb9342d6920f61c5e4882a69`
  KILLED before selection/holdout for CRLF fingerprints; >=6000 searches lost, no completed
  metrics claimed. LF/cohort equivalence receipts preserved; checkpoints fingerprint-bound.
- Final full check: **420 passed, zero skipped, one expected local-Qdrant warning**,
  434.35 seconds, including actual GPU/Qdrant tests. Log:
  `artifacts/reports/retrieval_scale_pytest_completion.log`. Ruff check/format and strict
  mypy pass (125 files). All implementation commit hooks pass, including the fast-test subset.
  Earlier interrupted/modified-during-run hook is retained as a failed attempt.
- Artifacts: `artifacts/reports/retrieval_scale/6fab32c16b3a6e05b29f/`.
  Baseline556 report files/369chunks/681queries verified unchanged in
  `artifacts/reports/retrieval_scale/baseline_integrity.json`.
- Existing pinned/deep-verified/restored snapshots: KB
  `data/stage6_scale_kb/20261113T051522Z-89ce6640`; earlier benchmark
  `data/stage6_scale_benchmark/20261113T051529Z-09f9c315`; load
  `data/stage6_scale_load/20261113T052952Z-ce8b17de`; indexes
  `data/stage6_scale_indexes/20261113T071321Z-b1407eff`. Delivery coverage is listed below.
- Final source/report audit passes:43 frozen source files,8 final MLflow report artifacts;
  report run`d49e79f234e6468aa1b1a73a3010f0e6` FINISHED. All41 tracked frozen files
  match committed Git blobs. All1,925 historical MLflow artifact files unchanged.
- Implementation commit **`6d5a6bdb98bc1e6e95753bd05ff4f0402f37149c`** is published;
  `git ls-remote origin refs/heads/main` verified that exact hash. Delivery metadata is
  recorded below; final publication and the documentation bundle are recorded
  in `artifacts/reports/retrieval_scale_delivery_receipt.json`.
- At Stage 6.1 delivery, Stage 7 awaited owner confirmation; authorization and its
  current execution checkpoint are now recorded above.
  User-owned untracked `.codex/` remains excluded from Git.

### Delivery snapshots (pinned, deep-verified and restored)

| Source | Snapshot under backups/ | Files |
|---|---|---:|
| `data/kb/demo_store` | `data/stage6_scale_serving_kb/20261113T095818Z-a3297f93` | 2155 |
| `data/benchmarks` | `data/stage6_scale_benchmarks/20261113T095831Z-bc6f6f5a` | 43 |
| `data/indexes` | `data/stage6_scale_index_cache/20261113T095838Z-c388fee1` | 110 |
| `data/scale_cache` | `data/stage6_scale_pair_cache/20261113T095848Z-5e2d0375` | 1 |
| `data/load` | `data/stage6_scale_load_delivery/20261113T095855Z-4e60ddbb` | 12 |
| `mlruns` | `data/stage6_scale_tracking/20261113T095910Z-325b2608` | 1982 |

All restored files match hashes; restored SQLite integrity checks pass. Full archive
SHA256 and restore paths: `artifacts/reports/retrieval_scale/delivery_snapshots.json`.
Full report snapshot: **`data/stage6_scale_reports/20261113T101036Z-b3a3b907`**,151 files,
pinned, deep-verified and all restored hashes verified. Archive SHA256:
`c011623294ef1ff4df04650beed8ba0c17f6c2bdeb7d944bf640a59a02683a3d`.
Includes selected/full development observations, holdout, source errors, load snapshots,
performance/recovery/diagnostics, failed attempts and finished check/commit logs.
Code bundle **`backups/code/supportpilot-20261113T101031Z.bundle`**, SHA256
`5c91a47b34b84b188a089cae63731540f832a48b6d9f728f2ba2d233aae1a337`, verified and
cloned separately at implementation HEAD. Final documentation HEAD gets its own verified
bundle and publication receipt. Data snapshots were taken from the verified working
implementation before commit; all41 tracked frozen inputs match implementation Git blobs.

Final publication receipt lives **outside** the report archive to avoid self-reference:
`artifacts/reports/retrieval_scale_delivery_receipt.json`. It records final HEAD, both
publication checks and code-bundle verification, with a checksum sidecar. Local datasets,
reports, models, MLflow, backups and user `.codex/` never enter public Git.

Completion decision: requested volume/diversity audits, comparisons and CIs, explicit
negative results, full tests and real restore evidence are present. No measured semantic
quality improvement over baseline is claimed. Correlated synthetic templates, draft qrels,
absent fluent review, incomplete blind negative coverage and FPR1.0 remain limitations
for Stage7–8, not silently passed quality gates.

### Large-cohort expansion history (before final freeze)

- `kb_sources/demo_store/technical_profiles.yaml` explicitly defines **200 products in
  20 groups and 2,600 numeric attribute facts**. Values are authored fictional data,
  not random or arithmetically scaled records. An explicit SKU registry keeps identifiers
  stable under insertion, removal and reordering. `kb/profiles.py` adds separate catalogue
  documents without rewriting the original catalogue or its chunk IDs.
- Verified large KB **`kb_demo_store-20261113-c74969ed`**: **218 document IDs, 654 language
  versions, 8,841 chunks**, all 13 KB gates pass, all original 369 chunk IDs retained.
  Registry count 3,685 includes metadata/composite entries; exactly 2,600 new numeric
  attribute facts are separately identifiable. Active serving KB remains the Stage 6 baseline.
- Profile-derived sources have explicit subject/predicate/condition need keys; wording and
  translations do not add needs. The first large audit has **3,538 needs**, 810 unanswerable,
  1,494 cross-language variants and 377 mixed needs; **115 related families**, with 2,953
  development and 585 holdout needs. Exact/lexical/13-gram overlap with SFT and across phases
  is zero. These are controlled specification needs and draft judgements, not 3,538
  independently human-reviewed natural customer scenarios. Family sensitivity remains required.
- Added dated archived payment/cancellation/pickup/repair documents and 12 authored temporal
  needs, including explicit two-version comparisons. `effective_to`, successor overlap
  checks, per-need `as_of` and opt-in `all_versions` retrieval are implemented; the default
  still filters validity dates. Context source metadata now carries validity dates. These
  latest additions need a new KB build and regression checks before freeze.
- The profile authoring step now also adds 20 needs outside document validity. Rerun
  `retrieval scale --stage author` after schema/template edits, then audit the new KB.
- Two KB build failures were encountered and fixed: unsigned matching rejected negative
  temperatures (99 fact placements), then a signed-token guard rejected sentence-ending
  grouped money (204 placements). Numeric matching now checks signs, token boundaries and
  decimals; eight targeted regressions pass. Failure receipts are in
  `artifacts/reports/retrieval_scale/failures/`. The first full failed report was overwritten
  by the existing builder's reused `.failed` path before capture; this evidence loss is
  explicitly recorded, not reported as a pass.
- No new ranking experiment or final holdout evaluation has run. Next: validate temporal
  ingestion and source-authoring transforms; renew the bound substantive-diversity review;
  run semantic decontamination, freeze, development comparison, sealed selection and holdout.
  Preserve all Stage 6 data/results; do not start Stage 7.

### Initial engineering and load-test evidence

- Preregistered protocol: `configs/retrieval_scale.yaml`,
  `docs/reviews/stage6_scale_protocol.md`. Family-disjoint development/holdout assignment,
  mandatory volume/lineage/leakage gates, immutable checksummed freeze and separate
  required-fact support groups. A rejected freeze does not publish a benchmark.
- Initial original authored additions: four trilingual documents, 24 SKU/procedure facts;
  `authored/retrieval_scale/` has 38 needs, 72 cross-language variants and five mixed needs,
  including ten unanswerable needs. All are LLM-authored `draft`, not human-reviewed.
  No claim of 2,500 independent needs or independently validated semantic diversity.
- KB `kb_demo_store-20261113-896f2b9d`: 18 document IDs / 54 language versions / 441 chunks;
  all 13 standard KB gates pass, original 369 chunks retained, 72 added, zero deleted.
  Audit counts 185 non-identifier canonical registry entries (including composite facts),
  not 2,500 distinct atomic facts. All earlier fact values and gold evidence IDs remain.
  This version is **staged**: active KB is restored to `kb_demo_store-20261112-d58e3f93`,
  so the last-good serving index stays compatible. Audit the staged version explicitly.
- `retrieval scale --stage audit|freeze|development|holdout|load` is separate from Stage 6's
  historical experiment commands. Index namespaces preserve baseline collection names and
  local state. The comparison matrix/selection path is implemented; its real large-scale
  semantic experiment is NOT_RUN because data gates fail. No final holdout was evaluated.
- Real isolated load corpus: 100,000 chunks and normalized random 1,024-dimensional vectors,
  identity `9264c39f794d3e16de74`, MLflow `064d70bed19640d3be2ac759db82e11f` (FINISHED).
  Precomputed upload 62.338 s / 1,604.17 vectors/s; HNSW build 61.567 s, 100,000 indexed.
  Exact p50/p95 20.187/35.690 ms; sequential 43.43 requests/s, 100 independent random queries.
  HNSW ef32/64/128 ANN Recall@10 against exact .163/.294/.466; p95 27.952/30.391/31.962 ms.
  This negative fidelity result does not justify replacing the exact semantic-search default.
  Python peak RSS 1,316,110,336 bytes, minimum available machine RAM 4,267,974,656 bytes;
  no model inference/VRAM allocated. Docker peak RAM, neural encoding/reranking, concurrency
  and production scale are not established by this workload.
- Load lifecycle: injected committed-batch interruption/replay, add/change/delete, full
  payload/vector inventory and real local snapshot upload to a separate collection passed;
  six exact query rankings agree. Snapshot 489,516,032 bytes, retained in
  `artifacts/reports/retrieval_scale/load/9264c39f794d3e16de74-1794503491694431100/`.
  A 1,000-point smoke run also FINISHED (`4fe1620c1f55405ea9daa80757e3dc3e`). The injected
  exception is not an OS kill. Failed/partial attempts must remain in their own receipts.
- Current generator `16fe088123cb8e192cc1` adds the text template to its identity;
  `generation_reproduction.json` verifies byte-identical 100k chunk/vector files to the
  measured `9264c39f794d3e16de74` corpus. The historical run is not relabelled.
  Docker restarted after session interruption; collection storage is 829,444,863 bytes.
  A later container sample was 1.523 GiB of a 2 GiB limit, including other collections;
  it is neither peak RAM nor a collection-specific delta.
- Pending before completion: substantive 200-document/7,500-chunk/2,500-fact corpus;
  2,500 independent needs including 750 unanswerable and 300 mixed needs; final frozen
  benchmark, real development selection/holdout comparisons, online packed-context
  performance and semantic abstention study, full checks and verified stage-end backups.
  Do not substitute generated repetitions or lower volume/quality gates to close the stage.

### Historical Stage 6 status

**Stage 6: COMPLETED — retrieval engineering and experiments, with the quality limits below.**
Implementation, measurements, checks and local restores are verified.
Semantic answerability is an explicit unresolved quality limit, not a passed safety gate.
The current datasets and follow-up results are below; older Stage 4/5 tables are historical.

| Stage | Status | Commits |
|---|---|---|
| 0 Audit and research | COMPLETED | `699dc23`, `cd50a6c`, `ced287e` |
| 1 Engineering foundation | COMPLETED | `6b17404`, `7b6bc4c` |
| 2 Data acquisition and source registry | COMPLETED | `a7bcc05`, `4384d13`, `f43dec0` |
| 3 EDA and data quality engineering | COMPLETED | `b2f5d86`, `809b47b` |
| 4 Multilingual dataset engineering | COMPLETED (gaps documented) | `da28191`, `6e5c537` |
| 5 Knowledge base and document extraction | COMPLETED | `c79d8af`, plus a follow-up progress commit |
| 6 Retrieval and RAG experiments | COMPLETED (limits below) | `326d223` + stage-end report (this change) |

## Stage 6 — retrieval implementation and measurements (2026-11-12; ADR-030)

**Implemented:** `src/supportpilot/retrieval/` exposes typed library and CLI evidence
retrieval through Qdrant: dense/sparse/hybrid, RRF/weighted fusion, two pinned rerankers,
canonical/window/adjacent views, corpus-language/document/type/effective-date filters,
whole-chunk token-budgeted contexts and explicit empty-evidence reasons. Original
`chunk_id`, KB/index/model version, source path and PDF/table/section locators survive.
This stage does not generate customer answers, train an LLM or implement an application.

**Index:** verified immutable artifacts, checksummed SQLite inference batches, OS writer
locks, bounded retries and fail-closed readiness. Real parent rollback and forward
`changes.json`: 90 deletes, 171 upserts, 198 vectors retained; unchanged rerun 0/0 in
.104 s. The selected collection contains exactly 369 live chunks. Internal Qdrant
tombstones exposed global-IDF drift after restore; KB-scoped IDF fixed it, and all nine
prepared searches/comparisons were recomputed. A downloaded 7,687,680-byte snapshot was
uploaded into a separate collection: all 369 payloads and named-vector hashes match,
plus six dense/sparse query checks. This is a real local-file restore, not just counting
points in the original collection. Process-kill and interrupted-batch tests cover resume.
Metadata-only validity/type changes refresh payload filters without re-encoding; tested
for date/type changes, rollback and real-server payload repair. No-work detection checks
the complete expected payload inventory, not just point IDs and KB tags.

**Data/protocol:** unchanged KB `kb_demo_store-20261112-d58e3f93`, retrieval
`retrieval-20261112-ecc28bf4`, 681 queries / 175 needs (633 answerable, 48 unanswerable).
Scenario-group development 142 needs/521 queries; holdout 33/160. All qrels stay `draft`.
Gold 936 and SFT 17,456 artifacts deep-verify unchanged; gold is not a tuning input.
One deterministic inference/seed, 1,000 grouped bootstrap resamples, per-language and
directional-language-pair metrics, graded nDCG and grade>=2 sensitivity; paired CIs.
The confidence rule was amended after the initial holdout was inspected. Final holdout
is **exploratory**, not an untouched confirmation set; human independent review is pending.

**Experiments/choice:** 96 variants (74 dense/sparse/fusion/top-k/view, seven rerank/control,
15 query gates), four encoders (E5-small/base, BGE-M3, Qwen3-Embedding-0.6B), two rerankers
(BGE-v2-m3, Qwen3-0.6B), BM25S reference, word/prefix features, k=10/20/40, weights
.25/.5/.75, five controlled E5 chunk views, context 512/1024/2048 +/- diversification.
Licences, full Hub revisions, protocols and card checksums are retained; a newer gated
Gemma candidate was screened but not acquired. Final: BGE-M3 + word sparse, weighted
fusion .75, 20 candidates/channel, BGE rerank 20, up to 10 results, 1024 Qwen tokens,
no diversification or score gate. `configs/base.yaml` and Python defaults agree.

Top-k retrieval, full answerable set, mean [95% scenario-bootstrap CI]:

| Slice | n | Recall@5 | MRR@10 | nDCG@10 |
|---|---:|---|---|---|
| RU mono | 163 | .9898 [.9785,.9980] | .9790 [.9616,.9939] | .9753 [.9608,.9875] |
| KK mono | 163 | .9678 [.9453,.9857] | .9692 [.9455,.9886] | .9616 [.9426,.9776] |
| EN mono | 163 | .9760 [.9560,.9923] | .9709 [.9468,.9898] | .9661 [.9437,.9833] |
| Cross-lingual | 138 | .9420 [.8768,.9928] | .9361 [.8646,.9964] | .9424 [.8833,.9909] |
| RU–KK code-switch | 6 | 1.000 [1,1] | 1.000 [1,1] | 1.000 [1,1] |
| Multi-fragment | 66 | .9621 [.9167,.9931] | 1.000 [1,1] | .9658 [.9365,.9863] |
| Semantic-near-SFT, answerable | 6 | 1.000 [1,1] | 1.000 [1,1] | 1.000 [1,1] |

Mono Recall exceeds the .85 goal separately in all three languages. These conditional
figures do not establish human relevance or production robustness. Six mixed queries
and three semantic-near needs are especially small cohorts. CIs containing zero in
paired comparisons are not evidence of improvement. All development/holdout/full CIs
and six directional pairs are retained, not averaged away.

**Online packed-context verification:** all 681 requests through the public API, no
query-vector/pair-score caches. Recall RU .9898 / KK .9647 / EN .9760 / cross .9420;
KK CI [.9412,.9826]. Context packing slightly reduces KK support coverage; all mono
targets remain met. Mono macro Recall .9778 [.9632,.9898] before packing and .9768
[.9615,.9894] online (three equally sized language cohorts). Four packed rankings differ
from cached inference; per-slice
metrics agree (retained in the diagnostic receipt), so byte-identical ranking is not
claimed. Context includes source metadata in its 1024-token budget, never partial facts.

**Performance on the RTX 3050 Laptop:** 681 sequential warm requests, batch one:
p50 **296.96 ms**, p95 **841.34 ms**, **2.181 requests/s**; cold first request **16.934 s**,
model loading 12.146 s. Peak Python RSS **5,152,636,928 bytes** (4.80 GiB), PyTorch VRAM
**2,302,613,504 bytes** allocated / **2,334,130,176** reserved; minimum machine-available
RAM 3,692,515,328 bytes during this run. Document re-encoding: 369 / 9.140 s = **40.37/s**,
vectors reproduce exactly. A fresh isolated collection upload of 369 precomputed vectors,
sparse payloads and full inventory verification takes .807 s (457.5 units/s); this excludes
model loading/encoding and is not an end-to-end cold indexing rate.
Portable index files 2,123,262 bytes; Qdrant snapshot
7,687,680 bytes. Earlier 105-query runs were faster and are not substituted for this
sustained profile. An earlier startup approached RAM pressure; close other model jobs.
Process RSS excludes Docker/other applications; LLM co-residency, concurrency, CPU-only
performance and large-scale SLOs remain unverified.

**Negative/failed results and limits:** E5-small/base KK Recall .6411/.7459; Qwen dense
cross .7029. Prefix sparse helps mono morphology but not translation. BGE reranking
improves development nDCG by .0339 [.0209,.0467] versus the selected fusion baseline;
against the matched k20 control the delta is .0335 [.0209,.0466]. K20 versus k10
reranking is not statistically established (.0012 nDCG [-.0019,.0053]); k20 follows the
development minimum-language recall rule, not a claimed significant win over k10.
Qwen is slower and k20 cross recall is lower. Qwen batch 8 adds no meaningful speed;
16 is slower. Per-chunk pruning loses supporting evidence. A .02 query gate retained
mono recall but reduced cross recall to .6051; rejected under the multilingual guard.
**Default unanswerable FPR = 1.0 (48/48)**; no tested positive threshold preserves
the cross-language recall budget. Semantic abstention is unsolved. Similarity must not
be treated as permission to invent a missing policy/live status. Empty filtered results,
expired policies, insufficient context and configured confidence rejection return no
evidence; downstream routing/claim grounding are necessary before customer-facing use.
No independent human review, LLM grounding evaluation or production acceptance is claimed.

Error analysis retains 50 distinct KK failures for each E5 baseline, up to 50 sparse
failures and the complete available selected residual census (RU 4, KK 10, EN 7 mono;
cross 12 query failures / four needs). Judgements were not edited. Failed Windows-lock,
weight/cache setup, interrupted Qwen inference, Qdrant-outage and global-IDF restore
attempts remain in artifacts and MLflow as failures/UNKNOWN, not silent omissions.
API cost USD 0; GPU hours across the interrupted/pause interval are UNKNOWN.

**Artifacts/reproduce:** `artifacts/reports/retrieval/comparison.md`, `summary.json`,
`comparisons/*/{metrics,observations}.json`, `paired_comparisons.json`,
`context_comparison.json`, `online_context_metrics.json`, `performance.json`,
`selection_evidence.json`, `gate_selection_history.json`, `error_analysis.md`,
`error_census.json`, `input_integrity.json`, `index_drill.json`, `mlflow_audit.json`.
Selected portable index `data/indexes/retrieval/c589a0db1c53a5a87322/versions/rindex-20261112-88e36128/`;
caches and other candidates stay under `data/indexes/`. Experiment `supportpilot/retrieval`
uses local `mlruns/mlflow.db` and artifacts. See RUNBOOK Stage 6 commands and ADR-030.
**Checks:** locked environment sync, Ruff lint/format and strict mypy pass;
full pytest **367 passed, 0 skipped** (113.56 s), including two GPU and three real-Qdrant
tests. The final metadata-repair server test also passed separately. One local-Qdrant
exact-search warning is expected. Initial hook line-ending fixes were applied; all
commit hooks then passed, including strict typing and the fast pytest subset.
Pinned index backup `retrieval_indexes/20261112T162836Z-722c87d7` deep-verifies; restore
into an empty directory verifies 84 files, ten immutable indexes, six SQLite databases
and the portable Qdrant snapshot hash.

**Stage-end provenance and backups:** implementation **`326d223`**; final delivery
MLflow run **`84384dd3cb5a441bbced7d223026aa19`**, from that clean implementation commit.
Frozen report **`retrieval_report-20261112-f18b0247`**, manifest checks all **556 members**;
its source fingerprints cover configuration, lock, retrieval code and tests. Historical
index/run manifests keep their original `58e7c45-dirty` build provenance; they are not
relabelled retroactively. The delivery records the later metadata-only correction;
encoding/search/rerank/packing measurements remain unchanged.

All three snapshots are pinned, deep-verified and restored into separate empty paths:

| Backup under `backups/` | Snapshot | Files | Restore verification |
|---|---|---:|---|
| `data/retrieval_indexes/` | `20261112T162836Z-722c87d7` | 84 | 10 index manifests, 6 SQLite caches, snapshot SHA |
| `data/retrieval_reports/` | `20261112T163654Z-e2fdce6d` | 557 | report manifest + all 556 member hashes |
| `data/retrieval_mlflow/` | `20261112T163650Z-7b0cc4b0` | 1,926 | SQLite integrity; delivery run FINISHED; artifact hashes |

Restored MLflow contains 257 FINISHED and nine FAILED runs across the existing project
history; the current retrieval matrix has 96 audited configuration runs. Index snapshot
SHA256 `90ff9e3554de9337d0b95189198cdd56a27a7d6cf453cbbab0be3f2c38b81676`;
restored named-vector digest `9ca16f9f4cf351446517028eff7c3a717b4d8b91ec4fc9d933888df03c766976`.
Code bundle: `backups/code/supportpilot-20261112T163941Z.bundle` plus `.sha256`;
bundle verification and a separate clone check cover the stage-end `main` commit.
Publication/backup receipt: `artifacts/reports/retrieval_delivery_receipt.json` (outside
the frozen report to avoid a self-referencing archive). Runtime data, models, reports,
backups and pre-existing untracked `.codex/` are excluded from Git/publication.

**Completion decision:** Stage 6 ROADMAP deliverables are implemented and measured:
chosen multilingual pipeline, conditional language targets, comparison CIs, resumable
incremental index, ADR, tests and verified restores. This closes the retrieval experiment
stage; it does **not** accept semantic abstention or customer-facing RAG quality. Keep
draft relevance status, the exploratory holdout warning and downstream grounding/routing
requirements. Stage 7 requires a later user confirmation.

**Next stage:** Stage 7 is not started. Stop here; a later user confirmation is required.

## Multilingual data expansion (2026-11-12; ADR-028, ADR-029)

**Status: COMPLETED for the requested data preparation scope.** No retrieval model, vector
index, RAG orchestration or fine-tuning was implemented. Human fluency/relevance verification
remains pending; this is readiness for the planned experiments, not production validation.

- Authoring: **300 situations**, up from 106; 194 new situations, not slot copies.
  Six additional scenario files cover orders, delivery, payment, returns, products and
  account/meta cases. All section answers have at least two wording variants per language.
- External inputs: pinned mangi public sample and MASSIVE EN/RU (CC-BY-4.0, source registry
  and attribution preserved). Native selection: 876 rows (EN 372 / RU 373 / KK 131), before
  final SFT decontamination/balancing. Original generation method of mangi questions is unknown.
- SFT **`sft-20261112-b0388c64`**: 17,456 examples; **16,857 train / 313 validation /
  286 test**. Every one of 35 intents exists in every language in each split, after filters.
  Train EN 4,857 / RU 6,333 / KK 5,667 (28.81% / 37.57% / 33.62% vs 30/35/35 target).
  Authored/original/MT train counts: 1,810 / 4,774 / 10,273; MT share 60.94%.
  Train scenario clusters: 350; held-out: 39 each. Counts include source clusters and must
  not be confused with the 300 independent authored situations.
- Train tasks: grounded 6,153; tool 1,559; clarify 4,782; escalate 1,258; abstain 1,024;
  conflicting 244; injection 909; OOS 928 (previously 71). Code-switch: 114/22/16;
  multi-turn: 180/6/24 (train/validation/test). Full per-language task statistics in `stats.json`.
- Copies: 702 train examples excluded to protect held-out scenarios at the unchanged
  thresholds; one external greeting excluded to reserve gold. No gold target was used in SFT.
  Final exact/lexical/13-gram/id checks pass, including validation vs test.
- Demo KB **`kb_demo_store-20261112-d58e3f93`**: 14 documents x 3 languages, **369 chunks**, 261 canonical facts,
  1,059 fact placements/locators, all 13 gates pass. Four new documents and six similar
  products; existing fact values preserved. Forced render/parse finds all 42 files unchanged.
  Training KB: **1,950 chunks**, all variants distinct from demo facts.
- Gold **`gold-20261112-306dfb08`**: all 936 items preserved, authored source YAML unchanged;
  facts/evidence/language and prohibited overlaps all pass. Semantic-near-SFT flags remain
  diagnostic, and all review statuses stay `draft`.
- Retrieval **`retrieval-20261112-ecc28bf4`**: **175 questions per language + 150 cross-lingual + 6 RU-KK =
  681 queries**, from 175 information needs. Graded active-KB chunk judgements and TREC
  qrels; all categories, reference/manifest/language/PII/leakage checks pass. Six distinct
  query texts are semantic-near-SFT (10 variants flagged); report that slice separately.
  Gold was not an authoring input. Scores Recall@5/MRR@10/nDCG@10 are Stage 6 work.
- RU/KK author self-check: 100 new independent inputs per language. Before repairs:
  RU 0 major/2 minor; KK 1 major/5 minor. Distinct-city generation and language issues
  repaired with a 100-seed x 3-language regression. Evidence:
  `docs/reviews/data_expansion_review.yaml`; raw sample/annotations in local reports.
  This is an LLM self-check, not human verification or a post-repair population error rate.
  Historical MT estimates (RU 3/60; KK 11/55 major) retain their original scope and risk.

**Checks and reproducibility:** Ruff lint/format and strict mypy pass; full pytest **343
passed, 0 skipped**, including GPU and Docker Qdrant. Seven raw/converted sources deep-verify.
Forced SFT rebuild: same version and byte-identical train/validation/test, statistics and
quality report; unchanged rerun skips. KB forced parse: same version. Snapshot restore into
empty paths reproduced SFT bytes, all 13 KB gates and valid retrieval references. No new
dependencies; existing locked environment reused. A fresh dependency install was not repeated.

**Local artifact paths:** `data/processed/sft/sft-20261112-b0388c64/`, `data/processed/kb_train/`,
`data/kb/demo_store/`, `data/gold/gold-20261112-306dfb08/`, `data/benchmarks/retrieval/retrieval-20261112-ecc28bf4/`.
Complete receipts and statistics: `artifacts/reports/data_expansion/summary.json`,
`reproducibility.json`, `backups.json`, logs and the annotated language sample.
Builds/snapshots precede the code commit and record a dirty parent; the completion Git
commit contains their sources and verified pipeline code. The immutable KB's manifest is
retained, rather than relabelled retroactively. Code-commit/bundle evidence is recorded below.

**Pinned data backups** (`backups/data/`, originals in the earlier checkout preserved):
- `mangi_kk_support_qa_raw/20261112T111114Z-5e6ca0f3` (3 files; deep-verified).
- `massive_en_ru_raw/20261112T111114Z-c9bab9eb` (7 files; deep-verified).
- `clean_records/20261112T111115Z-238129f8` (14 files; deep-verified).
- `canonical/20261112T111115Z-d5d50397` (7 files; deep-verified).
- `stage4_interim/20261112T111116Z-2b975042` (84 files; deep-verified).
- `kb_train/20261112T111116Z-453adf00` (2 files; deep-verified).
- `sft/20261112T111116Z-19991bcb` (6 files; deep-verified).
- `gold/20261112T111116Z-d512ffb1` (3 files; deep-verified).
- `kb_demo_store/20261112T111117Z-56a67b84` (133 files; deep-verified).
- `retrieval/20261112T111656Z-38d447f2` (21 files; deep-verified).

The completed checkout and data are synchronised to the original local SupportPilot
directory after Git verification; no processed datasets, backups or weights are tracked.
No authored-data redistribution licence has been invented. Independent human RU/KK and
qrels review remain open, along with the historical MT-quality limitations.

**Expansion Git/code backup:** implementation commit `893a7c74cf08c6fa54a22f57fa7ada04f25e9c5a`;
all commit hooks pass. Verified code bundle: `backups/code/supportpilot-20261112T111926Z.bundle`
with adjacent SHA256. The following documentation commit records the final evidence.

**Local installation:** synchronised to `C:/Users/nurs/SupportPilot`; clean `main`,
SFT rerun skips, and SFT/gold/retrieval/KB manifests verify there. The initial push was
rejected by automatic approval review; the owner then explicitly approved publication of
source code and authored YAML. Implementation/installation commits were pushed to `origin/main`.
Processed datasets, backups and model weights remain local and git-ignored.

**Installed-artifact recheck (2026-11-12):** the requested expansion was already present
on `main`; no dataset content was regenerated or changed. Reverified five output manifests,
35 intents in each language in every SFT split, zero scenario/family overlap across splits,
all 681 retrieval queries and 845 qrels against the active KB and authored query sources.
Existing demo facts and gold source YAML remain unchanged from Stage 5; current SFT bytes
match the saved forced-rebuild SHA256 receipts. Seven raw/converted sources, Stage 3 gates,
all 13 KB gates and deep SFT/KB/retrieval snapshot checks pass. Forced KB build reparsed all
42 documents without a version change; the SFT rerun skips unchanged inputs.
Ruff lint/format and strict mypy pass; full pytest: **343 passed, 0 skipped** (93.86 s,
including GPU and Qdrant). Initial sandbox/cache and nested-temp test issues, followed by
the successful normal-temp rerun, are recorded in
`artifacts/reports/data_expansion/recheck_checks.json`; content checks are in `recheck.json`
with the repeatable `recheck.py` script. Pinned report snapshot:
`data_expansion_reports/20261112T113236Z-65884674` (33 files).
RUNBOOK §7c now follows ADR-029: preserve gold and exclude copied SFT inputs on leakage.
Human RU/KK and relevance review remain pending; gold/retrieval stay `draft`.

## What exists (cumulative)

| Component | Path | Notes |
|---|---|---|
| Foundation (Stage 1) | `src/supportpilot/common/`, `ops/`, `cli.py`, `configs/base.yaml`, `compose.yaml` | config, logging/redaction, IDs, hashing, manifests, contracts, MLflow, backups, checks |
| Source registry | `configs/sources.yaml` | 16 sources reviewed; licence policy enforced by the schema (ADR-018) |
| Usage policy | `data/policy.py` | acquire/convert/analysis ≥ reference_only, evaluation ≥ eval_only, training = train |
| Acquisition, conversion (Stage 2) | `data/acquire.py`, `converters.py`, `convert.py`, `records.py` | pinned, hash-checked raw data → `RawRecord` Parquet (ADR-019) |
| Cleaning, labels v0, dedup, gates, EDA (Stage 3) | `data/textnorm.py`, `placeholders.py`, `langid.py`, `clean.py`, `labels.py`, `neardup.py`, `build.py`, `gates.py`, `eda.py`, `report.py`, `review.py` | ADR-020..022; unchanged in Stage 4 except that its gate now uses `INTENTS_V0` |
| Taxonomy v1 (Stage 4) | `configs/taxonomy.yaml`, `data/taxonomy.py`, `data/select.py` | 35 intents, actions, conditions, rule-split labels (ADR-023) |
| Localisation, MT, QE (Stage 4) | `configs/locale_kz.yaml`, `data/locale.py`, `data/mt.py`, `data/mtrun.py`, `data/qe.py`, `data/embed.py` | KZT/cities/names, resumable MT cache, glossary, QE filter (ADR-024) |
| Training KB, answers, scenarios (Stage 4) | `authored/kb_train/`, `authored/responses/`, `authored/scenarios/`, `data/kbtrain.py`, `data/textfmt.py`, `data/scenarios.py` | LLM-agent-authored, native en/ru/kk (ADR-025) |
| Demo-store KB sources (Stage 5) | `kb_sources/demo_store/{facts,catalog}.yaml`, `docs/*.yaml` | canonical values once + 10 document specs in en/ru/kk (ADR-027); training KB must differ |
| KB pipeline (Stage 5) | `kb/spec.py`, `render.py`, `writers.py`, `parse_pdf.py`, `parse_office.py`, `parse_text.py`, `blocks.py`, `structure.py`, `chunk.py`, `checks.py`, `locate.py`, `register.py`, `build.py`, `cli.py`; `common.schemas` KBDocument/KBChunk | render -> parse -> structure -> chunk -> 13 gates -> versioned KB `data/kb/demo_store/` |
| SFT build, splits, gates (Stage 4) | `common/grounding.py`, `data/sft.py`, `data/sftbuild.py`, `data/split.py`, `data/sftgates.py`, `data/steps.py`, `data/stage4.py` | `sft/1` + `AssistantResponse` contracts, leakage checks (ADR-026) |
| Gold benchmark (Stage 4) | `authored/gold/`, `data/gold.py` | 312 scenarios × en/ru/kk, `gold/1`, review status per item |
| CLI | `supportpilot data …`, `supportpilot kb …` | Stage 3 commands + `taxonomy`, `select`, `kb-train`, `mt-compare`, `translate`, `qe-score`, `sft`, `gold` (RUNBOOK §7c); `kb render/build/validate/status/activate` (RUNBOOK §7d) |

## Stage 5 results (2026-11-12)

### Knowledge base content (ADR-027)
- Fictional store Altyn Sauda (KZT, Kazakhstan). Canonical values are written once:
  `facts.yaml` (policy values; Stage 5 added `sms_retry_seconds`, `cart_days`,
  `payment_retry_minutes`) and `catalog.yaml` (30 products with demo prices, 11 categories,
  12 delivery-zone cities). No real customer data; no real store.
- 10 documents x en/ru/kk = 30 files. The text is written natively per language by the
  Stage 5 agent (LLM; human review pending). Values enter only through placeholders, so every
  language states the same policy.

  | doc_id | type | format | sections / records |
  |---|---|---|---|
  | returns_policy | policy | PDF (table) | 7 |
  | payment_policy | policy | PDF (table) | 5 |
  | delivery_policy | policy | DOCX (table) | 5 |
  | orders_policy | policy | DOCX | 4 |
  | terms | terms | DOCX (level-2 sections) | 8 |
  | warranty_policy | policy | TXT | 4 |
  | account_faq | faq | TXT (Q:/A:) | 9 |
  | store_info | faq | JSON (FAQ export) | 6 |
  | catalog | catalog | XLSX | 30 products |
  | delivery_zones | policy | CSV | 12 cities |

- Every section the gold benchmark cites (34 `doc.section` references, e.g.
  `returns_policy.window`) exists in all three languages (`evidence` gate), and
  `evidence_map.json` resolves them to chunk ids.

### Parser choice (measured, ADR-027)
| candidate | table rows exact | word / fact recall | decision |
|---|---|---|---|
| pdfplumber 0.11.10 (MIT) | 1.000 | 1.000 / 1.000 | **chosen** (PDF) |
| PyMuPDF 2.1 (AGPL-3.0) | 0.389 | 1.000 / 1.000 | rejected (tables, licence) |
| pypdf 7.0.1 (BSD) | no tables | 1.000 / 1.000 | rejected |
| python-docx 1.2.0 (MIT) | 1.000 | 1.000 / 1.000 | **chosen** (styles give heading levels) |
| docx2python 3.7.2 (MIT) | 1.000 | 1.000 / 1.000 | rejected (no styles) |
| openpyxl 3.1.5 (MIT) | records 1.000 | integers 90/90 as int | **chosen** (XLSX) |
| python-calamine 0.9.1 (MIT) | records 1.000 | integers 0/90 (floats) | rejected |

Measured on 6 rendered PDFs + 3 multi-page stress PDFs, 9 DOCX and 3 XLSX. Docling was not run
(torch layout models, and there is no scanned input). Results:
`artifacts/reports/kb/parser_comparison/`.

### Build `kb_demo_store-20261112-bcae8093` (13 gates, all pass)
| gate | result |
|---|---|
| facts | 912 / 912 placements found in the chunk of their section (304 per language: money 78, periods 46, identifiers 47, dates/hours 4, condition lists 3, text 126) |
| locators | 912 / 912: the source file, re-read at the chunk locator, contains the fact |
| coverage | 221 / 221 canonical facts placed in every language |
| completeness | word recall 1.0 and 0 extra words in all 30 documents (running headers, footers and page numbers removed) |
| cross_lingual | 96 / 96 translation groups have en/ru/kk with equal numbers and fact keys |
| sections_linked | 270 / 270 sections and records found once; none unlinked |
| language | 198 / 198 prose chunks: LID = document language |
| metadata, sources, ids, pii, tokens, evidence | pass (max chunk 280 Qwen3 tokens, p50 81; budget 400) |

- 288 chunks: catalog 90, policy 126, faq 45, terms 27 (96 per language).
- **Reproducibility.** Re-rendering gives identical sha256 for all 30 files. Deleting
  `data/kb/` and rebuilding gives byte-identical `chunks.parquet`, `documents.parquet`,
  `evidence_map.json` and `changes.json` and the same `kb_version`, which is also the version
  of a build from a copy of the sources in another directory. A `--force` re-parse of all 30
  documents reports `unchanged` (6.4 s, peak RSS 269 MiB).

### Verification evidence (2026-11-12)
- `ruff check`, `ruff format --check` and `mypy --strict` are clean; pre-commit passes;
  full `pytest`: **317 passed, 0 skipped** (Qdrant server up).
- **New tests: 40** (unit 33, integration 7). They cover:
  - spec validation and split YAML values; RU plural and KK case surfaces; effective-date
    lines in 3 languages x 12 months; byte-deterministic writers for all 6 formats;
  - parsers: PDF headings, tables, page furniture, paragraphs across pages, a 70-row table
    split over pages, a scanned page -> error; DOCX styles and levels; XLSX header detection
    and integer cells; CSV delimiter sniffing and multi-line rows; TXT setext, Q:/A: and pipe
    tables; JSON FAQ ids and generic JSON;
  - structure and chunking: FAQ = Q+A, records keep header, key and sheet, long sections and
    tables split with the heading / header repeated; every gate on passing and failing input;
  - **every document in every compatible format and language (84 combinations): 100% of
    facts, words and locators**;
  - the versioned build: unchanged re-run, identical forced rebuild, a changed fact that
    re-parses only its 3 files and replaces only their chunk ids, removed and added documents,
    failed gates that keep the current version, a damaged version that cannot be activated,
    rollback, a reverted edit that reuses the stored version, and the CLI round trip.
- **Bugs found and fixed during the stage:**
  - Stage 4's `facts.yaml` lost the house number of the RU/KK showroom address (an unquoted
    comma in a YAML flow mapping). The cross-lingual gate found it, the Stage 4 session fixed
    it, and the KB loader now rejects split values.
  - A changed ground truth with unchanged content skipped the gates. Re-saving a spec with CRLF
    could overwrite a stored version of identical content. The version identity is now the
    content plus the ground truth.
  - The table of a section that fits the budget went into a separate chunk; the sheet name was
    expected in CSV text.

## Stage 5 limitations and open items

1. **Synthetic, clean layouts.** The parsers are proven on this generator's documents
   (born-digital PDF, styled DOCX, plain XLSX). Real scans, merged cells or multi-column PDFs are
   not covered; a scanned page raises an error instead of being skipped. Revisit with real
   documents (OCR / Docling evaluation).
2. **The RU/KK texts are LLM-authored.** Fluent human review is pending (open decision 2), as
   in Stage 4.
3. **One effective period.** Every document has `effective_from` = the facts date and no
   `effective_to`. Superseded versions live in older KB versions, not as dated chunks in the
   current one. Future-dated policies are not modelled yet.
4. **In-memory build.** Fine for this KB (2.4 MB); a KB larger than memory needs a streaming build.
5. The retrieval query set (EVALUATION §1, ≥ 150 queries per language) is Stage 6 work.

## Stage 4 results (2026-11-12)

### Taxonomy v1 and relabelling (ADR-023)
- 35 intents (v0's 27 + `change_address`, `product_defect`, `contact_info`, `personal_data`,
  `newsletter`, `cart_help`, `app_help`, `technical_issue`), 10 escalation reasons, 4 tools.
- Input: Stage 3 canonical (55,648) + pending `intent_unmapped` (11,494) EN rows of train sources.
  Relabelled 65,550; excluded 1,592: label excluded 1,268 (`check_cancellation_fee`, `pay` with
  "transfer"), no rule matched 320 (`customer_service` without explicit wording), family split
  across labels 4. `label_conflict`/`label_suspect` rows stay out. while-ai: excluded entirely.
- EN backbone: one row per family, ≤ 260 per source label+action, profanity ≤ 10%:
  21,190 rows. MT queue: 100 rows per label+action and language = 7,554 RU + 7,554 KK.

### MT comparison and translation (ADR-024, `docs/reviews/stage4_mt_review.yaml`)
Major-error rate on 50 reviewed items per system (LLM reviewer, not human), and the
automatic QE accept rate on the 100-item sample (QE v1):

| system | RU major | KK major | RU auto-accept | KK auto-accept |
|---|---|---|---|---|
| Opus-MT en-ru / en-trk | 52% | 92% | 84% | 36% |
| M2M100-418M | 50% | 100% | 84% | 2% |
| MADLAD-400-3B nf4 | **40%** | **34%** | 80% | 69% |
| MADLAD pivot en→ru→kk | – | 52% | – | 69% |

- Automatic QE alone would have picked the wrong RU system and accepts many wrong KK
  outputs: the decision rests on the reviewed sample.
- Production (MADLAD, RTX 3050, nf4, 2 beams): KK 6,486 new + 7,141 back-translations in
  4,354 s + 4,090 s; RU 7,532 in 6,878 s, Opus-MT ru-en back-translation 372 s.
- QE (v2, with the `latin_word` rule added after the first KK audit): accepted RU 5,511 / 7,554
  (73%), KK 5,098 / 7,554 (67%). Top KK rejections: back-translation chrF < 0.30 1,637,
  lower-case Latin typo copied 510, untranslated Latin 150, length ratio 127, name lost 119,
  glossary (cart as fence/trash bin, item as article, human-agent calque) 197, Russian
  function words 101.
- **Post-QE audit** (deterministic sample, LLM reviewer): RU major errors 3/60 = 5%, minor
  14/60 = 23% (literal calques); KK major errors 11/55 = 20% (Wilson 95% CI ≈ 11–32%),
  minor 9/55 = 16%. QE scores do not separate the remaining KK semantic errors (wrong
  person, meaning shifts) from correct rows. **This is the main quality risk of the dataset.**
- Localisation: USD amounts < 1,000 → KZT at 500/USD; Stage 3 slot amounts ≥ 1,000 are kept as
  tenge values (first build converted them to 100+ million tenge; fixed, 38 rows re-translated).

### SFT dataset `sft-20261112-a33e9c1b` (`data/processed/sft/sft-20261112-a33e9c1b/`)

| split | examples | EN | RU | KK | scenarios | origin |
|---|---|---|---|---|---|---|
| train | 15,548 | 4,537 (29.2%) | 5,718 (36.8%) | 5,293 (34.0%) | 141 | MT 10,472 · original 4,297 · authored 779 |
| validation | 98 | 28 | 36 | 34 | 16 | authored |
| test | 116 | 33 | 39 | 44 | 16 | authored |

- Tasks in train: grounded_answer 5,979 · clarify 4,835 · tool_answer 1,241 · escalate 1,190 ·
  injection_resist 1,006 · abstain_no_evidence 994 · conflicting_context 232 · refuse_oos 71.
  Per language × task counts are in `stats.json`.
- 35 intents in train; validation 14 intents, test 12 intents (authored scenarios only).
  Code-switching: 67 train / 11 validation / 11 test examples; multi-turn: 95 train.
- The language mix is reached by **down-sampling English only** (16,875 EN Bitext examples
  dropped); RU/KK were never up-sampled. 117 duplicate examples and 38 tool rows without an
  order reference were dropped while building.
- Leakage filter (held-out vs train): exact 0, lexical (J ≥ 0.8) 0, 13-gram 0, semantic (e5
  cosine ≥ 0.95, digits masked) 63 texts → 63 held-out examples removed. Median held-out max
  cosine to train 0.93.
- Gates (all pass): schema, ids, language (user text 100%, answer 100%), licence/provenance,
  grounding (0 unsupported numbers/citations/tool facts), evidence per task, duplicates 0,
  leakage 0, placeholders 0, PII 0, kb_variant_distinct 0. Kazakh: 99.2% of KK user texts and
  100% of KK answers contain Kazakh-specific letters; 81 KK user texts contain a Russian
  function word (code-switched authored texts and loanword noise).
- Training KB `kbtrain` (`data/processed/kb_train/`): 6 stores × 8 documents × 3 languages,
  1,734 chunks (current + outdated). No value equals a demo-store value.
- Authored: 106 situations → 1,056 utterances (EN 324 / RU 353 / KK 379; 92 code-switched,
  95 multi-turn), split by scenario 70/15/15 stratified by action.

### Gold benchmark `gold-20261112-8d91fbf4` (`data/gold/gold-20261112-8d91fbf4/gold.jsonl`)
- 312 scenarios × en/ru/kk = 936 items, written natively per language, separately from the
  training scenarios. Categories (per language): grounded 98, tool 48, escalate 37, clarify 34,
  abstain 23, out_of_scope 23, multi_turn 22, injection 15, conflicting 12; difficulty easy 73 /
  medium 148 / hard 91; 72 multi-turn items, 59 code-switched items.
- Expected facts refer to the demo store (`kb_sources/demo_store/facts.yaml`); all 0 problems
  in the fact, evidence-section and language checks.
- Leakage vs all SFT splits: scenario 0, exact 0, lexical 0, 13-gram 0. Semantic near-train
  (cosine ≥ 0.95): 79 items (KK 42, RU 21, EN 16; mostly short tool requests such as "status of
  order N"), flagged in each item's review notes and reported, not removed (ADR-026).
- **Review status: all 936 items `draft`** (authored by the LLM agent, automatic checks only).
  No item is `human_reviewed` or `verified`.

### Verification evidence (Stage 4)
- `ruff check`, `ruff format --check`, `mypy --strict` clean on Stage 4 code; pre-commit hooks
  run in a clean worktree at commit time.
- `pytest`: **277 passed, 0 skipped** with the Qdrant server up (Stage 5 WIP tests excluded:
  they are untracked files of another session). New Stage 4 tests: 41 unit
  (`tests/unit/test_stage4_units.py`) + 2 integration (`tests/integration/test_stage4_pipeline.py`:
  select → KB → resumable translation with a fake MT backend → QE → SFT gates → gold, re-runs,
  forced rebuild identical, stale partial removed).
- Real data: forced SFT rebuild gives the same `dataset_version` and byte-identical
  `train.parquet`; a plain re-run is skipped. Stage 3 is unchanged: `data clean --all` skips
  with `canonical-20261111-8fe01586`, `data gates` and `data verify --all` pass.
- Interruption: two MT runs were killed by agent-session restarts; the cache kept every finished
  512-row shard and the next run resumed (1,636 KK rows reused).
- Bugs found and fixed: unquoted commas in YAML flow mappings split addresses (demo facts;
  reported by the Stage 5 session; regression test added); QE step skipped after a localisation
  change (sources hash now in its parameters); KK dates without case suffixes in tool answers;
  injection note added when only a planted passage carried the injection; KZT-scale amounts
  converted twice.

## Stage 3 data state (unchanged in Stage 4; build `canonical-20261111-8fe01586`)

| source_id | allowed_use | records | distinct texts | families (J ≥ 0.8) | scenarios | canonical | pending | rejected |
|---|---|---|---|---|---|---|---|---|
| bitext_retail_ecom | train | 44,884 | 43,915 | 34,590 | 46 | 35,980 | 7,937 | 967 |
| bitext_customer_support | train | 26,872 | 23,513 | 20,511 | 32 | 19,668 | 3,845 | 3,359 |
| whileai_ecommerce_intent | train | 24,710 | 24,533 | 24,520 | 24,520 | 0 | 24,533 | 177 |
| support_json_ru | reference_only | 716 | 716 | 91 | 26 | 0 | 716 | 0 |
| belebele | eval_only | 2,700 | 2,700 | 789 | 789 | 0 | 2,700 | 0 |
| **total** | | **99,882** | | | | **55,648** | **39,731** | **4,503** |

- **Canonical:** 55,648 EN rows from Bitext; 23 v0 intents; 45,460 families; **55 scenarios**.
  Actions: answer_kb 30,351, clarify 19,354, call_tool 2,996, escalate 2,947. Every row is
  flagged `labels:intent_map_draft_v0`.
- **Pending reasons:**
  - record_type:dialogue 24,533 (while-ai);
  - intent_unmapped 11,494 (11 Bitext labels without a v0 intent);
  - allowed_use:eval_only 2,700;
  - allowed_use:reference_only 716;
  - label_suspect 237;
  - label_conflict 51.
- **Rejected reasons:**
  - exact_duplicate 4,327 (CS 3,359 = 12.5%, retail 967 = 2.2%, while-ai 1);
  - garbled_text 147;
  - unsupported_language 16;
  - malformed_placeholder 11;
  - encoding_damage 2.
- **Independent scenarios.** Bitext (71,756 rows) has 68 scenarios together, plus 18
  scenarios joined across retail and CS. Volume is about 1,000× larger than real diversity.
- **Overlap.** Retail and CS share 27 exact texts and 129 near-duplicate pairs at J ≥ 0.8.
  No other source overlaps.
- **Token lengths (Qwen3 tokenizer, p50 / p95 / max):**
  - Bitext retail text 13 / 19 / 27, response 175 / 278 / 503;
  - Bitext CS text 10 / 15 / 24, response 104 / 256 / 478;
  - while-ai first turn 26 / 137 / 554, whole dialogue 274 / 1,595 / 10,144;
  - support-json-ru text 30 / 44 / 50;
  - Belebele question 25 / 61 / 104.

  The Qwen3.5 tokenizer is equal on EN and 20–27% shorter on RU and KK text (Belebele
  p50 20 vs 25, support-json-ru 22 vs 30).
- **Register.** In Bitext retail, 49% of rows contain inserted profanity and 51% carry the
  typo tag. In CS it is 5% profanity and 20% typos. while-ai has emoji in 14% of rows, and
  7% of rows contain e-mail-like PII, now masked.
- **Language ID.**
  - Belebele calibration: questions en/ru 100%, kk 99.89% (1 KK question without
    KK-specific letters went to ru); passages 100%.
  - Canonical LID agreement is 100%. 9% of canonical rows have low confidence (typo-heavy
    short EN), and all reviewed ones are English.
- Full report: `artifacts/reports/eda/report.md` (git-ignored, backed up). Regenerate with
  `supportpilot data eda`.

## Stage 3 quality gates (build canonical-20261111-8fe01586)

| gate | result |
|---|---|
| accounting | pass: 99,882 in = 99,882 out, no record in two outputs |
| schema | pass: validity 1.0 (≥ 0.995); canonical 55,648/55,648, pending 39,731/39,731, rejected 4,503/4,503 |
| ids | pass: no duplicate or non-recomputable example ids, no family in 2 scenarios, no dangling `duplicate_of` |
| language | pass: LID re-run on canonical 55,648/55,648 = lang (≥ 0.99) |
| duplicates | pass: 0 exact duplicates and 0 identical texts in canonical; 0 compatible J ≥ 0.8 pairs across families; 2,441 near-identical (J ≥ 0.8) different-label pairs kept apart by the label guard |
| placeholders | pass: no `{{` in text; all slot values are from the catalogue |
| pii | pass: no unmasked e-mail, phone or card |
| licence | pass: canonical only from train sources; allowed_use recorded on every pending/rejected row |
| labels | pass: only v0 intents; no escalation without a reason |
| splits_and_leakage | not applicable: splits are made in Stage 4 |

## Stage 3 qualitative review (LLM agent, not an independent human review)

1,024 records were read and coded: 100 random + 100 flagged per source, and 24 flagged for
Belebele (all it had). Annotations are in `docs/reviews/stage3_review.yaml`.

| source | random: OK / hard defect | main findings |
|---|---|---|
| Bitext retail | 42% / 9% | profanity inserted mechanically (53/100); responses with prompt leaks ("offensive language used in your instruction", a "previous response" that does not exist); systematic mislabel: 265 product-information rows in return_product_in_store (now `label_suspect`); store/online/policy return variants conflict |
| Bitext CS | 59% / 6% | placeholders filled; responses keep placeholders when the text has a literal value; contract-fee intent off-domain; over-the-top response style; Stage 3 currency mismatch found and fixed |
| while-ai | 43% / 52% | about half off-domain (crypto, P2P, bills, homework); labels describe the whole dialogue; garbled turns (147 rejected, single-script word salad still missed); PII-like data masked |
| support-json-ru | 0% / 100% (off-domain by design) | decisions consistent with policies; template grammar errors in 89/716 rows ("1 дней"); human_reviewed=false |
| Belebele | 82% / 0% | reliable language labels; KK translation issues in about 6/33 KK items; CSV quote artefacts in passages |

"OK" is strict (any soft code counts as not OK). Hard codes are LABEL_WRONG, GARBLED,
WEAK_TEXT, OFF_DOMAIN, LANG_OTHER, LID_ERROR and RESPONSE_DEFECT.

## Stage 3 verification evidence (2026-11-11)

- **Checks:**
  - `ruff check`, `ruff format --check` and `mypy --strict` (57 files) are clean;
  - `pre-commit run --all-files` passes;
  - `pytest`: **233 passed, 0 skipped**, with the Qdrant server up.
- **New tests: 59.**
  - Unit: text repair, profanity and fuzzy variants (no false hits on "funky", "fluke"),
    PII masking (including order numbers not masked), garbled detection, placeholder
    filling and classification, intent-map validation and rules, LID (en/ru/kk/und/other,
    names vs code-switch), shingles, MinHash determinism, LSH, exact Jaccard (including the
    chunked path).
  - Integration: the end-to-end pipeline on synthetic sources. It covers routing reasons,
    template and literal duplicates, the label guard, scenarios split by action, slots in
    response, PII, lineage and untouched raw records, skip and stale-partial cleanup, forced
    rebuild byte-identical, crash mid-build, failed gate, corrupt records, missing clean
    checkpoint, and the CLI (`clean --all`, `gates`, `eda --no-tokens`).
- **Real data:**
  - The full forced build gives the same `dataset_version` on every run.
  - The final forced run on the committed code: 10 of 10 clean and canonical Parquet files
    byte-identical to the previous run.
  - Peak RSS was 2.84 GiB. The full run takes 220 s (clean 115 s, build 104 s), and EDA
    takes 63 s.
- **Real interruption.** `taskkill /F` was run twice: during grouping, and while
  `.partial-canonical-*` was being written. Both times the previous build stayed intact,
  and `data verify --all` and `data gates` exited 0. A re-run skipped everything in 2.8 s and
  removed the leftover partial.
- **Bugs found and fixed during the stage:**
  - Non-deterministic MinHash: shingle ids followed Python set order, which is
    hash-randomised.
  - Template/literal duplicates were missed: 242 identical canonical texts, found by the
    duplicates gate.
  - The label check in duplicate groups was first-vs-others instead of pairwise.
  - Card masking hit glued order numbers.
  - The currency symbol disagreed between text and response.
  - A CLI test resolved the real project root. The test is now pinned to `tmp`, and the
    real outputs were rebuilt identically.

## Stage 4 historical limitations (expansion updates above)

1. **No fluent human review** of RU/KK. The MT error rates, the authored scenarios, the answer
   templates, the training KB texts and all gold items were written or judged by the LLM agent.
   Gold items are `draft`. Needed: a fluent KK and RU reviewer (open decision 2) to audit
   ≥ 100 MT rows per language and to review gold (`verified` = author + independent human).
2. **KK machine translation quality.** About 20% of accepted KK MT rows still carry a major
   error (LLM-judged, CI 11–32%). Options: a paid MT/LLM budget (open decision 1), Tilmash or
   KazParC access (HF account), human post-editing, or down-weighting `mt:` rows in Stage 7.
3. **Original volume.** 15,548 / 98 / 116 instead of 90k / 5k / 5k. Nothing was duplicated. Limits: one
   GPU (MT ≈ 2.5 rows/s), 55 Bitext scenarios, authored scenarios written by one agent.
   Validation/test originally covered only 14 / 12 intents; the expansion closes this gap
   with 35 intents in every language in both sets. Use validation for tuning and keep gold
   for final evaluation.
4. **Templated targets.** Original answers came from authored templates (≈ 40 section answers,
   ≈ 30 tool answers per language). The expansion adds wording variants, but targets remain
   template-based and are not independently human-validated.
5. **Bitext register.** EN originals keep typos and 10% inserted profanity; RU/KK
   translations of typo-heavy sources are literal. The original train had 71 refuse_oos
   examples; the expanded train has 928.
6. Still open from earlier stages: gated sources, `uv sync --no-group gpu` not run, no LICENSE
   file (authored content has no declared licence yet), paid-API budget, reviewers,
   Kaggle/Colab accounts. Detector gaps from Stage 3 (single-script word salad, obfuscated
   e-mails, Latin-script Kazakh) are unchanged.

## Git and backups

- Remote: `origin` = https://github.com/Nusultan11/SupportPilot (public); `main` is pushed
  after each stage.
- Backups live in `backups/` at the project root (git-ignored, ADR-013).
- Code bundles (`backups/code/`): Stage 0 `…T104345Z`, Stage 1 `…T115429Z`, Stage 2
  `…T134356Z`, Stage 3 `…1T154236Z`, Stage 4 `…2T084303Z`, Stage 5 `…2T085955Z`.
- Stage 3 data snapshots (pinned): `clean_records/20261111T154052Z-5d85c3cb`,
  `canonical/20261111T154055Z-c62f57ae`, `eda_report/20261111T154057Z-9fe64e24`.
- **Stage 4 data snapshots** (pinned, deep-verified, 2026-11-12):
  - `stage4_interim/20261112T083636Z-dceb7610` (6.3 MB: backbone, MT cache, translations + QE);
  - `kb_train/20261112T083638Z-a667cddf` (training KB chunks);
  - `sft/20261112T083640Z-7ca2f2fc` (`sft-20261112-a33e9c1b`, 2.3 MB);
  - `gold/20261112T083643Z-8d81b605` (`gold-20261112-8d91fbf4`, 0.7 MB);
  - `stage4_reports/20261112T083645Z-aec9e7b1` (MT comparison samples and metrics).
- Models are not backed up: they are pinned by revision in `configs/base.yaml` and re-download
  into the HF cache (MADLAD-400-3B 11.8 GB, Opus-MT, M2M100, multilingual-e5-small).
- **Stage 4 code:** commit `da28191` on `main` (committed from a clean worktree so that the
  hooks checked only Stage 4 files); bundle `backups/code/supportpilot-20261112T084303Z.bundle`
  (verified, sha256 file next to it).

- **Stage 5 data snapshots** (pinned, deep-verified, 2026-11-12):
  - `kb_demo_store/20261112T085739Z-f82d885d` (`data/kb/demo_store`: sources, version
    `kb_demo_store-20261112-bcae8093`, `current.json`; 71 files). Restore drill: restored into
    an empty directory, byte-identical, and `kb validate` passed all 13 gates on the copy.
    Taken from the final code before the commit, so its `code_commit` reads `6e5c537…-dirty`.
  - `kb_reports/20261112T085742Z-59678c41` (parser comparison results and script).
- **Stage 5 code:** commit `c79d8af` on `main`; bundle
  `backups/code/supportpilot-20261112T085955Z.bundle` (verified, sha256 file next to it).
  A clean clone of the bundle (`uv sync --locked`, 22 s; `kb build`) reproduced
  `kb_demo_store-20261112-bcae8093` with byte-identical `chunks.parquet` and `documents.parquet`.
- Stage 4 builds after Stage 5: `facts.yaml` gained three keys (values of Stage 4 gold
  `FIXED_FACTS`; no existing value changed). The gold fact check still finds 0 problems on all
  936 items and `kb_variant_distinct` still passes; a Stage 4 re-run will rebuild `kb-train`
  because the facts file hash is one of its inputs.

## Notes for the next stages

- Stage 6 (retrieval): index the chunks of the current KB version
  (`data/kb/demo_store/current.json` -> `versions/<kb_version>/chunks.parquet`, `chunk_json`
  = KBChunk). Reindex incrementally from `changes.json` (`chunks_delete`, `chunks_upsert`).
  Filter by `lang`, `doc_type`, `effective_from/to`; use `translation_group` for cross-lingual
  evaluation and `evidence_map.json` for gold citations. Demo facts must keep their values
  (gold, `kb_variant_distinct`).
- Stage 7 (training): load `data/processed/sft/<version>/{train,validation}.parquet`;
  `messages_json` is ready for chat templates; `quality_flags` (`mt:madlad400_3b`,
  `src:bitext`, `src:authored`, `injection:*`, `conflict:*`) allow weighting. Report every
  metric per language and on gold, with the semantic-near-train subset separately.
- MCP dev tools (ADR-012 amendments, 2026-11-12; rules AGENTS §12, setup RUNBOOK §4a).
  Not runtime: no stage status, pipeline, contract or dependency changed.
  - Remote, on by default: `context7`, `exa`, `github` (read-only), `huggingface`.
  - Local stdio, off by default via `/mcp` in this clone: `serena`, `playwright`, `mlflow`.
    A new clone must disable them once (RUNBOOK §4a step 1).
  - Verified: Claude Code health check (all remote servers connected; `github` with a token
    in the process env only), one safe call per server, no leftover local processes.
  - Open: `github` needs the owner's fine-grained read-only PAT in the `GITHUB_MCP_PAT`
    user env var; until then it fails on connect and the others are unaffected. Not
    exercised here: re-enabling a server in `/mcp` inside a running session, and the tool
    calls through the desktop app itself (the client was checked with `claude mcp`, the
    calls with a fastmcp client).
  - `.claude/settings.json` now denies the MLflow write tools (the earlier open item).
- Re-running Stage 4: RUNBOOK §7c. MT is the only GPU step; everything else rebuilds in
  minutes from the cache.

## Next recommended action

Stage 6 (retrieval experiments) can use the expanded KB and independent retrieval set.
Stage 7 can use the expanded SFT. Paid API budget and fluent human reviewers remain open;
no next-stage experiment has been started by the data preparation task.
