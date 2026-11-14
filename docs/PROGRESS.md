# PROGRESS — current handoff

Updated 2026-11-15. This file owns current state and next actions.
Experiment interpretation belongs to the linked reports; commands belong to RUNBOOK.

## Current work

| Scope | State | Remaining work |
|---|---|---|
| Stages 0–6.1 | Completed with documented quality limits | Preserve frozen evidence and rollback paths |
| Stage 7 | IN PROGRESS: final Kaggle assessment /2 RUNNING | Retrieve terminal outputs, verify, analyse, report and deliver |
| Repository maintenance | COMPLETED, owner-authorized independent task | Preserve the lifecycle rules and delivery evidence below |
| Stages 8–10 | Not started | Require a separate owner instruction |

The owner authorized Stage 7 and the existing private Kaggle transfers/execution/retries.
Routing and selected-training payloads contain source/configuration and synthetic train/validation.
The separately authorized final-assessment payload includes frozen test/gold/safety inputs.
Credentials and public publication are excluded. Preserve those scopes; do not upload new content
or retune from held-out outcomes.

## Active final assessment

- Exact kernel: `nurs2005/supportpilot-stage7-final-evaluation-run/2`.
  Last verified RUNNING; transition observed 2026-11-14 17:24 UTC (22:24 Asia/Qyzylorda).
  Execution limit 21,600 seconds; this is not a promised successful completion time.
- Private Dataset: `nurs2005/supportpilot-stage7-final-evaluation`, version 1.
  Canonical package: `artifacts/kaggle/stage7_final_evaluation/`.
  Source `kernel/`, explicit-version readback `remote/2/`, immutable `submission.json`,
  status history `status_history.jsonl`, outputs `output/2/`.
- Payload SHA256: `6ad278d4b7ceb78bf37b83ea96f1e03b3bdf61b660484871c9885de9229a379a`.
  Independently downloaded Dataset readback verifies all 128 members/127 manifest file hashes
  (`received_dataset/1/verification.json`).
- Runtime source LF SHA256:
  `a5645a96206b6944e516c0346b39d12950c8f1a100f568a18f2bb5e03dd61b3d`.
  Pinned image matches the verified training image; actual final runtime is not yet read back.
- Expected work: 12 tasks across two workers; 3,050 generation observations plus 572 routing
  observations = 3,622. Cohorts: test 286, gold 936, safety 150, sufficiency 144, conflict 9.
- Final /1 was cancelled while queued/empty; /2 keeps the same payload/configuration.
  Version 2 maps to session 369527684 (`output/2/download_mapping.json`).
  Earlier 22-byte empty archives/receipts are historical evidence; do not overwrite them.
- No completed final LLM outputs have been verified/imported. Empty logs do not establish progress.
  API authentication outages occurred; fresh CLI recovered access. On access failure record
  UNKNOWN_ACCESS_DENIED with last known state; do not cancel or duplicate a running job.
- Auxiliary queue preflight /1 completed on two T4 GPUs. Probes /2–5 were cancelled queued/empty;
  L4 hardware execution remains NOT_OBSERVED. Conditional L4 runner is NOT_SUBMITTED.
  Do not launch it or a local full replacement while final /2 runs.
- Foreground terminal watcher: `artifacts/reports/llm/launch/watch_final_status.py`.
  CLI 2.2.5 / kagglesdk 0.1.38; OAuth recovery and exact-version retrieval: RUNBOOK Stage 7 section.

## Frozen selection and protected artifacts

Selection seal: `570521fe81adaf13af962728b66f76da4f98e05aed8c47a47c891402699fa935`
at `artifacts/reports/llm/selection.json`; it still verifies after documentation edits.

| Role | Frozen configuration |
|---|---|
| Selected generator | Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`; schema prompt; primary seed 20261113 |
| Adapter | All-linear rank16/alpha32, LR1e-4, 512 steps, MT weight0.5; FP32 trainable adapters over FP16 base |
| Deployment control | Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; few-shot, three frozen train demonstrations |
| Final inference | FP16, quantization none, input6144/output512/time90s; constrained decoding disabled |
| Router | Frozen E5-small/C10 head; confidence uncalibrated |

- Immutable inference export: `artifacts/models/llm/selected_inference/`.
  Weights 40,422,168 bytes; complete bundle 51,853,699 bytes.
  Do not edit its README/manifest after sealing. Mutable evaluation card:
  `artifacts/reports/llm/model_card.md`.
- Raw training/checkpoints: `artifacts/kaggle/stage7_selected_training/`, exact run /1 COMPLETE.
  All 9 tasks finished; all 218 result hashes and 12 optimizer checkpoints verified.
  Primary checkpoint 256 actually resumed to512; all final tensors finite.
- Inputs: `sft-20261112-b0388c64`; frozen heldout directory
  `data/benchmarks/llm_heldout/<selection_identity>/`.
  Do not regenerate sealed validation reports, demonstrations or inputs during housekeeping.
- Baseline/sweep/routing/smoke packages remain under `artifacts/kaggle/`.
  Failed/partial attempts and the original routing1 missing-archive-hash limitation are retained.
- Source/lock/configuration and frozen protocols have not changed during maintenance.
  Preserve `docs/reviews/*_protocol.md`, authored review YAML and frozen checksums.

## Confirmed results and limits

[Stage 7 report](reviews/stage7_results.md) owns per-language tables/CIs, training warnings,
routing failures and local resource measurements. Main conclusions:

- Validation: adapter schema improves matched 0.6B diagnostics; score .83906 versus 2B control .51049.
  KK intent accuracy .51327; its improvement over 2B has a CI including zero.
- Primary training has no logged nonfinite gradients. Each secondary seed logs one NaN grad_norm;
  saved weights are finite, optimizer skip count UNKNOWN. No favorable-seed substitution.
- Frozen encoder outperforms measured LLM routing on validation, but critical escalation failures
  prevent autonomous routing acceptance.
- Grounded-answer accuracy, semantic citation precision, unsupported-claim/false-answer quality:
  NOT_EVALUATED. No calibrated independent judge or fluent KK human review.
  Default generation accepts no factual candidate with semantic support UNKNOWN.
- Local selected-adapter probes: three calls (one per language) in each isolated/joint scope.
  They establish limited compatibility only. FP16 joint reservation exceeds physical VRAM;
  NF4 comparisons lack captured original prompt bytes and are not a causal precision ablation.
- Local 2B FP16 resource probes retain two pre-load RAM-guard failures and three successful
  already-exposed validation calls after temporary service isolation. Allocated memory exceeds
  physical VRAM; full residency/sustained feasibility NOT_ESTABLISHED.
  Docker/Qdrant restored healthy; `local_control_fp16/service_pause.json`.
- Gold has 79 semantic-near-train rows; 857 unflagged rows are not certified contamination-free.
  Sufficiency reuses18 exposed development needs; conflict9 derives from one artificial need.
  Gold allowed citation sections are not AND-proof groups.

[Stage 6.1 report](reviews/stage6_scale_results.md) owns retrieval findings:
222 document IDs /666 language versions /8,865 chunks; freeze
`6fab32c16b3a6e05b29f` has3,570 needs/12,593 variants/119 families.
Holdout RU/KK/EN Recall@5 .90155/.93264/.97927, multi-document packed sufficiency .70297;
FPR1.0 on606 negative variants. Draft correlated synthetic judgements and packing loss remain.
The historical Stage 6 holdout was inspected during rule revision and is exploratory.

## Serving and recovery identities

- Active KB: `kb_demo_store-20261113-f3b204a6`.
- Namespace: `scale_6fab32c16b3a6e05b29f`; index `rindex-20261113-9ba94665`.
- Defaults: BGE-M3, word sparse, weighted dense0.75, candidates20, BGE rerank20,
  output10, context1024, no evidence-score gate.
- Baseline KB `kb_demo_store-20261112-d58e3f93`, its index/configuration and all 369 original
  chunk IDs are retained. Rollback instructions: RUNBOOK §16.
- Qdrant Docker named volume is active and protected. Local model caches and .venv are retained.
  No Stage 8 API/tools/UI exists.

## Verification and recoverability

Maintenance removed about16.0GB of proven redundant restore copies, old test fixtures and
reproducible caches; validation-generated temporary files are accounted separately.
The measured workspace fell from45.85GB/89,808files to29.85GB/58,505files before the small
delivery backup. Data, tracking, environment and existing backup byte/file totals were unchanged.
All10 primary documents were reconciled, all33 ADR IDs retained, and four unmatched recovery
directories preserved. Evidence/retention reasons: `artifacts/reports/maintenance/summary.json`.
The24-file maintenance evidence snapshot was deep-verified and actually restored;
publication and code-bundle checks: `artifacts/reports/maintenance_delivery_receipt.json`.

Maintenance verification: 466 passed, zero skipped (405.86s), including real GPU/Qdrant
checks; one expected embedded-Qdrant warning. Ruff/format (173 files), strict mypy
(158 files), 41 internal Markdown links and all33 preserved ADR IDs pass.
An earlier run inside the ignored checkout failed one Git-isolation test; its traceback
is retained and the complete external-temp rerun passed without code/test changes.
Evidence: `artifacts/reports/maintenance/verification.json`, `pytest.log`,
`test_isolation_diagnosis.json`. Runtime/configuration/data sources, frozen protocols,
review records and selection identity are unchanged; MLflow SQLite integrity is ok.

Latest pre-maintenance code is published at `546f2df`; its bundle
`backups/code/supportpilot-20261114T183753Z.bundle` was verified and cloned clean.
Private recovery evidence:
`artifacts/reports/llm/running_assessment_code_backup.json`,
`running_assessment_backups.json`, `selected_verified_backups.json`,
and the manifests under `backups/`.
Latest running-assessment snapshots:
`stage7_queue_execution/20261114T183557Z-060225d0`,
`stage7_reports/20261114T183558Z-0131d314`,
`stage7_tracking/20261114T183606Z-c1e56159` (restored SQLite integrity ok).
Selected inference snapshot:
`stage7_selected_inference/20261114T103634Z-3786a359` (actual restored PEFT load verified).
Stage 6.1 delivery: `artifacts/reports/retrieval_scale/delivery_snapshots.json` and
`artifacts/reports/retrieval_scale_delivery_receipt.json`.

Maintenance inventory/deletion evidence is in `artifacts/reports/maintenance/`.
Redundant restore copies may be removed only after exact backup verification; their receipts
and original archives remain. No backups are authorized for deletion.
Earlier chronological detail is preserved in Git at commit546f2df and private tracking/artifacts.

## Next actions

1. Continue observing exact final /2. On COMPLETE, run
   `artifacts/reports/llm/launch/download_terminal_final.py` with the Kaggle-enabled global Python,
   then `final_analysis.py` using the project environment. Verify all 12 tasks/3,622 observations,
   source, seal and all checksums before importing/reporting quality.
2. On ERROR/timeout, preserve partial output and failed attempts; diagnose execution without
   changing selected parameters or treating a technical retry as fresh holdout.
3. Update the external model card, resource/cost accounting and measured report; retain semantic
   NOT_EVALUATED where unsupported. Back up/restore results and MLflow, commit/push Stage 7.
4. Report the Stage 7 completion/readiness decision; do not implement Stage 8.
