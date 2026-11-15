# PROGRESS — current handoff

Updated 2026-11-15. This file owns current state and next actions; detailed results
belong to the linked reviews, commands to RUNBOOK, and raw evidence to local artifacts/MLflow.

## Completion and next boundary

| Scope | State | Remaining limitation |
|---|---|---|
| Stages 0–6.1 | COMPLETED | Retrieval and synthetic-evaluation limits remain documented |
| Stage 7 | COMPLETED: engineering and research assessment | Autonomous factual support NOT_READY; critical escalation NOT_ACCEPTED |
| Repository maintenance | COMPLETED: independent owner-authorized task | Preserve the documentation and artifact lifecycle rules |
| Stages 8–10 | NOT_STARTED | Separate owner authorization required |

No jobs remain to watch for the final assessment. Do not retune from the exposed test/gold
outcomes or start Stage 8. Future improvements need fresh development evidence, independent
fluent RU/KK review, calibrated claim support and measured end-to-end delivery.
Semantic quality remains NOT_EVALUATED; UNKNOWN factual support must take the safe fallback.
No customer-facing API/tools/UI or actual operator-delivery integration exists yet.

## Frozen Stage 7 identity

Selection seal: `570521fe81adaf13af962728b66f76da4f98e05aed8c47a47c891402699fa935`
at `artifacts/reports/llm/selection.json`; verified unchanged after assessment and maintenance.

| Role | Frozen configuration |
|---|---|
| Generator | Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`; schema prompt; seed 20261113 |
| Adapter | All-linear rank16/alpha32, LR1e-4, 512 steps, MT weight0.5; FP32 trainable adapters over FP16 base |
| Deployment control | Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; few-shot, three frozen train demonstrations |
| Assessment inference | FP16, quantization none, input6144/output512/time90s; constrained decoding disabled |
| Router | Separate E5-small/C10 encoder head; confidence uncalibrated |

- Immutable inference export: `artifacts/models/llm/selected_inference/`, seven files,
  40,422,168-byte weights; complete bundle 51,853,699 bytes. Its native README/manifests
  remain sealed. Mutable assessment card: `artifacts/reports/llm/model_card.md`.
- Selected training: `artifacts/kaggle/stage7_selected_training/`, exact /1 COMPLETE,
  nine tasks/three seeds, all 218 result hashes and 12 optimizer checkpoints verified.
  Primary checkpoint256 actually resumed to512. Final saved tensors are finite.
- Inputs: `sft-20261112-b0388c64`; frozen heldout inputs under
  `data/benchmarks/llm_heldout/<selection_identity>/`. Do not regenerate sealed inputs,
  selection reports or demonstrations. Failed/partial attempts remain retained.

## Verified assessment and readiness

[Stage 7 report](reviews/stage7_results.md) owns the complete per-language results,
paired uncertainty, error review, training warnings and resource interpretation.
Canonical private evidence: `artifacts/reports/llm/final_evaluation.json`,
`assessment_verification.json`, `readiness.json`, `error_review.json` and `model_card.md`.

- Exact private kernel `nurs2005/supportpilot-stage7-final-evaluation-run/2` COMPLETE;
  session 369527684. All 12 tasks FINISHED, 3,622 observations (3,050 generation/572 routing),
  both workers exit0, no inference execution exceptions. Source/payload and every result
  checksum verified before analysis/import. All 12 assessment MLflow runs are FINISHED;
  `assessment_tracking_identity.json` clarifies actual cohort/KB provenance separately
  from the importer's legacy parent-SFT dataset tag.
- Package: `artifacts/kaggle/stage7_final_evaluation/`; exact-version readback `remote/2/`,
  archive/results/receipts `output/2/`. Terminal ZIP SHA256:
  `96aed44930c4d2db3f23b3a9fa845d7257c6f01a67a7d32e2dd34c96e3eae8db`.
  Runtime source, payload and frozen analysis hashes are in `assessment_verification.json`.
  Actual remote Python3.13.16/Torch2.11.0+cu130 differs from the local locked environment.
- The adapter improves JSON validity. Test EN/RU/KK JSON: 98.89/98.95/99.01%;
  intent: 74.44/71.58/47.52%. Gold intent: 52.24/44.55/35.90%.
  KK test intent improvement over the 2B control has a scenario CI including zero;
  gold KK macro-F1 is about .224 for both. Different model/prompt controls do not isolate SFT.
- Safety required escalation 26/language; adapter raw misses EN 16/RU 18/KK 24.
  These are draft-label disagreements, not measured harm rates. Some refusals are safe
  but still miss the requested escalation. Encoder routing also fails critical acceptance.
- Semantic grounding, claim support and false-answer quality remain NOT_EVALUATED.
  Targeted LLM self-review inspected 50 escalation failures/language, not an independent
  semantic benchmark. Conflict handling and incomplete evidence reveal concrete unsafe
  candidates. Default UNKNOWN-to-fallback prevents factual candidate acceptance.
- Gold contains 79 semantic-near-train rows; 857 unflagged rows are not certified clean.
  Gold families are nine coarse categories and allowed citations are not proof groups.
  Sufficiency reuses 18 exposed development needs; conflict 9 is one artificial need.
  Mixed-language slices overlap language totals. Sparse-class F1 bootstrap is exploratory.
- Local probes establish limited compatibility only: three calls per scope, one per language,
  not a sustained service. Joint FP16 reservation exceeds physical VRAM; full residency/SLO
  NOT_ESTABLISHED. NF4 comparison lacks original prompt bytes. Docker/Qdrant restored healthy.
  Secondary seeds each recorded one NaN grad_norm; final weights finite, optimizer skips UNKNOWN.

Raw cloud packages, cancelled/failed attempts and the original routing1 archive-hash UNKNOWN
remain recorded; no new selection or favorable-seed substitution followed the heldout assessment.

## Retrieval and serving identities

[Stage 6.1 report](reviews/stage6_scale_results.md) owns retrieval results and limitations.
Active KB `kb_demo_store-20261113-f3b204a6`, namespace `scale_6fab32c16b3a6e05b29f`,
index `rindex-20261113-9ba94665`. Defaults: BGE-M3 + word sparse, weighted dense0.75,
candidates20, BGE rerank20, output10, context1024, evidence-score gate disabled.
The baseline KB `kb_demo_store-20261112-d58e3f93`, index and 369 chunk IDs remain available;
rollback: RUNBOOK §16. Preserve the Qdrant named volume, model caches and .venv.

## Maintenance and verification

Maintenance consolidated all 10 primary documents, retained all 33 ADR IDs and original
protocols/review records, and established one owning document per topic (AGENTS §3).
Cleanup removed 16,002,635,789 bytes of verified redundant restore copies, old test fixtures
and reproducible caches; validation-created temporary files are counted separately.
Measured workspace before/after delivery: 45,849,331,301/29,862,231,804 bytes and
89,808/58,517 files. Later assessment results/backups legitimately add new evidence.
No backup, dataset, KB/index, selected model/checkpoint, MLflow history, environment,
required model cache or Qdrant storage was deleted. Four unmatched recovery trees remain
with recorded reasons. Evidence: `artifacts/reports/maintenance/summary.json` and
`artifacts/reports/maintenance_delivery_receipt.json`; published maintenance commit `7aa615b`.

Full shared checks: Ruff/format 173 files, strict mypy 158 files, pytest 466 passed/zero skipped
(405.86s), including real GPU/Qdrant checks; one expected embedded-Qdrant warning.
The earlier inside-checkout Git-isolation test failure is retained; the full isolated-temp
rerun passed without runtime/test changes. No runtime/configuration/data-source change
followed those checks. Final documentation/provenance checks and recovery receipts:
`artifacts/reports/stage7_delivery_receipt.json`.

## Recovery and continuation

- Final assessment package, reports and updated MLflow are pinned, deep-verified and actually
  restored; exact snapshots, SQLite integrity, code bundle and publication proof are indexed
  by `artifacts/reports/stage7_delivery_receipt.json`.
- Selected inference snapshot `stage7_selected_inference/20261114T103634Z-3786a359`
  has an actual restored PEFT-load proof; raw selected-training snapshot
  `stage7_selected_training/20261114T054643Z-613ff4e9` preserves optimizer checkpoints.
- Maintenance evidence snapshot `maintenance_evidence/20261114T192803Z-484ebcd7`
  and its code bundle have verified restore receipts. Stage6.1 recovery receipts:
  `artifacts/reports/retrieval_scale/delivery_snapshots.json` and
  `artifacts/reports/retrieval_scale_delivery_receipt.json`.
- Backups are local/same disk: they do not protect against disk loss. No backup deletion is
  authorized. Remove only newly created verified restore copies, retaining archives/receipts.
- Stop here. The next product stage requires the owner's instruction and must retain the
  explicit semantic, KK, escalation and resource limitations above.
