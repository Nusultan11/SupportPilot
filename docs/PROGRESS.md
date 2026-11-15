# PROGRESS — current handoff

Updated 2026-11-15. This file owns current state and next actions; detailed results
belong to the linked reviews, commands to RUNBOOK, and raw evidence to local artifacts/MLflow.

## Completion and next boundary

| Scope | State | Remaining limitation |
|---|---|---|
| Stages 0–6.1 | COMPLETED | Retrieval and synthetic-evaluation limits remain documented |
| Stage 7 | COMPLETED: engineering and research assessment | Autonomous factual support NOT_READY; critical escalation NOT_ACCEPTED |
| Repository maintenance | COMPLETED: independent owner-authorized task | Preserve the documentation and artifact lifecycle rules |
| Engineering MCP integration | COMPLETED: independent owner-authorized task | Opt-in lifecycle and upstream advisory/recovery limits retained; delivery receipt below |
| Stage 8 | COMPLETED: guarded local engineering prototype | Semantic quality NOT_EVALUATED; autonomous support NOT_READY |
| Stage 9 | PARTIALLY COMPLETED: engineering assessment delivered | Independent/semantic quality acceptance incomplete |
| Stage 10 | NOT_STARTED | Separate owner authorization required |

No cloud jobs remain to watch for the Stage 7 assessment. Do not retune from the exposed test/gold
outcomes. Stage 8 is delivered; Stage 9 was authorized on 2026-11-15. Improvements need fresh
development evidence, independent
fluent RU/KK review, calibrated claim support and measured end-to-end delivery.
Semantic quality remains NOT_EVALUATED; UNKNOWN factual support must take the safe fallback.
Stage 8 now implements authorized local tools, API/UI and a durable operator queue;
recording in that queue is not an external operator response or production acceptance.

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
- Stop before Stage 10. The next product stage requires the owner's instruction and retains the
  explicit semantic, KK, escalation and resource limitations above.

## Stage 8 delivery and next boundary

[Stage 8 report](reviews/stage8_results.md) owns implementation scope, actual laptop observations,
E2E denominators and limitations. Contracts: DATA_CONTRACTS §18; decisions: ADR-034;
verified local launch and one-command E2E: RUNBOOK §18. FastAPI/Streamlit, authorized synthetic
SQLite tools, bounded sessions and the persistent operator queue are implemented. Queue recording
is observed; external operator response is not promised. Research generation remains optional,
with every unverified candidate withheld. No sealed model, prompt, index or heldout selection changed.

Canonical private evidence is under `artifacts/reports/service/`; `latest_probe.json` points to
final attempt `4dcde951fdb847818ccb481cbbc07b58` with matching delivered runtime/config hashes.
The same one-command attempt contains the final controlled E2E log
and actual inference calls. The report distinguishes scenario-state cases from linguistic needs,
model calls from tool/guard outcomes, and earlier timing/metric limitations. Five retained probe
attempts are imported into MLflow with execution provenance separate from importer Git metadata.

Shared checks, final targeted E2E/UI checks and actual loopback API/UI verification passed.
Data/service state, reports and updated MLflow have pinned deep-verified snapshots and actual
restore/SQLite/run-identity proofs. Final code-bundle recovery and origin/main publication are
indexed by `artifacts/reports/stage8_delivery_receipt.json`. Task-owned verification servers are
stopped. Backups remain same-disk; no backup deletion was authorized.

Stage 9 is authorized. Retain semantic, KK, critical-risk coverage and resource limitations;
do not tune from exposed Stage 7 test/gold outcomes.

## Stage 9 current work

Protocol: [Stage 9 preregistration](reviews/stage9_protocol.md). New draft inputs are under
`authored/system/`; the private freeze is indexed by `artifacts/reports/system/freeze_pointer.json`.
Translations/followups share needs. Semantic independence is uncertified; Stage 7 is unchanged.

Development-only selection is sealed in `artifacts/reports/system/selection.json`.
Both once-only confirmation attempts finished; `holdout_stage8.json` and
`holdout_boundary_guard.json` identify raw reports. The narrow word-boundary guard fix is
selected; overall confirmation delivery improvement is not established. Intent targets are
missed; critical coverage and independent semantic review remain limitations. Do not tune from
these now-exposed confirmation outcomes.

Resource/HTTP/precision studies are complete: `artifacts/reports/system/performance/`;
supplemental locator assessment: `system/retrieval/17f947d9ce20441285435e7457b2364b/`.
[Stage 9 results](reviews/stage9_results.md) owns measured effects/limits; logs: `system/checks/`.

Million stress source `a7057c9dfb70c66d718c`, collection `sp_scale_load_a7057c9dfb70c66d718c`.
Initial run `16e4cc274a9a44a5b83f98f461995f7a` FAILED at the first cold exact query's10s
deadline after successful1M upload/HNSW. Receipt:
`artifacts/reports/retrieval_scale/load/a7057c9dfb70c66d718c-1794727860994123100/receipt.json`.
Server stayed healthy; no restart/OOM. Technical continuation `c0d5b3bf62e448ada7c97350949949b8`
FINISHED with the same source/query bytes and120s research deadline, without upload/build replay.
Receipt directory ends `a7057c9dfb70c66d718c-1794734687280595500`; wrapper:
`artifacts/reports/system/stress/7d1c37ce1b8d4c9794189f2e0354be43/report.json`.
All1M vector/payload records and six exact-query rankings match after actual snapshot restore.
Do not rerun or remove the retained source/restore collections.

Embedding audit and real HTTP-generation probe finished; all3 actual generated candidates were
withheld and actually queued. Audit flags89/480 rows (38 near-train); do not exclude or relabel
them after outcomes. No independent-semantic/gold claim is valid. Owner confirmed that fluent
reviewers and labelled calibration data are unavailable; semantics stays NOT_EVALUATED.

Shared checks:810 passed/zero skipped (429.55s), Ruff/format195, strict mypy176 plus UI.
Five pinned snapshots cover benchmarks, load vectors/evidence, system reports and tracking;
deep verification and actual restores pass, including nine service SQLite databases and17
current MLflow run identities/statuses. Exact snapshots, code-bundle restore and origin/main
publication proof are indexed by `artifacts/reports/stage9_delivery_receipt.json`; check that
receipt rather than inferring publication from this handoff. Backups remain local/same-disk.

Next: independent overlap adjudication, fluent calibration and fresh development/confirmation
within Stage 9 before quality acceptance/advancement. Do not retune exposed outcomes.
Stage 10 and autonomous generation remain unauthorized/not ready.

## Engineering MCP integration (standalone owner request)

Scope: optional Shrimp internal task tracking and a single validated Structurizr C4 model;
no new product stage, data/training experiment or runtime change. Canonical model/architecture
review: ARCHITECTURE; decision: ADR-036; reproducible setup and opt-in lifecycle: RUNBOOK §4a.
Native project Codex entries are installed disabled by default; direct stdio/HTTP MCP clients
verify actual operations. An existing host chat does not hot-reload its native tool catalogue.
Shrimp's private dependency graph/results are under `artifacts/engineering/shrimp/`;
verified live checks/DSL parse/inspection receipts: `artifacts/reports/engineering/mcp/`.
Original stage reports and frozen identities are unchanged. The Zod3 advisory and nontransactional
upstream writes remain documented; local bounded single-writer operation/verified snapshots apply.
Full pytest:810 passed/zero skipped; Ruff/format195 and strict mypy176 plus UI pass. Verified
task-state/evidence snapshots, actual restores, final hooks, code-bundle recovery and exact
origin/main publication are indexed by `artifacts/reports/engineering_delivery_receipt.json`;
verify that receipt rather than inferring publication from this handoff. Both MCPs default off
between tasks; no engineering daemon is required. Native interactive Claude remains unverified.
Stage 9 quality limitations and the Stage 10 authorization boundary above remain in force.
