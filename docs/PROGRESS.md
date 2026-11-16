# PROGRESS — current handoff

Updated 2026-11-16. This file owns verified state, authorization and next action. Stage criteria
live in ROADMAP, interpretation in reviews, commands in RUNBOOK and raw evidence in private artifacts.

## State and authorization

| Scope | State | Remaining limitation |
|---|---|---|
| Stages 0–6.1 | COMPLETED | Synthetic retrieval/data and independent-review limits remain |
| Stage 7 | COMPLETED: engineering and research assessment | Autonomous factual support NOT_READY; critical escalation NOT_ACCEPTED |
| Stage 8 | COMPLETED: guarded local engineering prototype | Semantic quality NOT_EVALUATED; local recording is not external operator response |
| Stage 9 | COMPLETED (local engineering and research scope, owner accepted) | Owner decision: ADR-040; product-quality acceptance remains open |
| Stage 9 product-quality acceptance | OPEN | Unmet numerical targets NOT_MET; independent human/semantic assessment NOT_EVALUATED; autonomous factual support NOT_READY |
| Maintenance and engineering MCP integration | COMPLETED: separate owner-authorized tasks | Prior recovery/publication receipts below; MCPs opt-in |
| Architecture and development-control redesign | COMPLETED: separate engineering task | Product-quality gates remain open; verify delivery receipt below |
| Stage 10 | NOT_STARTED | Requires separate explicit owner authorization; Stage 9 acceptance grants none |

On 2026-11-16 the owner explicitly accepted Stage 9's local research, engineering implementation
and self-assessment delivery after reviewing the concrete package. [ADR-040](DECISIONS.md#adr-040--owner-accepted-stage-9-local-engineering-and-research-scope)
records the completion boundary and retained restrictions. This closes the stage in that scope,
not its unmet product-quality criteria. No production-readiness claim or additional stage is
authorized. External experts/helpers and further recruitment/contact remain prohibited.
New private evidence: `artifacts/reports/system/qualification/`. The train-only C10 control
reproduces the frozen head exactly. Both local heads have terminal development and once-only
confirmation; development rejected balancing, so the original head remains selected. The local
support diagnostic rejected the pinned model as an assessor despite a verified JSON-enforcement
fix. `acceptance_packet.json` is the preserved preacceptance packet; `owner_review.json` is an
older intermediate audit. The additive owner acceptance record is
`artifacts/reports/stage9_owner_acceptance.json`; earlier receipts/packets remain unchanged.
Exact attempts, source archives and limitations live in the results review.
The shared local regression passes937 tests with zero skips, Ruff/format and strict mypy plus UI.
Delivery task `c42b010d-5e11-4f30-8200-2d60750fb52f` owns the exact hooks/recovery/publication
receipt; verify it before relying on delivery. Do not recreate completed engineering tasks.
Both9Q1 confirmation variants are consumed and terminal; no reselection/retuning is permitted.
Do not tune exposed Stage 7/Stage 9 outcomes, relabel flagged rows or replace sealed artifacts.
Independent fluent reviewers and labelled calibration are unavailable; semantic acceptance remains
NOT_EVALUATED. ROADMAP 9Q2/9Q3 stay open. A passing engineering delivery cannot waive those gates.

## Runtime and preserved identities

The delivered baseline is a single-worker FastAPI/Streamlit synthetic-store prototype with
SQLite tools, bounded ephemeral sessions and durable queue/replay. Generation is off by default;
default service KB candidates are withheld and queued. An optional qualified acceptance adapter
is implemented, but no real qualified assessor/approval is installed. Authorized tool values may be delivered;
cancellation records a confirmed request only. `/ready` checks local worker/SQLite state, not
full RAG/model or semantic readiness. ARCHITECTURE owns the detailed boundaries and limitations.

Frozen Stage 7 selection identity:
`570521fe81adaf13af962728b66f76da4f98e05aed8c47a47c891402699fa935`,
`artifacts/reports/llm/selection.json`.

| Role | Preserved configuration / evidence |
|---|---|
| Generator | Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, schema prompt, seed20261113 |
| Adapter | All-linear rank16/alpha32/LR1e-4/512steps/MT.5; FP32 trainable adapters over FP16 base |
| Deployment control | Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, few-shot with three frozen train demonstrations |
| Router | Separate E5-small/C10 head, confidence uncalibrated |
| Immutable export | `artifacts/models/llm/selected_inference/`; do not rewrite its native card/manifests |
| Training / heldout | `artifacts/kaggle/stage7_selected_training/` exact /1; `data/benchmarks/llm_heldout/<selection_identity>/`; SFT `sft-20261112-b0388c64` |
| Final assessment | `artifacts/kaggle/stage7_final_evaluation/` exact /2; verified terminal evidence and interpretation in Stage 7 report/receipt below |

Stage 7 raw FP16 assessment and later NF4 serving probes have different execution/resource scopes.
Do not refresh selection from heldout observations. Canonical assessment/readiness evidence:
`artifacts/reports/llm/{final_evaluation,assessment_verification,readiness,error_review}.json`.

Active KB `kb_demo_store-20261113-f3b204a6`, namespace `scale_6fab32c16b3a6e05b29f`, index
`rindex-20261113-9ba94665`. Preserve baseline `kb_demo_store-20261112-d58e3f93` and its369 chunk IDs.
Current recipe/defaults: `configs/base.yaml`; rollback: RUNBOOK §16; measured evidence/limits:
[Stage 6.1 report](reviews/stage6_scale_results.md). Retain Qdrant named volume, model caches and .venv.

## Stage 9 engineering assessment and quality continuation

[Stage 9 report](reviews/stage9_results.md) owns the original measured assessment. The lexical
word-boundary correction is retained; overall confirmation delivery improvement is unestablished.
Intent targets are missed. Semantic/linguistic independence is uncertified; queue success does not
establish grounded answers. Deterministic concurrency1 capacity does not generalize to generation
or multi-user load. Original freezes/results/pointers remain unchanged.

The million-vector research continuation is terminal and recovery-verified, following a retained
failed cold-query/deadline attempt. Source `a7057c9dfb70c66d718c`, collection
`sp_scale_load_a7057c9dfb70c66d718c`; canonical wrapper:
`artifacts/reports/system/stress/7d1c37ce1b8d4c9794189f2e0354be43/report.json`.
Full1M payload/vector and six exact-query restore checks are recorded in the original receipt.
Do not rerun/delete source or restore collections; random-vector fidelity is not semantic quality.

The9Q1 continuation and whole-stage engineering increment have terminal verification evidence.
ADR-037 covers service interpretation; ADR-039 covers optional qualified factual acceptance.
Canonical private evidence: `artifacts/reports/system/quality/`; fresh inputs:
`authored/system/quality.yaml`, freeze `f77c5353fb3b85df5c82` under `data/benchmarks/system_quality/`.
The original Stage 9 source/protocol/results and Stage 7 selection are unchanged.

Development control `4b426acf3fb74419b6082f6171bff559` is reused. CRLF-to-LF-only drift from the
prior seal was proved before confirmation; candidate development was rebound as
`b64e210fd76644f9bdbf801d61aa3ca6`. Current selection seal:
`5af7f5db893972d79a8e7a240669b5b709c6b77ab839ac6a9fd2db66e9fdd1b1`,
`quality/selection_pointer.json`; earlier seals/attempts are preserved. Confirmation control
`612d87b7447a4a879cd88c7bed1f6252` and candidate `a35bb092c14c456f983072fd458e4256` are FINISHED.
Exact sealed source bytes are archived under `quality/sources/<seal>/`; later dormant acceptance
and workload code are a separately verified increment. Raw reports/markers remain unchanged;
`quality/confirmation_analysis.json` owns corrected action scoring, uncertainty and error census.

Audit retains2 lexical,133 prior embedding and80 cross-split row flags without exclusions or
adjudication. Independent reviewers/calibration remain unavailable;9Q2 product qualification and
9Q3 real activation are open. The acceptance adapter is HTTP/SQLite-tested with explicitly
fabricated engineering doubles, never human quality evidence. Default generation remains off.

9Q4 guarded workload is terminal:
`artifacts/reports/system/workload/7f8a295f22214c6583419f418df036b2/report.json`.
It includes auth/tool/queue/KB, four slices, concurrency1/4/16, failures, resources and actual
new-process SQLite replay. Concurrent overload remains a limitation; no capacity change is adopted.
Real default-unqualified generation probe:
`system/live_generation/7196612afcf9485880fd02862be8d922/receipt.json` (three model calls).
Full-scope audit/accounting/checks: `artifacts/reports/system/completion/`.
Final working-tree checks:925 pytest passed/zero skipped, Ruff/format205 files, strict mypy185
files plus UI; C4 validate/parse/inspect pass with the two retained external-document INFO findings.
Final recovery/publication proof is indexed separately below; engineering checks do not close G2.

## Architecture redesign delivery

ARCHITECTURE and the canonical DSL separate implemented/experimental/planned capability across
online support, offline ML and operation. ADR-038 records the target modular architecture;
ROADMAP maps A1–A8 to dependent9Q/10P packages and G0–G5 transition evidence. Contracts and metric
targets stay in their owning documents. No runtime/dependency/model/data/index change is part
of this delivery; pending quality code/protocol/results are preserved separately.

Source/artifact audit, document links and protected identities pass. Structurizr validate/parse/
inspect pass with only two known external-document INFO findings. Full working-tree pytest829
passed/zero skipped (526.95s), including GPU/Qdrant; Ruff/format201 and strict mypy181 plus UI pass.
The working-tree suite includes the unpublished continuation; the public tree is separately
checked with all-files hooks in an isolated main-only clone. Interrupted diagnostic attempts
remain recorded. Recovery/publication are verified by the final index, not inferred from checks.
Private evidence: `artifacts/reports/architecture/`; final check/recovery/Git index:
`artifacts/reports/architecture_delivery_receipt.json`. Shrimp tracks only this task's internal
work and preserves the existing quality-continuation task. It never closes a product-quality gate.

## Recovery and next action

| Delivered scope | Canonical recovery/publication index |
|---|---|
| Stage 7 | `artifacts/reports/stage7_delivery_receipt.json` |
| Stage 8 | `artifacts/reports/stage8_delivery_receipt.json` |
| Original Stage 9 assessment | `artifacts/reports/stage9_delivery_receipt.json` |
| Full Stage 9 continuation /9Q1–9Q4 | `artifacts/reports/stage9_continuation_delivery_receipt.json` |
| Local-only Stage 9 research increment | `artifacts/reports/stage9_local_delivery_receipt.json` |
| Stage 9 owner acceptance and status publication | `artifacts/reports/stage9_owner_acceptance_delivery_receipt.json` |
| Maintenance | `artifacts/reports/maintenance_delivery_receipt.json` |
| Engineering MCP integration | `artifacts/reports/engineering_delivery_receipt.json` |
| Architecture redesign | `artifacts/reports/architecture_delivery_receipt.json` |

Prior stage evidence, selected inference/training checkpoints and pinned backups remain retained.
Backups are local/same-disk and do not protect against disk loss. No backup deletion is authorized.
Optional engineering MCPs must be off after this task; runtime needs no engineering daemon.

Earlier architecture evidence/task snapshots are pinned and restore-tested; that delivery
published architecture documents/model and preserved the then-pending quality increment.
The full Stage 9 continuation has its own recovery/publication index. Require its exact checks,
snapshot restoration, code-bundle clone and origin/main readback before relying on delivery.
Same-disk recovery remains the limit; no backup deletion is authorized.

Next action: await separate owner authorization for further work. Stage 9 is accepted and complete
in the local engineering/research scope; product-quality criteria remain open. Verify the owner
acceptance delivery index when resuming. Do not repeat completed experiments/confirmation, alter
negative results or exposed labels, recruit external help, claim production readiness or start
Stage 10 without separate authorization.
