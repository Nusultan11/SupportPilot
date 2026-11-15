# PROGRESS — current handoff

Updated 2026-11-15. This file owns verified state, authorization and next action. Stage criteria
live in ROADMAP, interpretation in reviews, commands in RUNBOOK and raw evidence in private artifacts.

## State and authorization

| Scope | State | Remaining limitation |
|---|---|---|
| Stages 0–6.1 | COMPLETED | Synthetic retrieval/data and independent-review limits remain |
| Stage 7 | COMPLETED: engineering and research assessment | Autonomous factual support NOT_READY; critical escalation NOT_ACCEPTED |
| Stage 8 | COMPLETED: guarded local engineering prototype | Semantic quality NOT_EVALUATED; local recording is not external operator response |
| Stage 9 | PARTIALLY COMPLETED | Engineering assessment delivered; intent/critical/independent-semantic acceptance incomplete |
| Stage 9 quality continuation / 9Q1 | IN_PROGRESS, unpublished staged increment | Development selected; confirmation, recovery and publication unfinished |
| Maintenance and engineering MCP integration | COMPLETED: separate owner-authorized tasks | Prior recovery/publication receipts below; MCPs opt-in |
| Architecture and development-control redesign | COMPLETED: separate engineering task | Product-quality gates remain open; verify delivery receipt below |
| Stage 10 | NOT_STARTED | Separate owner authorization after quality acceptance or an explicit changed delivery goal |

The active task is architecture/control redesign, not a quality experiment or new product stage.
The already authorized 9Q1 continuation remains separate: no confirmation is consumed by this task.
Do not tune exposed Stage 7/Stage 9 outcomes, relabel flagged rows or replace sealed artifacts.
Independent fluent reviewers and labelled calibration are unavailable; semantic acceptance remains
NOT_EVALUATED. ROADMAP 9Q2/9Q3 stay open. A passing engineering delivery cannot waive those gates.

## Runtime and preserved identities

The delivered baseline is a single-worker FastAPI/Streamlit synthetic-store prototype with
SQLite tools, bounded ephemeral sessions and durable queue/replay. Generation is off by default;
all service KB candidates are withheld and queued. Authorized tool values may be delivered;
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

The separately authorized 9Q1 continuation is present but not published. Its draft protocol and
ADR-037 remain staged with its code; they are not accepted/published by architecture maintenance.
Canonical private evidence: `artifacts/reports/system/quality/`; fresh inputs:
`authored/system/quality.yaml`, freeze `f77c5353fb3b85df5c82` under `data/benchmarks/system_quality/`.
The original Stage 9 source/protocol/results and Stage 7 selection are unchanged.

Development control `4b426acf3fb74419b6082f6171bff559` and candidate
`b73821f9c83a49adbf1e902a422895ba` are terminal. `selection_pointer.json` names the content-addressed
seal `501740b786cb90f5cc624ecb6e9acfc186cf8144b2eab7e9e15c4584c943b99b`.
The prior seal is retained/rejected for wrong-type reference reuse; regression and corrected
selection precede confirmation. Audit retains2 lexical,133 prior embedding and80 cross-split row
flags: disjoint family IDs do not certify independence. Eight critical draft needs/language do
not meet broad safety coverage. No confirmation attempt is present at architecture audit.
Its retained `checks/pytest_final.log` records829 passed/zero skipped; `checks/precommit.log`
passes. Recheck source/seal/check identities before continuation, rather than assuming publication.

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
| Maintenance | `artifacts/reports/maintenance_delivery_receipt.json` |
| Engineering MCP integration | `artifacts/reports/engineering_delivery_receipt.json` |
| Architecture redesign | `artifacts/reports/architecture_delivery_receipt.json` |

Prior stage evidence, selected inference/training checkpoints and pinned backups remain retained.
Backups are local/same-disk and do not protect against disk loss. No backup deletion is authorized.
Optional engineering MCPs must be off after this task; runtime needs no engineering daemon.

Architecture evidence/task snapshots are pinned, deep-verified and actually restored; code-bundle
restoration and exact origin/main publication are indexed by the final receipt. Only architecture
documents/model are published; pending Stage 9 source/protocol/results remain staged. Same-disk
recovery remains the limit.

Next: resume only the existing authorized9Q1 scope: verify selected runtime/inputs and checks,
run each sealed confirmation variant once, interpret without retuning and finish its own recovery/
publication. Independent quality dependencies remain open. Stop before Stage 10.
