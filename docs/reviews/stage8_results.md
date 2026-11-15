# Stage 8 — guarded end-to-end local prototype

Audience: the owner continuing the project or reproducing the local demo. This report owns
Stage 8 interpretation; commands are in RUNBOOK §18, current handoff in PROGRESS, and private
raw evidence/recovery/publication receipts under `artifacts/reports/`.

## Delivery and scope

The local prototype integrates the frozen encoder router, selected retrieval pipeline,
optional frozen-adapter inference, existing validators and safe fallback with FastAPI,
Streamlit, authorized synthetic SQLite tools, bounded sessions and a durable operator queue.
Customers can query their order/payment state and shared product stock, carry an authorized
reference across turns, clarify missing parameters, and record a confirmed cancellation
request. A recorded cancellation request is not completed cancellation or payment execution.
Operators alone can advance open -> in_progress -> resolved. Queue recording is confirmed by
the committed SQLite record, not by an LLM action. It does not promise external delivery or
an operator response. Independent guards precede learned routing; subsequent fallback cannot
downgrade an existing critical reason. Context is bounded structured summaries, allowed risk
cues, owned references and evidence IDs, not raw customer text or model output. ADR-034 owns
the engineering choices and alternatives.

## Observed engineering checks

- Full shared pytest: **789 passed, zero skipped**, 440.48 seconds. One known embedded-Qdrant
  exact-search warning; real GPU/Qdrant checks executed. The final added critical-reason
  regression is included in the subsequent Stage 8 suite below.
- Final controlled Stage 8 suite: **324 passed**, 31.40 seconds. This includes 100
  scenario-state cases each for RU/KK/EN and separate auth, idempotency, ownership, injection,
  queue rollback, timeout, mixed-language, session and Streamlit checks.
- The 300 multilingual cases are ten customer situations crossed with ten distinct business
  states, not 300 independently authored linguistic needs. Their correlated draft templates
  prove engineering outcomes, not semantic or fluent-language quality. Mixed detection and
  explicit reply preferences have a separate three-case slice. No test/gold retuning occurred.
- Ruff check/format and strict mypy pass; the UI also passes a separate strict mypy run.
- Streamlit AppTest exercises chat/history and ticket refresh against the HTTP boundary.
  Actual Uvicorn/Streamlit loopback servers also served health/readiness, tool chat, queued
  escalation and the UI HTTP page. The task-owned servers were stopped after verification.

Raw checks: `artifacts/reports/service/checks/`; real-port receipt: `service/live_ports.json`.
The single-command suite plus real models completed in attempt
`e21763d4a5f5427e8dfd028fc4aa3c51`; it contains the final 324-case log and captured prompts/raw
completions. The final LF-normalized source/config probe is
`c2abd6e39cfc4bc792fe54d0c9ede6d3`; all its source hashes match the delivered code/config.
Both are under `artifacts/reports/service/attempts/`; `latest_probe.json` binds the latest report.

## Actual laptop observations

Final probe: CPU E5 router plus active GPU retrieval plus the selected Qwen3-0.6B adapter in
operational NF4, schema prompt, input limit6144/output512 and generation limit45s. This differs
from the immutable Stage 7 FP16/90s assessment configuration; no seal was rewritten.
Active KB/index: `kb_demo_store-20261113-f3b204a6` / `rindex-20261113-9ba94665`.

|Request scope|Observed seconds|Delivered outcome|
|---|---:|---|
|Model load/composition|19.190|All requested local components loaded|
|First EN policy request|34.718|Invalid raw schema; actual queued assistance|
|RU policy request|17.159|Invalid raw schema; actual queued assistance|
|KK policy request|30.116|Semantic UNKNOWN; actual queued assistance|
|EN authorized order|0.038|Deterministic authorized tool result|
|RU stock|0.023|Deterministic synthetic stock result|
|KK human request|0.023|Actual queued assistance|

There are exactly three actual model calls in this six-request probe. First-request latency
includes cold LID work; warm model generation itself was21.053/16.726/29.602s EN/RU/KK.
Maximum observed process RSS4,950,282,240 bytes; maximum Torch allocated/reserved VRAM
3,415,982,080/3,758,096,384 bytes. These are scoped observations, not complete machine peaks
or a guarantee against shared-memory spill. Sustained service, maximum-context residency,
load behavior and a p95 SLO remain NOT_ESTABLISHED; Stage 9 was not started. Paid compute0 USD;
local energy/hardware costs are excluded.

## Accounting, recovery and acceptance limits

All four operational attempts remain recorded in `service/tracking_receipts.json` and local
MLflow. Earlier attempts remain visible: the first included stale last-backend metrics on
non-model responses and overbroad KK guard matching; the first e2e load_seconds also included
the controlled-test prelude. Neither is silently promoted to additional inference evidence
or a controlled quality comparison. Later reports separate these scopes and retain raw calls.
The sandbox-only controlled test invocation was interrupted before observable results;
the successful authorized reruns do not establish its outcome.

Data/service state, reports and updated tracking are pinned, deep-verified and actually restored;
SQLite integrity and restored MLflow run identities are checked. Exact snapshots, byte checks,
code-bundle restore and origin/main proof are owned by
`artifacts/reports/stage8_delivery_receipt.json`. Backups are local/same disk and do not protect
against disk loss. No existing backup/model/dataset/index/seal was removed or rewritten.

**Completion decision:** Stage 8 is accepted as the guarded local engineering prototype once
the delivery receipt records successful recovery and publication. Autonomous factual service,
unrestricted critical-escalation recall and independent fluent KK quality are not accepted.
Semantic quality remains NOT_EVALUATED. Default generation is disabled because no calibrated
claim verifier can accept factual candidates; policy/product requests show retrieved source
material and receive actual queued assistance. These limitations are deliberate delivery
boundaries, not measured semantic success. Stop before Stage 9 until owner authorization.
