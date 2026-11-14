# PROJECT — requirements

## Goal and users

Build a reproducible, measured online-store support assistant for RU/KK/EN and mixed RU–KK.
Customers ask free-form questions; operators receive unresolved cases with a summary and
evidence. The fictional Kazakhstan store uses KZT and synthetic policies/catalogue/orders.
Quality and verifiability take priority over technology count.

## Functional scope

| ID | Requirement |
|---|---|
| F1 | Delivery and order tracking through an authorized orders API |
| F2 | Return/refund policy from KB; refund/order state from tools |
| F3 | Payment methods/issues, order cancellation/change; distinguish policy from state |
| F4 | Product attributes from KB; availability from inventory API |
| F5 | Store rules, terms and FAQ with evidence |
| F6 | Intent/action routing |
| F7 | One targeted clarification for an ambiguous request |
| F8 | Human assistance for complaints, legal/safety/privacy risks, repeated failure, explicit request or insufficient confidence |
| F9 | Detect RU/KK/EN and code-switching; reply in the user's language |
| F10 | Cite policy evidence; separate supported facts, assumptions and missing information |
| F11 | Bounded session-only dialogue memory and carried slots such as order ID |

These are product requirements, not claims that every feature is implemented.
Implementation status belongs to [PROGRESS](PROGRESS.md).

## Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | No unsupported policies, prices, status or customer facts; insufficient evidence leads to clarification/abstention/escalation |
| N2 | Evaluate every language separately, including KK and mixed RU–KK |
| N3 | Measure local end-to-end p95; establish the serving target in Stage 9 |
| N4 | Trace datasets, indexes and models to code, manifest, configuration and seed |
| N5 | Stream/batch and incrementally update data; test 1M+ records in Stage 9 |
| N6 | No real customer PII; redact logs, protect secrets, test prompt injection and authorization |
| N7 | Fit the local 4 GB GPU/15.4 GB RAM constraints or use authorized free cloud compute; paid use requires approval |
| N8 | Bound external-call timeouts/retries and provide typed failures, fallbacks and health checks |

Current numerical acceptance targets and review requirements: [EVALUATION](EVALUATION.md).

## Boundaries

Excluded: general document-analysis platform, CRM/marketplace, payment execution,
real customer data, voice/images and an operator workspace beyond a simple escalation queue.
Expansion needs an explicit scope decision.

Facts live in KB/tools, not weights. Training uses altered fictional KB values.
A request, SFT row, document, chunk and token are different counting units.
The original ~100k SFT / 90k–5k–5k split and EN/RU/KK 30/35/35 distribution are planning
targets bounded by quality, not quotas to satisfy through duplication.

## Success

Complete the [Stage 10 acceptance criteria](ROADMAP.md#stage-10--production-readiness-and-delivery).
Report missed targets and uncertainty with error analysis; valid JSON or a running demo alone
does not establish grounded quality. Detailed feature dependencies live in ROADMAP.
