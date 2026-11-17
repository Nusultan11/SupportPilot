# PROJECT — requirements

## Goal

A measured, reproducible support assistant for a fictional Kazakhstan online store (KZT).
Languages: RU, KK, EN and mixed RU–KK. Customers ask free-form questions; unresolved cases go
to an operator with a summary and evidence. Quality and verifiability beat technology count.

## Functional requirements

| ID | Requirement |
|---|---|
| F1 | Order and delivery tracking through an authorized orders API |
| F2 | Return/refund policy from the knowledge source; refund/order state from tools |
| F3 | Payments, cancellation, order changes; policy kept separate from live state |
| F4 | Product attributes from the knowledge source; availability from the inventory API |
| F5 | Store rules, terms and FAQ with evidence |
| F6 | Intent/action routing (taxonomy v1) |
| F7 | One targeted clarification for an ambiguous request |
| F8 | Escalation: complaints, legal/safety/privacy, repeated failure, explicit request, low confidence |
| F9 | Detect RU/KK/EN and code-switching; reply in the user's language |
| F10 | Cite evidence; separate facts, assumptions and missing information |
| F11 | Bounded session memory with carried slots such as order ID |

Implementation status: [PROGRESS](PROGRESS.md).

## Non-functional requirements

| ID | Requirement |
|---|---|
| N1 | No unsupported policies, prices, state or customer facts |
| N2 | Evaluate each language separately, including KK and mixed RU–KK |
| N3 | Measure end-to-end latency (p95) |
| N4 | Trace every dataset, index and model to code, manifest, config and seed |
| N5 | Streaming/batch, incremental data processing |
| N6 | No real PII; redacted logs; protected secrets; injection and authorization tests |
| N7 | Fit 4 GB GPU / 15.4 GB RAM or authorized free cloud; paid use needs approval |
| N8 | Bounded timeouts/retries, typed failures, fallbacks, health checks |

## Out of scope

General document platform, CRM/marketplace, payment execution, real customer data,
voice/images, a full operator workspace. Expanding scope needs an owner decision.
