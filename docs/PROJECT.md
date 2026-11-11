# PROJECT — goal, scope, requirements

## Goal

An end-to-end, reproducible, measured ML system that resolves online-store customer requests
in RU/KK/EN with grounded, cited answers and correct escalation to humans. Quality and
verifiability beat the number of technologies used.

## Users and setting

- **Customer**: writes free text (chat) in RU, KK, EN or mixed RU–KK; may be vague, angry, or off-topic.
- **Operator**: receives escalations with a summary, detected intent, and the evidence gathered.
- **Store**: a fictional Kazakhstan-based online store ("demo store"). It uses KZT and has
  synthetic policies, catalogue, customers and orders.

## In scope (functional requirements)

| ID | Requirement |
|---|---|
| F1 | Delivery and order tracking (via orders API) |
| F2 | Returns and refunds: policy (KB) and status (payments/orders API) |
| F3 | Payment methods and issues; cancelling or changing an order (policy from the KB, state from the API) |
| F4 | Product attributes (KB catalogue) and availability (inventory API) |
| F5 | Store rules, terms, FAQ (KB) |
| F6 | Intent detection for routing to an action |
| F7 | Ambiguous request → one targeted clarification question |
| F8 | Escalation to a human for complaints, legal/safety topics, repeated failure, explicit request, low confidence |
| F9 | Language ID (en/ru/kk + code-switch) and reply in the user's language |
| F10 | Answers cite KB chunks. The reply separates facts, assumptions and missing information |
| F11 | Short dialogue memory: the current session only, bounded turns, order id carried over |

## Out of scope

General document-analysis platform, CRM, marketplace, operator workspace beyond a simple
escalation queue, payment execution, real customer data, voice, and images. Any expansion
needs an ADR.

## Non-functional requirements

| ID | Requirement | Initial target |
|---|---|---|
| N1 | Groundedness | No unsupported policy/price/status claims (metric targets: EVALUATION §2) |
| N2 | Language | Reply-language compliance tracked per language; KK reported separately |
| N3 | Latency (local demo, 4 GB GPU) | p95 measured and reported. The target is set in Stage 9 after the baselines |
| N4 | Reproducibility | Every dataset, index and model is traceable to code commit + data manifest + seed |
| N5 | Scale readiness | Pipelines stream/batch and support incremental updates; stress-tested at 1M+ records (Stage 9) |
| N6 | Privacy and security | No PII in logs. Prompt-injection tests. Secrets only in env |
| N7 | Cost | Runs locally; training on free cloud GPUs; paid APIs only with user approval |
| N8 | Robustness | Timeouts, retries and fallbacks for every external call; health checks |

## Units of measure (do not mix)

Customer request ≠ SFT example ≠ KB document ≠ chunk ≠ token. Targets: ~100k *quality* SFT
examples (planned EN 30% / RU 35% / KK 35%) and a 90k/5k/5k split. These are targets. The
actual size is set by quality gates in Stages 3–4. Duplicating data to reach them is forbidden.

## Success criteria

The project is complete when Stage 10's definition of done is met (see ROADMAP). Metric targets
are in EVALUATION.md. They are goals, not guarantees, and misses are reported with error analysis.
