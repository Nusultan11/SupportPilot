# PROGRESS — current state

Updated 2026-11-17. Short handoff: state, evidence, next action. Not a log.

## State

| Scope | Status |
|---|---|
| Project reset ([ADR-046](DECISIONS.md#adr-046--project-reset-2026-11-17)) | COMPLETED |
| Stage 1 — Data foundation ([ADR-047](DECISIONS.md#adr-047--stage-1-corpus-from-human-authored-sources-2026-11-17), [ADR-048](DECISIONS.md#adr-048--stage-1-completion-round-2026-11-17), [ADR-049](DECISIONS.md#adr-049--commissioned-collection-and-independent-review-2026-11-17)) | PARTIALLY COMPLETED, BLOCKED on one cost approval: no natural in-domain KK / RU–KK and almost no RU data; independent human review NOT_EVALUATED. Intake and review tooling ready; collection not started |
| Stages 2–5 | NOT_STARTED; Stage 2 needs owner authorization |

Earlier stages 0–9 are retired. Their code is in Git at `1561971`; their data and evidence
are in `backups/data/`, `artifacts/` and `mlruns/`. Do not resume them.

## What exists now

- CLI: `env`, `check mlflow`, `backup`, `data sources|acquire|verify|taxonomy|corpus|collection|review`.
- Commissioned collection (ADR-049, [COLLECTION](COLLECTION.md)): plan `configs/collection.yaml`,
  350 scenario briefs, delivery checks + immutable intake, `reader: commissioned` in the corpus
  build, blind review packages with Cohen's kappa. Tested on fixtures only; no delivery exists.
- Blind review package for the current corpus (300 existing EN/RU/KK rows, 60 double):
  `artifacts/review/corpus-20261117-26d02054/` (regenerate with `data review package --strata
  en_existing,ru_existing,kk_existing`); not sent to anyone.
- Sources: `configs/sources.yaml` (every investigated source with its decision); raw copies in
  `data/raw/`, verified. Taxonomy v1: `configs/taxonomy.yaml`. Recipe: `configs/corpus.yaml`.
- Corpus `data/corpus/corpus-20261117-26d02054/` (backup `corpus`, pinned; supersedes
  `379aa455`). Reproducible: a rebuild returns the same version (`created: false`). Its
  `REPORT.md` has the full per-language report. Leakage and residual-PII gates: all 0.

| Intent corpus (core) | EN | RU | KK | mixed RU–KK |
|---|---|---|---|---|
| rows / independent families | 34,914 / 30,884 | 10,794 / 10,237 | 636 / 612 | 0 |
| non-out-of-scope rows / families | 6,975 / 6,273 | 59 / 58 | 0 | 0 |
| of which retail-support rows (families) | 6,529 (5,892), role-play | 0 | 0 | 0 |
| v1 intents with ≥1 core row (of 35) | 21 | 4 | 1 (`out_of_scope`) | 0 |

- EN: ABCD role-play (retail) + CLINC150/MASSIVE (greetings, out-of-scope). Missing 14 intents
  (app_help, cart_help, contact_info, delivery_time, feedback, human_agent, invoice, newsletter,
  order_history, payment_methods, personal_data, place_order, product_defect, unclear).
- RU: 39 MINDS-14 banking voice requests (ASR text; `account_help` 24, `technical_issue` 15),
  20 human-localised MASSIVE greetings, 10,735 human-localised out-of-scope rows.
- KK: 636 human-translated xSID out-of-scope rows, validation/test only. No natural KK text.
- Mixed RU–KK: none. The one row v1 flagged was a false positive (fixed detector, ADR-048).
- Grounded (doc2dial, EN): 19,086 turns, 383 document families, split 15,292 / 1,893 / 1,901.

## Label quality

| Method | Result |
|---|---|
| Automatic | schema, licence/policy, dedup, leakage, PII, language-ID gates pass (REPORT.md) |
| Agent review (LLM, not independent) | seeded stratified sample of 307 rows of v1 (`artifacts/reports/stage1_completion/agent_label_audit.csv`): EN in-domain 12/147 wrong, 6 debatable; RU in-domain 24/60 wrong (one systematic rule), 1 debatable; EN / RU / KK out-of-scope 0/30, 0/30, 0/40. All 36 wrong labels are corrected in `26d02054` |
| Independent human review | NOT_EVALUATED: no reviewer engaged. The version's `review/` sheets show the label (verification, not agreement); the blind package with double labelling is listed above |

## Data research (2026-11-17)

HF Hub, arXiv/ACL/TechRxiv, ISSAI, J. Open Humanities Data, GitHub, Kaggle and vendor catalogues.
No new source passed licence + human authorship + support domain (decisions in `sources.yaml`,
section ADR-048). Closest natural KK/RU–KK support data is private (telecom operator) or commercial.
Collection routes (Prolific, Toloka, Yandex Tasks, Upwork, vendors, bureaus) compared in ADR-049.

## Next action — one owner decision (cost)

Approve or reject: Upwork direct contracts with Kazakhstan-resident freelancers,
**maximum USD 1,628.84** ([COLLECTION §2](COLLECTION.md#2-chosen-route-and-price-awaiting-owner-approval)):
12 writers × 3,150 accepted rows (RU, KK, mixed RU–KK; 350 scenarios × 3 writers per slice)
at USD 0.30; 2 independent reviewers × 1,116 blind labels at USD 0.15; ≤ 20 paid screening
tests at USD 3; Upwork fees and possible VAT included as maxima. Rights: Upwork §6.4
assignment + explicit ML-use clause + five attestations. Needs the owner's Upwork client
account and payment method; the owner funds and releases milestones.

After approval the agent continues without a new brief (COLLECTION §3). No vendor quote was
requested: quote forms need the owner's contact details and replies go to a mailbox the agent
cannot read. Stage 2 must not start before the owner authorizes it.
