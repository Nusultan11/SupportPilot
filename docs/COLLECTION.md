# COLLECTION — commissioned native-speaker data and independent review

Reader: the owner (approval, payments) and the agent running the collection. Decision and
alternatives: [ADR-049](DECISIONS.md#adr-049--commissioned-collection-and-independent-review-2026-11-17).
Contracts and checks: [DATA_CONTRACTS §10](DATA_CONTRACTS.md#10-commissioned-collection-configscollectionyaml-adr-049).
Commands: [RUNBOOK §4.1](RUNBOOK.md#41-commissioned-data-and-independent-review-adr-049-collection).

## 1. Target (Stage 1 acceptance)

| Slice | Rows | Scenarios | Writers | Writers per scenario |
|---|---|---|---|---|
| RU | 1,050 | 350 (35 intents × 10) | ≥ 6 native RU | 3 |
| KK | 1,050 | 350 | ≥ 6 native KK | 3 |
| mixed RU–KK | 1,050 | 350 | ≥ 6 native KK, fluent RU | 3 |

Review: 930 rows blind-labelled by reviewer A (100 EN, 100 RU, 100 KK existing core rows +
210 per commissioned slice); a seeded 20 % (186 rows) relabelled by reviewer B. Reviewers write
no rows. Rows, scenarios and writers are counted separately (`data collection check`).

## 2. Chosen route and price (awaiting owner approval)

Direct contracts with individual freelancers on Upwork, residing in Kazakhstan, fixed price per
accepted row. Upwork User Agreement §6.4: on full payment all rights in the work product pass to
the client; the job post and contract add an explicit grant for ML training, evaluation and
publication of derived data and models, plus the attestations in §4.

| Item | Quantity | Unit | Amount (USD) |
|---|---|---|---|
| Writers, accepted rows | 3,150 | 0.30 | 945.00 |
| Reviewers, labelled rows (A 930 + B 186) | 1,116 | 0.15 | 167.40 |
| Paid screening tests | ≤ 20 | 3.00 | ≤ 60.00 |
| Contractor payments | | | ≤ 1,172.40 |
| Upwork client marketplace fee (≤ 7.99 %) | | | ≤ 93.68 |
| Upwork contract initiation fees (≤ 20 × 14.99) | | | ≤ 299.80 |
| VAT on platform fees, if charged (16 %) | | | ≤ 62.96 |
| **Maximum total** | | | **≤ 1,628.84** |

Rates assume 30 rows/hour writing and 60 rows/hour labelling (≈ USD 9/hour), above the
USD 5/hour seen for Kazakh linguistic evaluation posts on Upwork (2026-10). Only accepted
rows are paid; rejected rows may be rewritten once. Expected duration: 1 week recruiting,
2 weeks writing, 1 week review.

The owner holds the Upwork client account and payment method and funds/releases milestones;
the agent prepares posts, tests, selection, checks and the exact release list per milestone.

## 3. Process

1. Reviewers first: 2 contractors, KK + RU native or fluent, EN fluent. Paid screening:
   30 items with known labels (10 EN, 10 RU, 10 KK from existing core rows); pass ≥ 0.8.
2. Writers: 12 contractors (6 KK natives fluent in RU write the KK and mixed slices,
   6 RU natives write RU). Paid screening: 5 briefs in their slice, judged by both reviewers
   for language, naturalness and intent fit; pass = 5/5 acceptable.
3. Briefs: `data collection briefs`; each writer gets 175 briefs per slice, each brief goes to
   3 different writers per slice (seeded assignment).
4. Writing through a submission form that timestamps each row (`duration_sec`).
5. Each batch: `data collection check`. Rows failing blocking checks are returned once.
6. Final delivery: registry entry + `data collection intake` + recipe entry +
   `data corpus build` → new immutable corpus version.
7. Review: `data review package` on the new version → sheets to reviewers (key never shared)
   → `data review score`. Kappa < 0.6 in a stratum: that stratum's labels are not
   `human_reviewed`; disagreements are adjudicated and recorded, never silently relabelled.
8. Stage 1 is `COMPLETED` only when §1 targets are met, all checks pass and every stratum has
   kappa ≥ 0.6 and ≥ 10 % double labelling.

## 4. Writer and reviewer terms

Writers write each request themselves, as a customer of an online store in Kazakhstan would
type in chat or e-mail, in the slice language (mixed: KK and RU mixed as in everyday speech).
They follow the brief's facts but never copy its wording, use placeholders (`<order_id>`,
`<name>`, `<phone>`) instead of any real personal data, and write one row per brief.

Every contributor signs five attestations (DATA_CONTRACTS §10): original work; no LLM or text
generator; no machine translation; no real personal data; all rights assigned, including use
for training and evaluating ML models and publishing derived datasets and models.
Contributors are recorded under pseudonymous ids; real identities and contracts stay in the
owner's Upwork account, never in the repository.

## 5. Limits of the evidence

Code cannot prove human authorship. Evidence is layered and reported separately: signed
attestations, screening, per-row timing, near-duplicate and language flags, and the
independent reviewers' naturalness judgement. A writer whose rows show generator or
translation patterns is removed with all their rows.
