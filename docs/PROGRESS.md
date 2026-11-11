# PROGRESS: current state (single source of truth for status)

Last updated: 2026-11-11

## Current stage

**Stage 3: EDA and data quality engineering. Status: COMPLETED.** Waiting for owner
confirmation to start Stage 4 (multilingual dataset engineering).

| Stage | Status | Commits |
|---|---|---|
| 0 Audit and research | COMPLETED | `699dc23`, `cd50a6c`, `ced287e` |
| 1 Engineering foundation | COMPLETED | `6b17404`, `7b6bc4c` |
| 2 Data acquisition and source registry | COMPLETED | `a7bcc05`, `4384d13`, `f43dec0` |
| 3 EDA and data quality engineering | COMPLETED | see "Git and backups" |

## What exists (cumulative)

| Component | Path | Notes |
|---|---|---|
| Foundation (Stage 1) | `src/supportpilot/common/`, `ops/`, `cli.py`, `configs/base.yaml`, `compose.yaml` | config, logging/redaction, IDs, hashing, manifests, contracts, MLflow, backups, checks |
| Source registry | `configs/sources.yaml` | 16 sources reviewed; licence policy enforced by the schema (ADR-018) |
| Usage policy | `data/policy.py` | acquire/convert/analysis ≥ reference_only, evaluation ≥ eval_only, training = train |
| Acquisition, conversion (Stage 2) | `data/acquire.py`, `converters.py`, `convert.py`, `records.py` | pinned, hash-checked raw data → `RawRecord` Parquet (ADR-019) |
| Cleaning (Stage 3) | `data/textnorm.py`, `placeholders.py`, `langid.py`, `clean.py` | repair, PII masking, synthetic slots, lingua LID, flags; checkpoint per source (ADR-020) |
| Labels (Stage 3) | `data/labels.py`, `configs/intent_map.yaml` | draft v0 map, rule-based `needs_reference` action, `suspect_unless` label-noise rule (ADR-021) |
| Dedup and grouping (Stage 3) | `data/neardup.py`, `build.py` | exact + MinHash/LSH + exact Jaccard, label guard, families and scenarios (ADR-022) |
| Quality gates (Stage 3) | `data/gates.py` | 9 gates + `splits_and_leakage: not_applicable`; failing build not swapped in |
| EDA (Stage 3) | `data/eda.py`, `report.py`, `review.py`, `docs/reviews/stage3_review.yaml` | profile, 12 charts, tables, review samples, hand-written review |
| CLI | `supportpilot data sources/acquire/convert/verify/clean/gates/eda` | RUNBOOK §7, §7b |

## Data state (Stage 3 build `canonical-20261111-8fe01586`)

| source_id | allowed_use | records | distinct texts | families (J ≥ 0.8) | scenarios | canonical | pending | rejected |
|---|---|---|---|---|---|---|---|---|
| bitext_retail_ecom | train | 44,884 | 43,915 | 34,590 | 46 | 35,980 | 7,937 | 967 |
| bitext_customer_support | train | 26,872 | 23,513 | 20,511 | 32 | 19,668 | 3,845 | 3,359 |
| whileai_ecommerce_intent | train | 24,710 | 24,533 | 24,520 | 24,520 | 0 | 24,533 | 177 |
| support_json_ru | reference_only | 716 | 716 | 91 | 26 | 0 | 716 | 0 |
| belebele | eval_only | 2,700 | 2,700 | 789 | 789 | 0 | 2,700 | 0 |
| **total** | | **99,882** | | | | **55,648** | **39,731** | **4,503** |

- **Canonical:** 55,648 EN rows from Bitext; 23 v0 intents; 45,460 families; **55 scenarios**.
  Actions: answer_kb 30,351, clarify 19,354, call_tool 2,996, escalate 2,947. Every row is
  flagged `labels:intent_map_draft_v0`.
- **Pending reasons:**
  - record_type:dialogue 24,533 (while-ai);
  - intent_unmapped 11,494 (11 Bitext labels without a v0 intent);
  - allowed_use:eval_only 2,700;
  - allowed_use:reference_only 716;
  - label_suspect 237;
  - label_conflict 51.
- **Rejected reasons:**
  - exact_duplicate 4,327 (CS 3,359 = 12.5%, retail 967 = 2.2%, while-ai 1);
  - garbled_text 147;
  - unsupported_language 16;
  - malformed_placeholder 11;
  - encoding_damage 2.
- **Independent scenarios.** Bitext (71,756 rows) has 68 scenarios together, plus 18
  scenarios joined across retail and CS. Volume is about 1,000× larger than real diversity.
- **Overlap.** Retail and CS share 27 exact texts and 129 near-duplicate pairs at J ≥ 0.8.
  No other source overlaps.
- **Token lengths (Qwen3 tokenizer, p50 / p95 / max):**
  - Bitext retail text 13 / 19 / 27, response 175 / 278 / 503;
  - Bitext CS text 10 / 15 / 24, response 104 / 256 / 478;
  - while-ai first turn 26 / 137 / 554, whole dialogue 274 / 1,595 / 10,144;
  - support-json-ru text 30 / 44 / 50;
  - Belebele question 25 / 61 / 104.

  The Qwen3.5 tokenizer is equal on EN and 20–27% shorter on RU and KK text (Belebele
  p50 20 vs 25, support-json-ru 22 vs 30).
- **Register.** In Bitext retail, 49% of rows contain inserted profanity and 51% carry the
  typo tag. In CS it is 5% profanity and 20% typos. while-ai has emoji in 14% of rows, and
  7% of rows contain e-mail-like PII, now masked.
- **Language ID.**
  - Belebele calibration: questions en/ru 100%, kk 99.89% (1 KK question without
    KK-specific letters went to ru); passages 100%.
  - Canonical LID agreement is 100%. 9% of canonical rows have low confidence (typo-heavy
    short EN), and all reviewed ones are English.
- Full report: `artifacts/reports/eda/report.md` (git-ignored, backed up). Regenerate with
  `supportpilot data eda`.

## Quality gates (build canonical-20261111-8fe01586)

| gate | result |
|---|---|
| accounting | pass: 99,882 in = 99,882 out, no record in two outputs |
| schema | pass: validity 1.0 (≥ 0.995); canonical 55,648/55,648, pending 39,731/39,731, rejected 4,503/4,503 |
| ids | pass: no duplicate or non-recomputable example ids, no family in 2 scenarios, no dangling `duplicate_of` |
| language | pass: LID re-run on canonical 55,648/55,648 = lang (≥ 0.99) |
| duplicates | pass: 0 exact duplicates and 0 identical texts in canonical; 0 compatible J ≥ 0.8 pairs across families; 2,441 near-identical (J ≥ 0.8) different-label pairs kept apart by the label guard |
| placeholders | pass: no `{{` in text; all slot values are from the catalogue |
| pii | pass: no unmasked e-mail, phone or card |
| licence | pass: canonical only from train sources; allowed_use recorded on every pending/rejected row |
| labels | pass: only v0 intents; no escalation without a reason |
| splits_and_leakage | not applicable: splits are made in Stage 4 |

## Qualitative review (LLM agent, not an independent human review)

1,024 records were read and coded: 100 random + 100 flagged per source, and 24 flagged for
Belebele (all it had). Annotations are in `docs/reviews/stage3_review.yaml`.

| source | random: OK / hard defect | main findings |
|---|---|---|
| Bitext retail | 42% / 9% | profanity inserted mechanically (53/100); responses with prompt leaks ("offensive language used in your instruction", a "previous response" that does not exist); systematic mislabel: 265 product-information rows in return_product_in_store (now `label_suspect`); store/online/policy return variants conflict |
| Bitext CS | 59% / 6% | placeholders filled; responses keep placeholders when the text has a literal value; contract-fee intent off-domain; over-the-top response style; Stage 3 currency mismatch found and fixed |
| while-ai | 43% / 52% | about half off-domain (crypto, P2P, bills, homework); labels describe the whole dialogue; garbled turns (147 rejected, single-script word salad still missed); PII-like data masked |
| support-json-ru | 0% / 100% (off-domain by design) | decisions consistent with policies; template grammar errors in 89/716 rows ("1 дней"); human_reviewed=false |
| Belebele | 82% / 0% | reliable language labels; KK translation issues in about 6/33 KK items; CSV quote artefacts in passages |

"OK" is strict (any soft code counts as not OK). Hard codes are LABEL_WRONG, GARBLED,
WEAK_TEXT, OFF_DOMAIN, LANG_OTHER, LID_ERROR and RESPONSE_DEFECT.

## Verification evidence (Stage 3, 2026-11-11)

- **Checks:**
  - `ruff check`, `ruff format --check` and `mypy --strict` (57 files) are clean;
  - `pre-commit run --all-files` passes;
  - `pytest`: **233 passed, 0 skipped**, with the Qdrant server up.
- **New tests: 59.**
  - Unit: text repair, profanity and fuzzy variants (no false hits on "funky", "fluke"),
    PII masking (including order numbers not masked), garbled detection, placeholder
    filling and classification, intent-map validation and rules, LID (en/ru/kk/und/other,
    names vs code-switch), shingles, MinHash determinism, LSH, exact Jaccard (including the
    chunked path).
  - Integration: the end-to-end pipeline on synthetic sources. It covers routing reasons,
    template and literal duplicates, the label guard, scenarios split by action, slots in
    response, PII, lineage and untouched raw records, skip and stale-partial cleanup, forced
    rebuild byte-identical, crash mid-build, failed gate, corrupt records, missing clean
    checkpoint, and the CLI (`clean --all`, `gates`, `eda --no-tokens`).
- **Real data:**
  - The full forced build gives the same `dataset_version` on every run.
  - The final forced run on the committed code: 10 of 10 clean and canonical Parquet files
    byte-identical to the previous run.
  - Peak RSS was 2.84 GiB. The full run takes 220 s (clean 115 s, build 104 s), and EDA
    takes 63 s.
- **Real interruption.** `taskkill /F` was run twice: during grouping, and while
  `.partial-canonical-*` was being written. Both times the previous build stayed intact,
  and `data verify --all` and `data gates` exited 0. A re-run skipped everything in 2.8 s and
  removed the leftover partial.
- **Bugs found and fixed during the stage:**
  - Non-deterministic MinHash: shingle ids followed Python set order, which is
    hash-randomised.
  - Template/literal duplicates were missed: 242 identical canonical texts, found by the
    duplicates gate.
  - The label check in duplicate groups was first-vs-others instead of pairwise.
  - Card masking hit glued order numbers.
  - The currency symbol disagreed between text and response.
  - A CLI test resolved the real project root. The test is now pinned to `tmp`, and the
    real outputs were rebuilt identically.

## Known limitations and open items

1. **No RU/KK training data** yet (unchanged). It is the core of Stage 4.
2. **Split unit (ADR-022).** Bitext has 68 scenarios. Splitting by `scenario_id` as
   DATA_CONTRACTS says would remove intents from test, and splitting by family leaves
   paraphrase near-copies (J 0.5–0.8) across splits. Stage 4 must decide, and generalisation
   must be measured on the independent gold set.
3. **Draft labels (ADR-021).** 11 Bitext labels are unmapped. Taxonomy decisions are open:
   customer_service vs human_agent, technical support, cart, `pay`, GDPR requests,
   shipping-address changes, and the return/tracking conflicts. Actions come from rules.
4. **Reference responses** are unfit as training targets. They contain unresolved store facts
   (65% of retail rows), template instructions and prompt leaks. Stage 4 targets come from
   the training KB variant.
5. **Detector gaps.**
   - Single-script word salad is not detected (3 of 100 reviewed flagged while-ai rows).
   - Obfuscated e-mails ("name dot surname at gmail") are not masked.
   - Latin-script Kazakh is not detected.
6. **The review is by an LLM agent** on truncated displays. KK/RU quality needs a fluent
   human reviewer (open decision 2).
7. Still open from earlier stages: gated sources (HF account), `uv sync --no-group gpu` not
   run, no LICENSE file, paid-API budget, reviewers, Kaggle/Colab accounts.

## Git and backups

- Remote: `origin` = https://github.com/Nusultan11/SupportPilot (public); `main` is pushed
  after each stage.
- Backups live in `backups/` at the project root (git-ignored, ADR-013).
- Code bundles (`backups/code/`): Stage 0 `…T104345Z`, Stage 1 `…T115429Z`, Stage 2
  `…T134356Z`, Stage 3 see below.
- Data snapshots (`backups/data/`): pinned raw snapshots per source from Stage 2, plus
  `raw_records/20261111T134036Z-ee0f165b` and `env_reports/20261111T113155Z-92ddf171`.
- **Stage 3 data snapshots** (pinned, deep-verified):
  - `clean_records/20261111T154052Z-5d85c3cb` (85.0 MB);
  - `canonical/20261111T154055Z-c62f57ae` (51.1 MB, build `canonical-20261111-8fe01586`);
  - `eda_report/20261111T154057Z-9fe64e24` (1.2 MB, report, figures, review samples).
- **Stage 3 code:** commit `ops:`/`data:` on `main` (hash in the git log), plus the bundle
  `backups/code/` created after the commit.

## Notes for the next stage (Stage 4)

- Load data only through `scan_records(..., purpose)`. Read the canonical build from
  `data/interim/canonical/` and check the source policy before training use (canonical holds
  train sources only; pending holds all of them).
- Use canonical texts and labels as the EN backbone. Cap rows per family, and down-sample
  the profane register.
- Do not train on `reference_response`.
- Pending while-ai dialogues: take only the in-domain ones, relabel per dialogue, and run a
  coherence pass.
- Localise money (currently `$`/USD slots) to KZT together with the response during translation.
- Decide the split unit with the ADR-022 evidence before building splits. Leakage gates
  then become active.

## Next recommended action

Owner confirmation, then Stage 4 (ROADMAP §Stage 4).
