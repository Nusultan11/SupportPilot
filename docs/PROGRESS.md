# PROGRESS: current state (single source of truth for status)

Last updated: 2026-11-12

## Current stage

**Stage 4: Multilingual dataset engineering. Status: COMPLETED with documented gaps** (no
fluent human review of RU/KK and gold; SFT volume and held-out sets far below the 90k/5k/5k
targets; see "Stage 4 open items"). Waiting for owner confirmation before Stage 5 continues.
Note: a separate Stage 5 session is working in this checkout on new, uncommitted files
(`src/supportpilot/kb/**`, `tests/*/test_kb_*`, `kb_sources/demo_store/docs/**`,
`kb_sources/demo_store/catalog.yaml`). They are not part of Stage 4.

| Stage | Status | Commits |
|---|---|---|
| 0 Audit and research | COMPLETED | `699dc23`, `cd50a6c`, `ced287e` |
| 1 Engineering foundation | COMPLETED | `6b17404`, `7b6bc4c` |
| 2 Data acquisition and source registry | COMPLETED | `a7bcc05`, `4384d13`, `f43dec0` |
| 3 EDA and data quality engineering | COMPLETED | `b2f5d86`, `809b47b` |
| 4 Multilingual dataset engineering | COMPLETED (gaps documented) | `da28191`, plus a follow-up progress commit |

## What exists (cumulative)

| Component | Path | Notes |
|---|---|---|
| Foundation (Stage 1) | `src/supportpilot/common/`, `ops/`, `cli.py`, `configs/base.yaml`, `compose.yaml` | config, logging/redaction, IDs, hashing, manifests, contracts, MLflow, backups, checks |
| Source registry | `configs/sources.yaml` | 16 sources reviewed; licence policy enforced by the schema (ADR-018) |
| Usage policy | `data/policy.py` | acquire/convert/analysis ≥ reference_only, evaluation ≥ eval_only, training = train |
| Acquisition, conversion (Stage 2) | `data/acquire.py`, `converters.py`, `convert.py`, `records.py` | pinned, hash-checked raw data → `RawRecord` Parquet (ADR-019) |
| Cleaning, labels v0, dedup, gates, EDA (Stage 3) | `data/textnorm.py`, `placeholders.py`, `langid.py`, `clean.py`, `labels.py`, `neardup.py`, `build.py`, `gates.py`, `eda.py`, `report.py`, `review.py` | ADR-020..022; unchanged in Stage 4 except that its gate now uses `INTENTS_V0` |
| Taxonomy v1 (Stage 4) | `configs/taxonomy.yaml`, `data/taxonomy.py`, `data/select.py` | 35 intents, actions, conditions, rule-split labels (ADR-023) |
| Localisation, MT, QE (Stage 4) | `configs/locale_kz.yaml`, `data/locale.py`, `data/mt.py`, `data/mtrun.py`, `data/qe.py`, `data/embed.py` | KZT/cities/names, resumable MT cache, glossary, QE filter (ADR-024) |
| Training KB, answers, scenarios (Stage 4) | `authored/kb_train/`, `authored/responses/`, `authored/scenarios/`, `data/kbtrain.py`, `data/textfmt.py`, `data/scenarios.py` | LLM-agent-authored, native en/ru/kk (ADR-025) |
| Demo-store facts seed | `kb_sources/demo_store/facts.yaml` | values reserved for Stage 5; training KB must differ |
| SFT build, splits, gates (Stage 4) | `common/grounding.py`, `data/sft.py`, `data/sftbuild.py`, `data/split.py`, `data/sftgates.py`, `data/steps.py`, `data/stage4.py` | `sft/1` + `AssistantResponse` contracts, leakage checks (ADR-026) |
| Gold benchmark (Stage 4) | `authored/gold/`, `data/gold.py` | 312 scenarios × en/ru/kk, `gold/1`, review status per item |
| CLI | `supportpilot data …` | Stage 3 commands + `taxonomy`, `select`, `kb-train`, `mt-compare`, `translate`, `qe-score`, `sft`, `gold` (RUNBOOK §7c) |

## Stage 4 results (2026-11-12)

### Taxonomy v1 and relabelling (ADR-023)
- 35 intents (v0's 27 + `change_address`, `product_defect`, `contact_info`, `personal_data`,
  `newsletter`, `cart_help`, `app_help`, `technical_issue`), 10 escalation reasons, 4 tools.
- Input: Stage 3 canonical (55,648) + pending `intent_unmapped` (11,494) EN rows of train sources.
  Relabelled 65,550; excluded 1,592: label excluded 1,268 (`check_cancellation_fee`, `pay` with
  "transfer"), no rule matched 320 (`customer_service` without explicit wording), family split
  across labels 4. `label_conflict`/`label_suspect` rows stay out. while-ai: excluded entirely.
- EN backbone: one row per family, ≤ 260 per source label+action, profanity ≤ 10%:
  21,190 rows. MT queue: 100 rows per label+action and language = 7,554 RU + 7,554 KK.

### MT comparison and translation (ADR-024, `docs/reviews/stage4_mt_review.yaml`)
Major-error rate on 50 reviewed items per system (LLM reviewer, not human), and the
automatic QE accept rate on the 100-item sample (QE v1):

| system | RU major | KK major | RU auto-accept | KK auto-accept |
|---|---|---|---|---|
| Opus-MT en-ru / en-trk | 52% | 92% | 84% | 36% |
| M2M100-418M | 50% | 100% | 84% | 2% |
| MADLAD-400-3B nf4 | **40%** | **34%** | 80% | 69% |
| MADLAD pivot en→ru→kk | – | 52% | – | 69% |

- Automatic QE alone would have picked the wrong RU system and accepts many wrong KK
  outputs: the decision rests on the reviewed sample.
- Production (MADLAD, RTX 3050, nf4, 2 beams): KK 6,486 new + 7,141 back-translations in
  4,354 s + 4,090 s; RU 7,532 in 6,878 s, Opus-MT ru-en back-translation 372 s.
- QE (v2, with the `latin_word` rule added after the first KK audit): accepted RU 5,511 / 7,554
  (73%), KK 5,098 / 7,554 (67%). Top KK rejections: back-translation chrF < 0.30 1,637,
  lower-case Latin typo copied 510, untranslated Latin 150, length ratio 127, name lost 119,
  glossary (cart as fence/trash bin, item as article, human-agent calque) 197, Russian
  function words 101.
- **Post-QE audit** (deterministic sample, LLM reviewer): RU major errors 3/60 = 5%, minor
  14/60 = 23% (literal calques); KK major errors 11/55 = 20% (Wilson 95% CI ≈ 11–32%),
  minor 9/55 = 16%. QE scores do not separate the remaining KK semantic errors (wrong
  person, meaning shifts) from correct rows. **This is the main quality risk of the dataset.**
- Localisation: USD amounts < 1,000 → KZT at 500/USD; Stage 3 slot amounts ≥ 1,000 are kept as
  tenge values (first build converted them to 100+ million tenge; fixed, 38 rows re-translated).

### SFT dataset `sft-20261112-a33e9c1b` (`data/processed/sft/sft-20261112-a33e9c1b/`)

| split | examples | EN | RU | KK | scenarios | origin |
|---|---|---|---|---|---|---|
| train | 15,548 | 4,537 (29.2%) | 5,718 (36.8%) | 5,293 (34.0%) | 141 | MT 10,472 · original 4,297 · authored 779 |
| validation | 98 | 28 | 36 | 34 | 16 | authored |
| test | 116 | 33 | 39 | 44 | 16 | authored |

- Tasks in train: grounded_answer 5,979 · clarify 4,835 · tool_answer 1,241 · escalate 1,190 ·
  injection_resist 1,006 · abstain_no_evidence 994 · conflicting_context 232 · refuse_oos 71.
  Per language × task counts are in `stats.json`.
- 35 intents in train; validation 14 intents, test 12 intents (authored scenarios only).
  Code-switching: 67 train / 11 validation / 11 test examples; multi-turn: 95 train.
- The language mix is reached by **down-sampling English only** (16,875 EN Bitext examples
  dropped); RU/KK were never up-sampled. 117 duplicate examples and 38 tool rows without an
  order reference were dropped while building.
- Leakage filter (held-out vs train): exact 0, lexical (J ≥ 0.8) 0, 13-gram 0, semantic (e5
  cosine ≥ 0.95, digits masked) 63 texts → 63 held-out examples removed. Median held-out max
  cosine to train 0.93.
- Gates (all pass): schema, ids, language (user text 100%, answer 100%), licence/provenance,
  grounding (0 unsupported numbers/citations/tool facts), evidence per task, duplicates 0,
  leakage 0, placeholders 0, PII 0, kb_variant_distinct 0. Kazakh: 99.2% of KK user texts and
  100% of KK answers contain Kazakh-specific letters; 81 KK user texts contain a Russian
  function word (code-switched authored texts and loanword noise).
- Training KB `kbtrain` (`data/processed/kb_train/`): 6 stores × 8 documents × 3 languages,
  1,734 chunks (current + outdated). No value equals a demo-store value.
- Authored: 106 situations → 1,056 utterances (EN 324 / RU 353 / KK 379; 92 code-switched,
  95 multi-turn), split by scenario 70/15/15 stratified by action.

### Gold benchmark `gold-20261112-8d91fbf4` (`data/gold/gold-20261112-8d91fbf4/gold.jsonl`)
- 312 scenarios × en/ru/kk = 936 items, written natively per language, separately from the
  training scenarios. Categories (per language): grounded 98, tool 48, escalate 37, clarify 34,
  abstain 23, out_of_scope 23, multi_turn 22, injection 15, conflicting 12; difficulty easy 73 /
  medium 148 / hard 91; 72 multi-turn items, 59 code-switched items.
- Expected facts refer to the demo store (`kb_sources/demo_store/facts.yaml`); all 0 problems
  in the fact, evidence-section and language checks.
- Leakage vs all SFT splits: scenario 0, exact 0, lexical 0, 13-gram 0. Semantic near-train
  (cosine ≥ 0.95): 79 items (KK 42, RU 21, EN 16; mostly short tool requests such as "status of
  order N"), flagged in each item's review notes and reported, not removed (ADR-026).
- **Review status: all 936 items `draft`** (authored by the LLM agent, automatic checks only).
  No item is `human_reviewed` or `verified`.

### Verification evidence (Stage 4)
- `ruff check`, `ruff format --check`, `mypy --strict` clean on Stage 4 code; pre-commit hooks
  run in a clean worktree at commit time.
- `pytest`: **277 passed, 0 skipped** with the Qdrant server up (Stage 5 WIP tests excluded:
  they are untracked files of another session). New Stage 4 tests: 41 unit
  (`tests/unit/test_stage4_units.py`) + 2 integration (`tests/integration/test_stage4_pipeline.py`:
  select → KB → resumable translation with a fake MT backend → QE → SFT gates → gold, re-runs,
  forced rebuild identical, stale partial removed).
- Real data: forced SFT rebuild gives the same `dataset_version` and byte-identical
  `train.parquet`; a plain re-run is skipped. Stage 3 is unchanged: `data clean --all` skips
  with `canonical-20261111-8fe01586`, `data gates` and `data verify --all` pass.
- Interruption: two MT runs were killed by agent-session restarts; the cache kept every finished
  512-row shard and the next run resumed (1,636 KK rows reused).
- Bugs found and fixed: unquoted commas in YAML flow mappings split addresses (demo facts;
  reported by the Stage 5 session; regression test added); QE step skipped after a localisation
  change (sources hash now in its parameters); KK dates without case suffixes in tool answers;
  injection note added when only a planted passage carried the injection; KZT-scale amounts
  converted twice.

## Stage 3 data state (unchanged in Stage 4; build `canonical-20261111-8fe01586`)

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

## Stage 3 quality gates (build canonical-20261111-8fe01586)

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

## Stage 3 qualitative review (LLM agent, not an independent human review)

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

## Stage 3 verification evidence (2026-11-11)

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

## Stage 4 open items and limitations

1. **No fluent human review** of RU/KK. The MT error rates, the authored scenarios, the answer
   templates, the training KB texts and all gold items were written or judged by the LLM agent.
   Gold items are `draft`. Needed: a fluent KK and RU reviewer (open decision 2) to audit
   ≥ 100 MT rows per language and to review gold (`verified` = author + independent human).
2. **KK machine translation quality.** About 20% of accepted KK MT rows still carry a major
   error (LLM-judged, CI 11–32%). Options: a paid MT/LLM budget (open decision 1), Tilmash or
   KazParC access (HF account), human post-editing, or down-weighting `mt:` rows in Stage 7.
3. **Volume.** 15,548 / 98 / 116 instead of 90k / 5k / 5k. Nothing was duplicated. Limits: one
   GPU (MT ≈ 2.5 rows/s), 55 Bitext scenarios, authored scenarios written by one agent.
   Validation/test cover 14 / 12 intents only; use gold (936 items) for decisions.
4. **Templated targets.** Answers come from authored templates (≈ 40 section answers,
   ≈ 30 tool answers per language): correct and grounded, but low lexical variety.
5. **Bitext register.** EN originals keep typos and 10% inserted profanity; RU/KK
   translations of typo-heavy sources are literal. refuse_oos is rare in train (71).
6. Still open from earlier stages: gated sources, `uv sync --no-group gpu` not run, no LICENSE
   file (authored content has no declared licence yet), paid-API budget, reviewers,
   Kaggle/Colab accounts. Detector gaps from Stage 3 (single-script word salad, obfuscated
   e-mails, Latin-script Kazakh) are unchanged.

## Git and backups

- Remote: `origin` = https://github.com/Nusultan11/SupportPilot (public); `main` is pushed
  after each stage.
- Backups live in `backups/` at the project root (git-ignored, ADR-013).
- Code bundles (`backups/code/`): Stage 0 `…T104345Z`, Stage 1 `…T115429Z`, Stage 2
  `…T134356Z`, Stage 3 `…1T154236Z`, Stage 4 `…2T084303Z`.
- Stage 3 data snapshots (pinned): `clean_records/20261111T154052Z-5d85c3cb`,
  `canonical/20261111T154055Z-c62f57ae`, `eda_report/20261111T154057Z-9fe64e24`.
- **Stage 4 data snapshots** (pinned, deep-verified, 2026-11-12):
  - `stage4_interim/20261112T083636Z-dceb7610` (6.3 MB: backbone, MT cache, translations + QE);
  - `kb_train/20261112T083638Z-a667cddf` (training KB chunks);
  - `sft/20261112T083640Z-7ca2f2fc` (`sft-20261112-a33e9c1b`, 2.3 MB);
  - `gold/20261112T083643Z-8d81b605` (`gold-20261112-8d91fbf4`, 0.7 MB);
  - `stage4_reports/20261112T083645Z-aec9e7b1` (MT comparison samples and metrics).
- Models are not backed up: they are pinned by revision in `configs/base.yaml` and re-download
  into the HF cache (MADLAD-400-3B 11.8 GB, Opus-MT, M2M100, multilingual-e5-small).
- **Stage 4 code:** commit `da28191` on `main` (committed from a clean worktree so that the
  hooks checked only Stage 4 files); bundle `backups/code/supportpilot-20261112T084303Z.bundle`
  (verified, sha256 file next to it).

## Notes for the next stages

- Stage 5 (KB): the demo store must keep the values in `kb_sources/demo_store/facts.yaml`
  (gold expected facts and the `kb_variant_distinct` check depend on them). Gold
  `accepted_evidence` uses the doc/section ids of `authored/kb_train/docs.yaml`
  (e.g. `returns_policy.window`); Stage 5 must resolve them to demo chunk ids.
- Stage 7 (training): load `data/processed/sft/<version>/{train,validation}.parquet`;
  `messages_json` is ready for chat templates; `quality_flags` (`mt:madlad400_3b`,
  `src:bitext`, `src:authored`, `injection:*`, `conflict:*`) allow weighting. Report every
  metric per language and on gold, with the semantic-near-train subset separately.
- Re-running Stage 4: RUNBOOK §7c. MT is the only GPU step; everything else rebuilds in
  minutes from the cache.

## Next recommended action

Owner review of the Stage 4 report and decisions on open items 1–2 (reviewer, MT budget),
then continue Stage 5 (KB and document extraction).
