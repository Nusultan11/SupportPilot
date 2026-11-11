# PROGRESS: current state (single source of truth for status)

Last updated: 2026-11-11

## Current stage

**Stage 2: Data acquisition and source registry. Status: COMPLETED.** Waiting for owner
confirmation to start Stage 3 (EDA and data quality engineering).

| Stage | Status | Commits |
|---|---|---|
| 0 Audit and research | COMPLETED | `699dc23`, `cd50a6c`, `ced287e` |
| 1 Engineering foundation | COMPLETED | `6b17404`, `7b6bc4c` |
| 2 Data acquisition and source registry | COMPLETED | Stage 2 commit on `main` (see `git log`; hash in the Stage 2 report) |

## What exists (cumulative)

| Component | Path | Notes |
|---|---|---|
| Foundation (Stage 1) | `src/supportpilot/common/`, `ops/`, `cli.py`, `configs/base.yaml`, `compose.yaml` | config, logging/redaction, IDs, hashing, manifests, contracts, MLflow, backups, checks |
| Source registry | `configs/sources.yaml` | 16 sources reviewed; licence policy enforced by the schema (ADR-018) |
| Usage policy | `data/policy.py` | acquire/convert/analysis ≥ reference_only, evaluation ≥ eval_only, training = train |
| Acquisition | `data/acquire.py` | pinned HF commits, upstream hash check, resumable `.partial-<rev>`, immutable raw (ADR-019) |
| Converters | `data/converters.py`, `data/convert.py` | 4 converters → `RawRecord` Parquet + `rejected.jsonl` + `RecordsManifest`; idempotent, atomic |
| Loader | `data/records.py` | `scan_records(registry, data_dir, ids, purpose)`, the only sanctioned way to load records |
| CLI | `supportpilot data sources/acquire/convert/verify` | RUNBOOK §7 |

## Source inventory (acquired 2026-11-11)

| source_id | revision | allowed_use | licence | raw files / bytes | records | rejected | records content_sha256 |
|---|---|---|---|---|---|---|---|
| bitext_retail_ecom | `12dd624d…` | train | CDLA-Sharing-1.0 | 2 / 42,634,113 | 44,884 | 0 | `1ec560bf027f7385…` |
| bitext_customer_support | `430d1a89…` | train | CDLA-Sharing-1.0 | 2 / 19,214,423 | 26,872 | 0 | `92d20a0f123649e8…` |
| whileai_ecommerce_intent | `d7639462…` | train | CC-BY-4.0 | 3 / 77,603,737 | 24,710 (train 22,733 + eval 1,977) | 0 | `0330b803efb5a3c5…` |
| support_json_ru | `280ce363…` | reference_only | CC-BY-4.0 | 6 / 4,275,284 | 716 (train 512 / val 96 / test 108) | 0 | `0a3b7b0a0f300c07…` |
| belebele | `7899cdfa…` | eval_only | CC-BY-SA-4.0 | 4 / 7,803,342 | 2,700 (900 each kk/ru/en) | 0 | `c3bd7288dadf519a…` |
| **total** | | | | **17 / 151,530,899** | **99,882** | **0** | |

- Training-eligible records (loader with `Purpose.TRAINING`): 96,466, all **EN**.
- **RU/KK training data does not exist yet.** It is the main job of Stage 4 (ADR-006).
- Not acquired:
  - gated (need an HF account): `kazparc`, `flores_plus`, `kazqad`;
  - candidates: `kz_multidomain` (24.7 GB), `wb_questions` (real-user content, legal basis unclear);
  - rejected (6, reasons in the registry): `kazakh_instruction_v2` (licence conflict),
    `powerful_kazakh_dialogue`, `lakshan_banking_support`, `synthdata_ecom_dialogues`,
    `runlu_intent`, `support_tickets_multilang`.

First observations for Stage 3 (counts only; EDA not done yet):
- **Exact duplicate texts:**
  - Bitext customer support: 24,635 unique of 26,872 (8.3% duplicates);
  - while-ai first user turns: 23,814 unique of 24,710;
  - support_json_ru messages: 353 unique of 716.
  - Bitext retail: 44,884 unique.
- **Labels in the data:** Bitext customer support has 27 intents and 11 categories (its card
  lists fewer); Bitext retail has 46 and 13.
- **Missing labels are null, not invented:** `source_intent` is null exactly for Belebele and
  support_json_ru (3,416 rows). `response` is null for Belebele and for 2 while-ai dialogues
  with no assistant reply.
- **Templating:** Bitext texts and responses contain `{{Placeholder}}` slots. while-ai has 3
  conversations that start with the assistant; `text` is the first user turn.

## Verification evidence (2026-11-11)

- **Tests:** 173 passed, 0 skipped (`uv run pytest -q`, with the Qdrant server up). New in
  Stage 2: 65 tests across `test_registry_policy`, `test_acquire`, `test_convert` and
  `test_cli_data`. They cover:
  - registry and licence-policy validation; the full allowed_use × purpose matrix;
  - upstream hash checks (LFS sha256, git blob sha1); retries; corrupted downloads;
  - resume after an interruption, with completed files kept; immutability of finished revisions;
  - exact value copying, rejection logging and the reject-ratio gate;
  - converter-version rebuilds; a crash mid-conversion that keeps the previous output;
  - tampered raw data blocking conversion; the policy-checked loader; CLI exit codes.
- **Lint and types:** ruff, ruff format, `mypy --strict` (40 files) and `pre-commit run --all-files` are clean.
- **Real acquisition:** all 5 selected sources matched the upstream hashes.
  - A re-run of `data acquire --all` skipped all 5 without network.
  - `data verify --all`: raw and records OK.
- **Real interruption:** `kill -9` 6 s into the while-ai download. The re-run kept the verified
  `README.md` and `eval.jsonl`, finished `train.jsonl`, and the result was identical to the
  first acquisition (`diff -r`). This ran in a fresh scratch data dir, rebuilt from the registry alone.
- **Determinism:** `data convert --all --force` reproduced identical `content_sha256` and
  identical Parquet file sha256 for all 5 sources.
- **Policy:** loading `belebele` for training and `support_json_ru` for evaluation or training
  raised `PolicyError`.
- **Backups:** pinned snapshots of each raw source plus a records snapshot, all deep-verified.
  A restore of `whileai_ecommerce_intent_raw` passed `RawManifest` verification and matched
  live data, manifest included.

## Known limitations and open items

1. Gated sources are untested: the code path exists but there is no HF account. FLORES+ is
   needed for Stage 4 MT evaluation, so the owner's HF account is the blocker for that step.
2. All training-eligible data is English. KK/RU must be created in Stage 4 (MT budget and
   reviewers are still open decisions).
3. "Clean machine" reproducibility was shown with a fresh data directory on this machine, not
   on another computer.
4. Still open from Stage 1:
   - `uv sync --no-group gpu` has not been run;
   - the public repo has no LICENSE file. Note CDLA-Sharing obligations if derived data is ever published;
   - RAM is tight with Docker running.
5. Still open from Stage 0: paid-API budget, fluent KK/RU reviewers, Kaggle/Colab accounts.

## Git and backups

- Remote: `origin` = https://github.com/Nusultan11/SupportPilot (public); `main` is pushed after each stage.
- Code bundles in `C:\Users\nurs\SupportPilot-backups\code\`:
  - Stage 0: `supportpilot-20261111T104345Z.bundle`;
  - Stage 1: `supportpilot-20261111T115429Z.bundle`;
  - Stage 2: the bundle created after the Stage 2 commit (named in the report).
- Data snapshots in `C:\Users\nurs\SupportPilot-backups\data\`. The raw ones are pinned:
  - `bitext_retail_ecom_raw/20261111T133457Z-7b3a9867`
  - `bitext_customer_support_raw/20261111T133459Z-890f7689`
  - `whileai_ecommerce_intent_raw/20261111T133500Z-50f1b129`
  - `support_json_ru_raw/20261111T133501Z-cc4f6661`
  - `belebele_raw/20261111T133502Z-d1f467f9`
  - `raw_records/20261111T134036Z-ee0f165b` (latest, not pinned; reproducible from raw. The earlier `...133503Z-76b840f7` has the same records and an older manifest timestamp)
  - `env_reports/20261111T113155Z-92ddf171` (Stage 1)

## Notes for the next stage (Stage 3)

- Load data only through `scan_records(..., Purpose.ANALYSIS)`, and use `Purpose.TRAINING`
  when building training sets.
- Start EDA from `data/interim/records/*/*/records.parquet`. `extra_json` holds the full
  dialogues (while-ai) and the policies/facts (support_json_ru).
- Write EDA reports to `artifacts/reports/eda/`, not to notebooks.
- Planned checks: language ID on `text` (all sources declare their language, nothing has been
  detected yet), near-duplicates and paraphrase families (Bitext), placeholder inventory,
  register tags, and token lengths with the Qwen tokenizer (adds `transformers`/`tokenizers`,
  record it in ADR-015).

## Next recommended action

Start Stage 3 (ROADMAP §Stage 3) after owner confirmation.
