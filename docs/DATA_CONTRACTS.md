# DATA_CONTRACTS — schemas, identifiers, invariants

Code: `common/schemas.py`, `common/ids.py`, `common/manifest.py`, `ops/backup.py`,
`data/records.py`, `data/recipe.py`.
A breaking change needs a new `schema_version` and an ADR.

## 1. Conventions

- Languages: ISO 639-1 `en`, `ru`, `kk`. Kazakh output in Cyrillic.
  Mixed text: dominant `lang` + `code_switch: true`.
- Slugs (`source_id`, dataset names): `^[a-z0-9]+([_-][a-z0-9]+)*$`.
- Dataset versions: `{name}-{yyyymmdd}-{sha256[:8]}`.
- Timestamps: UTC ISO-8601. Money: integer minor units + ISO 4217 currency (`KZT`).
- Text: UTF-8, NFC, `\n` newlines. Originals stay untouched in raw storage.
- Manifest paths: relative POSIX, no `..` (enforced by `FileEntry`).

## 2. Source registry (`configs/sources.yaml`, `sources/1`)

Every investigated source is listed, including rejected ones with the reason.

| Field | Notes |
|---|---|
| `source_id` | unique slug |
| `url`, `revision` | canonical page; full 40-char commit SHA |
| `licence`, `licence_evidence` | licence id from the policy; where/when it was checked |
| `allowed_use` | `train` > `eval_only` > `reference_only` > `blocked`, capped by licence |
| `status` | `selected` (acquired) · `candidate` · `rejected` · `unavailable` |
| `access` | `public` or `gated` (needs `HF_TOKEN`) |
| `languages`, `size`, `reviewed_at` | |
| `purpose`, `decision`, `notes` | why it is used or rejected |
| `fetch` | `{type: hf_dataset \| github_repo, repo_id, include}`; required when `selected` |

Minimum `allowed_use` per step (`data/policy.py`): acquire, convert, analysis → `reference_only`
(acquire also needs status ≠ `rejected`); evaluation → `eval_only`; training → `train`.

## 3. Raw storage (`data/raw/{source_id}/{revision}/`)

- Original files exactly as downloaded + `manifest.json` (`raw_manifest/1`:
  `source_id, revision, url, retrieved_at, licence, repo_id, files[{path, bytes, sha256}]`).
- Each file is checked against the upstream LFS SHA256 or git blob SHA1 before acceptance.
- Never modified. A failed verification stops the pipeline; restore from backup.
- In-progress downloads: `data/raw/{source_id}/.partial-{revision}/`, resumable.

## 4. Intent taxonomy (`configs/taxonomy.yaml`, `taxonomy/1`)

Authoritative list of intents, actions, tools, escalation reasons and conditions.
`common.schemas.INTENTS` must equal it (unit test). Do not copy the table elsewhere.

## 5. Snapshot manifest (`snapshot/1`, `supportpilot backup`)

`snapshot_id` (`{yyyymmddTHHMMSSZ}-{content_sha256[:8]}`), `name`, `kind` (`data`|`model`),
`created_at`, `source_path`, `code_commit`, `note`, `pinned`, `content_sha256`,
`archive_sha256`, `archive_bytes`, `total_bytes`, `files`.

## 6. Corpus recipe (`configs/corpus.yaml`, `corpus_recipe/1`)

Code: `data/recipe.py`. Seed, split ratios, near-duplicate parameters, `min_words`, and one
entry per selected source: `kind` (`intent`|`grounded`), `active`, `domain`, `origin` per
language, `placement` (`any`|`heldout` = validation/test only), `evidence`, source-wide `veto`,
and a `labels` table. Each source label has exactly one of:

| Kind | Result |
|---|---|
| `intent: X` | `mapped_direct` |
| `rules: [{cue, intent, action?, escalation_reason?}]` | first cue found → `mapped_cue`; none → `unverified` |
| `candidate: X` | always `unverified` (the source label does not fix one v1 intent) |
| `exclude: reason` | excluded with the reason |

A `veto` pattern excludes the row first. Texts below `min_words` content words are `unverified`.
A label is an authored mapping of the source's own label, never independent adjudication.
A selected source missing from the recipe fails a unit test.

## 7. Corpus layout (`data/corpus/{version}/`)

Version: `corpus-{yyyymmdd}-{content_sha256[:8]}`. Immutable; built in `.partial-*`, renamed.
A rebuild with identical content reuses a verified version and never overwrites a damaged one.

| Path | Content |
|---|---|
| `intents/{train,validation,test}.jsonl` | `corpus_example/1`, pool `core` only |
| `intents/unverified.jsonl` | candidate labels the text does not confirm; never train/eval targets |
| `intents/excluded.jsonl` | `corpus_exclusion/1`: every source row not admitted, with reason |
| `grounded/{train,validation,test}.jsonl` | `grounded_turn/1` |
| `reports/stats.json`, `REPORT.md` | per-language counts, gaps, leakage checks |
| `reports/near_duplicate_pairs.tsv` | verified near-duplicate pairs |
| `review/sample_{lang}.csv` | blind sheets for an independent fluent reviewer |
| `manifest.json` | `corpus_manifest/1` |

`content_sha256` = SHA256 of the sorted `path\tsha256\n` lines of all files except
`manifest.json` and `REPORT.md`. Same raw inputs + recipe + taxonomy + code → same value.
Rows are sorted by id; JSON has no timestamps.

## 8. Intent example (`corpus_example/1`, `data/records.py`)

| Field | Notes |
|---|---|
| `example_id` | `ex-` + sha256(`source_id|source_item_id`)[:16] |
| `source_id`, `source_revision`, `source_item_id`, `source_split`, `source_label` | provenance |
| `text`, `text_sha256` | NFC, masked (`<name>`, `<email>`, `<phone>`, `<card>`, `<number>`, `<url>`, `<order_id>`, …) |
| `lang`, `code_switch` | source language; `code_switch` is an automatic lexicon signal |
| `origin`, `is_translation`, `parallel_id` | `human_written`·`human_roleplay`·`human_localised`·`human_spoken_asr`; `is_translation` ⇔ `human_localised` |
| `domain` | source domain (`retail_support`, `virtual_assistant`, `banking_voice`) |
| `intent`, `action`, `escalation_reason` | taxonomy v1; `escalate` ⇔ one reason |
| `label_status`, `label_rule`, `action_rule` | how the labels were derived (`mapped_direct`/`mapped_cue`/`unverified`; recipe rule id; taxonomy condition) |
| `pool` | `core` ⇔ status ≠ `unverified` |
| `group_id`, `family_id`, `split` | split unit (dialogue, parallel id, near-duplicate cluster); `split` fixed by a seeded hash of the group |
| `quality` | words, chars, `lang_guess`, `lang_consistent`, `pii_masked`, `flags`, `human_review: NOT_EVALUATED` |

Actions: taxonomy default; `needs_reference` → `call_tool` with an order reference, else
`clarify`; `product_info`/`availability` → `answer_kb`/`call_tool` only with a named product.

## 9. Invariants (enforced at build; any violation stops publication)

- Raw revisions re-verified; every source passes `check_use(training)` (inactive: analysis).
- Exact duplicates (same language + case/punctuation-insensitive text) keep one row; the same
  text with different intents is excluded as `label_conflict`.
- Near duplicates: char-4-gram MinHash/LSH, confirmed by Jaccard ≥ 0.7, merge groups.
- No group, family, parallel id, exact text or near-duplicate pair crosses splits; `heldout`
  sources never reach train; grounded document families do not cross splits; no residual PII.
- Grounded rows keep contact values printed in their own document; others are masked.
