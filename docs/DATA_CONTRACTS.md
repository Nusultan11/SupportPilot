# DATA_CONTRACTS — schemas, identifiers, versioning, validation

Schemas are implemented as Pydantic models in `supportpilot.common.schemas` and ID helpers in
`supportpilot.common.ids`. §2–4 and §9 have been implemented since Stage 1, §8 since Stage 5. Polars schemas
are added with the first Parquet pipeline (Stage 2–3). This doc is the specification. Any breaking change needs a new `schema_version`
and an entry in DECISIONS.md.

## 1. Conventions

- **Language codes**: ISO 639-1 `en`, `ru`, `kk`. Mixed text gets `lang` = the dominant language
  and `code_switch: true`, plus `langs: list[str]`.
  Kazakh is Cyrillic script (`kk-Cyrl`). Latin-script Kazakh is accepted as input, flagged
  `script: Latn`, and not generated.
- **IDs**: deterministic, no PII. All are lowercase except `source_row_key`.
  - `source_id`: slug `^[a-z0-9]+([_-][a-z0-9]+)*$`, e.g. `bitext_retail_ecom`.
  - `record_id` = `{source_id}:{source_row_key}`. The key is the source's own key or the row
    index at a pinned revision, in its original case, using characters `[A-Za-z0-9._-]`.
  - `example_id` = first 16 hex chars of `sha256(source_id|record_id|lang|origin|text_normalised)`.
  - `scenario_id`: a group of examples that express the same request and require the same
    action: the same source label and action, or a source-provided scenario id. It is joined
    across sources only through identical or near-identical texts with the same v0 intent and
    action (ADR-022). Format `scn_<16 hex>`. Splits are by `scenario_id` (ADR-026): all
    Bitext-derived scenarios are train; validation/test are authored scenarios; Stage 4 ids
    are defined in §11.
  - `family_id`: a paraphrase family inside a scenario. These are near-duplicates (char-5-gram
    Jaccard ≥ 0.8) with compatible labels, plus source alignments. All translations of an
    example share its `family_id`. Format `fam_<16 hex>`.
  - `doc_id`: slug of the KB document. `chunk_id` = `{doc_id}:{version[:8]}:{position:04d}`.
- **Timestamps**: UTC ISO-8601. **Money**: integer minor units + `currency` (ISO 4217, `KZT`).
- **Text**: UTF-8, NFC normalised, `\n` newlines. Original text is preserved in raw storage.

## 2. Source registry (`configs/sources.yaml`)

| Field | Type | Notes |
|---|---|---|
| source_id | str | unique slug |
| url | str | canonical page |
| revision | str | full 40-char commit sha for HF sources (enforced) |
| licence | str | id from the licence policy (ADR-018). `unknown` / `conflicting` force `blocked`. Unlisted ids are rejected |
| licence_evidence | str | where the licence was checked, and when |
| allowed_use | enum | `train` > `eval_only` > `reference_only` > `blocked`, capped by the licence policy |
| status | enum | `selected` (acquired), `candidate`, `rejected`, `unavailable` (e.g. gated) |
| access | enum | `public` (default) or `gated` (needs `HF_TOKEN` with accepted terms) |
| languages | list[str] | |
| size | object | rows / files / bytes, as published |
| reviewed_at | date | date of the licence/content review |
| purpose | str | what it is used for |
| decision | str | why it is selected or rejected |
| notes | str | quality, overlaps, caveats |
| fetch | object? | `{type: hf_dataset, repo_id, include: [fnmatch patterns]}`. Required when `status=selected` |

Which pipeline step may use which level is defined in `supportpilot.data.policy`:

| Step | Minimum `allowed_use` |
|---|---|
| acquire | `reference_only`, and status not `rejected` |
| convert | `reference_only` |
| analysis | `reference_only` |
| evaluation | `eval_only` |
| training | `train` |

## 3. Raw storage

`data/raw/{source_id}/{revision}/` holds the original files exactly as downloaded, plus
`manifest.json` = `{schema_version: "raw_manifest/1", source_id, revision, url, retrieved_at
(timezone-aware), licence, repo_id, files:[{path, bytes, sha256}]}`. The manifest does not list itself.

- Every file is checked against the upstream hash before it is accepted: the LFS sha256, or
  the git blob sha1 for files stored in git.
- Raw files are never modified. A revision that fails verification stops the pipeline; restore
  it from backup.
- A new upstream revision gets a new directory.
- Downloads in progress live in `data/raw/{source_id}/.partial-{revision}/` and are resumed.

## 4a. Raw record (`data/interim/records/{source_id}/{revision}/`)

This is Stage 2 output: one source row in a uniform shape, with values copied exactly
(no normalisation, no inferred labels). The directory holds:

- `records.parquet` (zstd, streamed in batches of 5000);
- `rejected.jsonl` (`{source_file, source_row, reason, snippet}`), only when rows were rejected;
- `manifest.json` (`records_manifest/1`).

| Field | Type | Notes |
|---|---|---|
| schema_version | str | `raw_record/1` |
| record_id | str | `{source_id}:{file_key}.{row}`, where row is the 0-based data row at the pinned revision |
| source_id, revision | str | lineage |
| source_file, source_row | str, int | file path inside the raw revision, row index |
| source_split | str? | split as named by the source (`train`, `eval`, `test`, ...), else null |
| record_type | enum | `single_turn`, `dialogue`, `reading_comprehension`, `policy_decision` |
| lang | str | language **declared** by the source (detection happens in Stage 3) |
| text | str | primary customer message / question, verbatim |
| response | str? | agent response if the source has one |
| source_intent, source_category, source_tags | str? | original labels, null when absent |
| extra_json | str | every other source field, JSON with sorted keys (e.g. full dialogue turns, MCQ options) |
| text_sha256 | str | sha256 of `text` (exact-duplicate key) |

The records manifest holds: `source_id`, `revision`, `allowed_use` at conversion time,
`converter` + `converter_version`, `raw_manifest_sha256` (lineage to the raw revision),
`code_commit`, `row_count`, `rejected_count`, `rejected_reasons`, `counts` (by lang / type /
split), `content_sha256` (over canonical JSON of all rows, in order) and `files`.

A conversion is skipped when the raw manifest hash, the converter version and the files all
still match. It fails when no rows are produced or when rejected rows exceed
`data.max_reject_ratio` (default 1%).

## 4b. Clean record (`data/interim/clean/{source_id}/{revision}/clean.parquet`)

This is the Stage 3 checkpoint (`clean_record/1`, manifest `clean_manifest/1`). There is one
row per raw record, so rejected rows are included with `reject_reason` (`empty_text`,
`no_letters`, `malformed_placeholder`, `unknown_placeholder`, `encoding_damage`,
`garbled_text`, `unsupported_language`).

Each row carries the cleaned and PII-masked `text` with slots filled, plus:
- `text_template` (placeholders kept) and `reference_response`;
- `slots` and `slot_signature`;
- `response_placeholders` (`{class: [names]}`);
- language fields: `lang`, `lang_declared`, `lang_detected`, `lid_conf`, `script`,
  `code_switch`, `langs`;
- `context_json`: dialogue turns, policies/facts or passage, cleaned;
- dedup keys `exact_key` and `sim_text`, and source hints `family_hint` and `scenario_hint`;
- `flags`, `fixes`, and `text_sha256` (lineage to the raw text).

The manifest records the input `records_content_sha256`, `clean_version`, `params` (with
`params_sha256`, including the placeholder catalogue hash and the lingua version), counts,
and the content hash.

## 4. Canonical example (`data/interim/canonical/examples/lang=<xx>/part-00000.parquet`)

| Field | Type | Req | Notes |
|---|---|---|---|
| schema_version | str | ✓ | `canonical/1` |
| example_id | str | ✓ | §1 |
| record_id, source_id | str | ✓ | lineage |
| scenario_id, family_id | str | ✓ | §1 |
| lang | str | ✓ | en/ru/kk |
| code_switch | bool | ✓ | |
| text | str | ✓ | customer message, normalised |
| reference_response | str | – | source response if any (style reference only, not a fact source) |
| intent | str | ✓ | SupportPilot taxonomy (§7) |
| source_intent, source_category | str | – | original labels |
| action | enum | ✓ | `answer_kb`, `call_tool`, `clarify`, `escalate`, `refuse_oos` |
| escalate | bool | ✓ | |
| escalation_reason | str? | – | required if `escalate` |
| slots | map<str,str> | – | e.g. `order_id`. Placeholders replaced with synthetic values; values are the surface strings shown in the text (money: amount as written + `currency` ISO code) |
| origin | enum | ✓ | `original`, `machine_translated`, `llm_generated`, `post_edited`, `human_written` |
| translation_of | str? | – | `example_id` of the source example |
| translation_system | str? | – | model/version |
| quality | struct | ✓ | `{lid_conf, mt_qe?, intent_agree?, dup_cluster?, flags: list[str]}` |
| split | enum? | – | assigned in Stage 4 only |

Stage 3 status: canonical rows exist only for train-allowed sources with a mapped draft-v0
label. Each such row carries the flag `labels:intent_map_draft_v0` (ADR-021).
`quality.dup_cluster` is the exact-duplicate group (`dup_<16 hex>`) when the text had duplicates.

## 4c. Pending example (`data/interim/canonical/pending.parquet`, `pending/1`)

A valid, cleaned and deduplicated example that cannot be canonical yet. It carries the same
lineage and grouping fields as canonical/1, plus `record_type`, `source_split`, `langs`,
`source_tags`, `candidate_intent`/`candidate_action` (when the label maps), `context_json`,
`allowed_use`, `reason` and `reason_detail`.

Reasons, checked in this order:
- `label_conflict`;
- `allowed_use:<level>`;
- `record_type:<type>`;
- `intent_unmapped`, `label_suspect`, `no_source_label` or `intent_not_in_map`.

## 4d. Rejected record (`data/interim/canonical/rejected.parquet`, `rejected/1`)

Fields: `record_id`, `source_id`, `revision`, `example_id`, `stage` (`clean` | `dedup` |
`validate`), `reason`, `detail`, `duplicate_of` (set exactly for `exact_duplicate`),
`family_id`, `allowed_use`, `text_sha256`.

Every input record appears exactly once in examples, pending or rejected. This is the
`accounting` gate. `groups.parquet` maps every kept row to its exact group, family and
scenario. `pairs.parquet` lists the verified near-duplicate pairs (J ≥ 0.5) with their
label compatibility.

## 5. SFT example (`data/processed/sft/{dataset_version}/`)

```json
{
  "schema_version": "sft/1",
  "example_id": "…", "scenario_id": "…", "lang": "kk", "task": "grounded_answer",
  "messages": [
    {"role": "system", "content": "<system prompt id + rendered text>"},
    {"role": "user", "content": "…"},
    {"role": "tool", "name": "get_order", "content": "{…}"},
    {"role": "assistant", "content": "<AssistantResponse JSON>"}
  ],
  "context_chunk_ids": ["returns_policy:3fa2c1d0:0002"],
  "distractor_chunk_ids": ["delivery_faq:…"],
  "target": { "AssistantResponse": "…" }
}
```

`task` ∈ `grounded_answer`, `tool_answer`, `clarify`, `escalate`, `abstain_no_evidence`,
`refuse_oos`, `conflicting_context`, `injection_resist`. Context chunks used for SFT come from a
**training KB variant** (§12), whose policy values differ from the demo KB, so the model learns
to read context instead of memorising policies.

Implemented as `common.schemas.SFTExample` (Stage 4). Message order: `system` (prompt id
`support_json_1` + text), earlier `user`/`assistant` turns, the `user` message, then `tool`
messages: `kb_search` (`{"results": [{chunk_id, title, effective_from, effective_to, text}]}`)
and/or a tool call (`{"call": {name, arguments}, "result": {...}}`), and the final `assistant`
message = `target` as JSON. Lineage fields: `source_id`, `record_id`, `origin`,
`translation_of`, `translation_system`, `kb_variant`, `system_prompt_id`, `quality_flags`.
Authored rows use `source_id = supportpilot_authored` (project-written content).

On disk: `data/processed/sft/<dataset_version>/{train,validation,test}.parquet` with columns
`messages_json` and `target_json` (JSON strings), the other fields as columns, plus
`user_text`, `history_len`, `intent`, `action`; `stats.json` (counts per split x language x
task, leakage filter), `quality_report.json` (gates) and `manifest.json`.

## 6. AssistantResponse (model output and API output)

```json
{
  "language": "ru",
  "intent": "return_policy",
  "action": "answer_kb",
  "answer": "…text shown to the customer…",
  "citations": ["returns_policy:3fa2c1d0:0002"],
  "facts_from_tools": ["order 100234 status=shipped"],
  "missing_information": [],
  "needs_human": false,
  "escalation_reason": null,
  "clarification_question": null
}
```

Validation:
- `citations` ⊆ the chunk ids provided in context.
- Each number, date or price in `answer` appears in the cited evidence or in a tool result.
- `language` = the detected user language unless the user asked for another.
- `needs_human` ⇒ `escalation_reason`.

## 7. Intent taxonomy (v1, frozen in Stage 4, ADR-023)

The authoritative v1 list, with actions, tools, KB documents and clarify / escalate / refuse
conditions, is `configs/taxonomy.yaml` (35 intents). v1 adds `change_address`, `product_defect`,
`contact_info`, `personal_data`, `newsletter`, `cart_help`, `app_help` and `technical_issue` to
the v0 table below, which remains the Stage 3 draft (`INTENTS_V0`).

### Draft v0 (Stage 3)

Derived from the Bitext retail (46 intents) and customer-support (27 intents) labels, merged
into product scope. Each source intent maps to exactly one SupportPilot intent. The mapping
table lives in `configs/intent_map.yaml`.

| Group | Intents |
|---|---|
| order | `track_order`, `cancel_order`, `change_order`, `order_history`, `place_order` |
| delivery | `delivery_time`, `delivery_options_cost`, `delivery_problem` (damaged / missing / wrong item) |
| returns | `return_policy`, `return_product`, `exchange_product`, `request_refund`, `refund_status` |
| payment | `payment_methods`, `payment_problem`, `invoice` |
| product | `product_info`, `availability` |
| store | `store_info` (location, hours, sales periods), `policy_faq` |
| account | `account_help` |
| contact | `human_agent`, `complaint`, `feedback` |
| meta | `unclear`, `out_of_scope`, `greeting` |

## 8. KB document and chunk (Stage 5, ADR-027)

Implemented as `common.schemas.KBDocument` (`kb_document/1`) and `KBChunk` (`kb_chunk/1`).

**KBDocument** fields: `doc_id`, `version` (sha256 of the source file bytes), `lang`, `title`,
`doc_type` (`policy`, `faq`, `catalog`, `terms`), `format` (`pdf`, `docx`, `xlsx`, `csv`,
`json`, `txt`), `source_path`, `effective_from/to`, `parser` (library/version + revision),
and `sections`: a tree of `KBSection {section_id, heading, level, text, tables, locator,
children}`. Level 0 is the text before the first heading; level-2 sections nest under level 1.
A record (catalogue row, CSV row) is a level-1 section whose one-row table is the record and
whose `section_id` is the slug of the row key. `KBTable {header, rows, locator}` is
rectangular. One document per (`doc_id`, `lang`): the same `doc_id` in en/ru/kk is one logical
document; chunk ids still differ because the versions differ.

**Locator** (1-based, inclusive): `page`/`page_end` (pdf), `block_range` (docx body
paragraphs and tables in order), `sheet` + `row_range` (xlsx rows), `row_range` (csv file
lines), `line_range` (txt), `json_pointer` (json, RFC 6901), `record_id` (row key). At least
one field is set.

**KBChunk** fields: `chunk_id` (§1, recomputable), `doc_id`, `version`, `lang`, `position`
(0..n-1 per document), `part` (> 0 when a long section was split), `section_id`,
`section_path` (headings as printed), `title`, `text`, `locator`, `doc_type`, `format`,
`source_path`, `effective_from/to`, `token_count` (pinned Qwen3 tokenizer, `kb.tokenizer`),
`translation_group` (`<doc_id>#<section_id>`, `#intro` for level 0: links the same content
across languages), `checksum` (sha256 of `text`).

Chunk text: the heading path on the first line (`4. Personal data > 4.1. What data we store`),
then the section text; tables as pipe tables with the header row in every chunk. A FAQ chunk
is one Q+A pair and is not split unless it exceeds `kb.max_tokens`. A record chunk is
`<title> · <sheet>` and one `header: value` line per non-empty cell (the row key included).

On disk (`data/kb/<kb>/versions/<kb_version>/`): `chunks.parquet` (key columns + `chunk_json`),
`documents.parquet` (key columns + `document_json`, `params_sha256`), `sources/` (the exact
files, `register.json`, `expected.json`), `changes.json` (`documents.{added,changed,unchanged,
removed}`, `chunks_delete`, `chunks_upsert`, `chunks_kept` vs the parent), `evidence_map.json`
(`doc_id.section_id` -> lang -> chunk ids; resolves gold `accepted_evidence`),
`quality_report.json` and `manifest.json` (`dataset_manifest/1`, `kb_version` =
`kb_<kb>-<yyyymmdd>-<hash8>`, inputs = register and expected sha256 + parent version).
`data/kb/<kb>/current.json` names the active version.

KB sources: `kb_sources/<kb>/facts.yaml` (`demo_facts/1`), `catalog.yaml` (`demo_catalog/1`),
`docs/<doc_id>.yaml` (`kb_doc/1`). Rendered: `register.json` (`kb_register/1`) and
`expected.json` (`kb_expected/1`: fact placements `{key, value, surface, section_id, where}` and
content strings per file).

## 9. Dataset version manifest

Each processed dataset or index has a `manifest.json` with these fields: `schema_version`
(`dataset_manifest/1`), `dataset_version` (`{name}-{yyyymmdd}-{hash8}`), `data_schema_version`
(schema of the rows, e.g. `sft/1`), `code_commit`, `inputs` (source_id@revision
or parent dataset_version), `params`, `row_counts` (by split × lang × task), `files`
(`{path, bytes, sha256}` each), `created_at`, and `quality_report` (path).

**Snapshot manifest** (`snapshot/1`, written by `supportpilot backup`, ADR-017):
`snapshot_id` (`{yyyymmddTHHMMSSZ}-{content_sha256[:8]}`), `name`, `kind` (`data`|`model`),
`created_at`, `source_path`, `code_commit`, `note`, `pinned`, `content_sha256` (hash of the
sorted file list), `archive_sha256`, `archive_bytes`, `total_bytes`, and `files`.

File paths in every manifest are relative POSIX paths without `..`. This is enforced by
`FileEntry`.

## 10. Validation gates (enforced in code)

They are implemented in `supportpilot.data.gates` (Stage 3) and run on every build. The
results go to `quality_report.json`, and `supportpilot data gates` re-runs them. Stage 3 also
checks:
- `accounting`: every record exactly once;
- `ids`: example ids are unique and recomputable, and each family sits in one scenario;
- `pii`: no unmasked e-mail, phone or card in the text;
- `labels`: only v0 intents, and escalation always has a reason.

Stage 4 SFT gates (`data/sftgates.py`, written to the dataset's `quality_report.json`):
`schema`, `ids`, `language` (user text LID = lang or code-switch ≥ 99%; answer LID = lang ≥ 99.9%),
`licence_provenance` (train sources or `supportpilot_authored`; MT systems with
`training_use`), `grounding` (check_response re-run from the stored messages), `evidence`
(task shape: citations, tool facts, abstain/refuse/escalate/clarify fields), `duplicates`,
`leakage`, `placeholders`, `pii` and `kb_variant_distinct`. The gold build checks schema, demo
facts, evidence sections, language and leakage against every SFT split.

| Gate | Rule |
|---|---|
| schema | 100% of rows parse. Invalid rows go to `rejected/` with a reason. Post-cleaning validity ≥ 99.5% |
| language | LID(text) = `lang` (or `code_switch`) for ≥ 99% of rows. Mismatches are flagged |
| duplicates | exact dup = 0 within a split. Near-dup clusters never cross splits |
| leakage | 0 shared `scenario_id`/`family_id` across train/validation/test/gold. No n-gram (≥ 13-gram) overlap between gold and train |
| placeholders | no unresolved `{{…}}` in SFT targets |
| licence | every row's source has `allowed_use` ∈ {train} for training sets |

## 11. Stage 4 interim outputs (`data/interim/stage4/`)

| Path | Schema | Content |
|---|---|---|
| `selected/backbone.parquet` | `stage4_backbone/1` | every Stage 3 canonical row and every `intent_unmapped` pending row of a train source, relabelled with taxonomy v1 (`intent`, `action`, `escalate`, `escalation_reason`, `rule`), v1 `scenario_id`, `selected_en` (EN backbone) and `mt_ru` / `mt_kk` (translation queue) |
| `selected/excluded.parquet` | | rows without a reliable v1 label: `excluded` (label excluded), `unlabelled` (no rule matched), `family_label_conflict` |
| `mt/<system>/<src>-<tgt>/shard-*.parquet` | cache | `{key, system, src_lang, tgt_lang, src_text, mt_text}`, key = sha256(system\|src\|tgt\|text)[:32]; append-only shards, resumable |
| `translated/translations.parquet` | `stage4_translated/1` | one row per queued translation: lineage (`example_id` of the EN source, `record_id`, `scenario_id`, `family_id`), `localized_en`, `mt_raw`, post-edited `text`, `back_translation`, `bt_chrf`, `emb_sim`, `lid_conf`, `fixes`, `hard` / `soft` QE flags, `accept` |

v1 scenario ids: a Stage 3 scenario keeps its id when all its rows map to one (intent, action);
otherwise `scn_` + sha256(old id \| intent \| action)[:16]. Translations share the family and
scenario of their source. Authored scenario ids are `scn_` + sha256(`supportpilot_authored`\|
situation id)[:16]; each authored template is one family.

## 12. Training KB chunk (`data/processed/kb_train/chunks.parquet`, `kb_chunk_train/1`)

`chunk_id` (§1 format), `doc_id`, `version` (sha256 of the rendered document), `variant`,
`status` (`current` | `outdated`), `lang`, `position`, `section_id`, `section_path`,
`doc_type`, `text`, `facts` (fact keys), `fact_values_json`, `sku` (catalogue rows),
`effective_from`, `effective_to` (set for outdated versions). Sources:
`authored/kb_train/{variants,docs}.yaml`. No variant may reuse a demo-store value of the facts
listed in `distinct_facts` (`kb_variant_distinct`).

## 13. Gold item (`data/gold/<benchmark_version>/gold.jsonl`, `gold/1`)

Fields (`common.schemas.GoldItem`): `gold_id` (`g_<scenario>_<lang>`), `scenario_id`, `lang`,
`code_switch`, `category` (grounded, tool, clarify, escalate, abstain, out_of_scope,
conflicting, injection, multi_turn, code_switch), `difficulty`, `turns` (alternating, last =
user), `intent`, `action`, `escalate`, `escalation_reason`, `required_facts` (`{key, value}`:
demo-store fact keys or tool fields), `tool_state` (mock tool results the item depends on),
`accepted_evidence` (`{doc_id, section}`; resolved to demo-KB chunk ids by the KB version's `evidence_map.json`, §8),
`must_not` (claims that make an answer wrong), `kb` (`demo_store`) and `review`
(`{status, author, reviewers, notes}`).

Review status: `draft` (authored, automatic checks passed), `llm_reviewed` (re-read by an LLM;
never "independently verified"), `human_reviewed` (one fluent human), `verified` (author +
independent fluent human; requires a named reviewer). Source: `authored/gold/*.yaml`; every
scenario is written natively in en, ru and kk.

## 14. Independent retrieval set (data expansion, ADR-029)

Sources: `authored/retrieval/*.yaml`, validated as `retrieval_source/1` by
`data/retrievalset.py`. The source header is the registry for this internal authored source:
`source_id=supportpilot_authored_retrieval`, `origin=llm_generated`, `allowed_use=eval_only`,
author, review status and licence note. Source file SHA256s enter the manifest. The external
source registry remains `configs/sources.yaml`; no fake Hub downloader is attached to authored
content. No redistribution licence for project-authored datasets is declared by this change.

Each information need has one stable ID, category (`simple`, `hard`, `ambiguous`,
`multi_fragment`, `unanswerable`), separately written `en/ru/kk` questions, explicit section
judgements `{section: doc_id.section_id, grade: 0..3}`, rationale and optional `mix_ru`,
`mix_kk` and `cross_lingual`. Grade 3 = direct support, 2 = partial/alternative support,
1 = useful background, 0 = explicitly judged non-relevant alternative. Multi-fragment
needs have at least two positive sections. Unanswerable needs have no positive judgement.

Output `retrieval_query/1`: `query_id`, `scenario_id`, `query`, `lang`, `corpus_lang`,
`code_switch`, `cross_lingual`, `category`, `answerable`, `relevant` (real `chunk_id`,
`doc_id`, `section_id`, grade), rationale, `kb_version`, source/author/review metadata and
`quality_flags`. Source selectors resolve against the active KB; a missing section fails
the build. Cross-lingual variants have a different corpus language from the question and
only target-language qrels. Variants share a scenario ID and do not increase independent
scenario or monolingual-query counts.

Storage: `data/benchmarks/retrieval/<benchmark_version>/` contains `queries.jsonl`,
standard TREC `qrels.tsv` (`query_id`, `0`, `chunk_id`, grade), `stats.json`,
`quality_report.json`, `manifest.json`. `current.json` points to the published local version
and its KB version. Version identity includes query content, source hashes, KB/SFT versions
and validation parameters. Existing versions are immutable; a failed build leaves the
current pointer intact. Rebuild after KB or SFT changes.

Gates: source/query schemas, unique IDs and monolingual questions, at least 150 monolingual
queries per language, all five categories per language, cross-lingual/code-switch presence,
language, placeholders/PII, KB references, input file hashes, exact/lexical/13-gram leakage
against every SFT split (including user history). Gold is never a query source. Semantic
near-SFT questions are flagged and reported separately, as for gold; a high embedding
similarity alone is not proof of a copied question.

SFT expansion amendment: source-native rows keep their upstream lineage and are train-only.
Authored scenarios are stratified by intent/action with trilingual scenarios reserved for
validation/test. Training near-copies of those held-out scenarios are excluded, without
lowering thresholds. Gold question text is used only to reject copied SFT inputs, never as
training examples or target text. Final gates require every intent in each split/language,
and check exact/lexical/13-gram overlap between validation and test.
