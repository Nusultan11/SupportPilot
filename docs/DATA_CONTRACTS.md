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

## 15. Retrieval index, results and context (Stage 6)

`configs/retrieval_models.yaml` is the model registry, separate from the dataset source
registry: pinned Hub commit, licence/card evidence, pooling, query/document protocol,
dimensions and bounded inference settings. `retrieval.models.ModelSpec` rejects unknown
licences and unpinned revisions. No remote model code is executed.

`retrieval.models.Recipe` identifies the encoder, sparse features, token window/overlap,
length normalisation and tokenizer checksum. Its content hash owns one Qdrant collection
and `data/indexes/retrieval/<recipe_hash>/`. Immutable `versions/rindex-<date>-<hash>/`
contains a `DatasetManifest` (`retrieval_index/1`), `recipe.json`, `units.jsonl`,
`vectors.npy` and canonical `chunks.jsonl`. The manifest links the exact KB/model versions
and hashes every file. Vectors use float32 storage and cosine distance; model inference
uses float16 on CUDA, float32 on CPU. SQLite embedding caches are keyed by full encoder
configuration, text hash and query/document role, with a checksum for each vector blob.

Search-unit IDs hash the recipe, source chunk IDs, span and search text. Canonical,
token-window and adjacent-section views retain their original chunk IDs; scores project
back to unique canonical chunks, each consuming a ranking slot. A search view does not
create new evidence, mutate KB chunks or relabel qrels. Unit payloads carry language,
document/type, canonical members and inclusive effective-date ordinals.

`state.json` names the materialised collection/index/KB version and readiness. Synchronising
sets readiness false before any delete/upsert. A matching-parent transition validates
`changes.json`; rollback/non-parent transitions use an exact inventory diff. Unchanged
vectors are retained, deleted IDs are removed from the live collection, and cached old
snapshots allow rollback. Qdrant may retain internal segment tombstones until compaction.
The ready pointer is published only after exact point-inventory verification. Readers
refuse incomplete/stale versions, including a KB activation not yet synchronised. This
is fail-closed synchronisation with brief unavailability, not zero-downtime activation.

`retrieve(query, lang, filters, k)` is exposed by `retrieval.Retriever`; use
`with retrieval.open_retriever(settings)` to reuse models and release resources.
`SearchFilters` supports `corpus_lang`, `doc_ids`, `doc_types`, inclusive `as_of` date.
Omitting corpus language searches all languages; it is independent of the query language.
`SearchOptions` separates candidate count, rerank count and output count, fusion and
optional dense/reranker confidence thresholds. `lang` is a convenience corpus-language
filter; use `SearchFilters` to make the corpus language explicit for cross-language input.
Sparse IDF statistics are scoped to the live `kb_version` via `SearchParams.idf.corpus`
(Qdrant 1.20); global segment statistics were not stable after delete/restore.
IO retries are bounded to three attempts.

`evidence_threshold` gates the entire query by its maximum reranker score, preserving
weak supporting chunks when the query passes. It requires reranker scores and yields
`low_evidence_confidence` on rejection. It is disabled in the selected configuration:
calibration could not improve abstention while preserving the multilingual recall guard.
`min_rerank_score` instead prunes individual chunks; these are distinct operations.

`ScoredChunk` contains `chunk_id`, `kb_version`, `index_version`, the original `KBChunk`
(including locator/source path and validity dates), dense/sparse/fused/rerank scores and
matching unit IDs. `RetrievalContext` (`retrieval_context/1`) contains complete selected
evidence, a JSON-data text representation, exact token count/budget, truncation and empty
evidence/abstention reason. Metadata counts toward the budget; chunks are never cut inside
a fact/table. Insufficient budget and empty filtered results produce no evidence.
Retrieved source text remains untrusted data. Similarity is relevance, not a proof that
the KB contains a requested live status or missing policy exception; downstream grounding
and routing must still check answerability.

Evaluation preserves `retrieval_query/1` and its draft judgements. Fixed evaluation date,
seed, scenario-level development/holdout assignment, query-level observations, grouped
bootstrap CIs, paired comparisons and resource measurements live under
`artifacts/reports/retrieval/` and in MLflow. Gold, SFT and their schemas are unchanged.

## 16. Large-scale retrieval inputs and validity (Stage 6.1)

`technical_profiles/1` in `kb_sources/demo_store/technical_profiles.yaml` is an internal
LLM-authored source registry: author, origin, review status and licence note accompany
explicit product rows, native names, field units/labels, related families and a separate
stable `sku_registry`. Every row supplies one value per declared field; duplicate IDs/SKUs,
unknown fields, nonfinite numbers and inverted operating/storage intervals are rejected.
The adapter emits separate `technical_<product_id>` documents and canonical
`technical.<product_id>.<attribute>` facts. Identifiers, translated names, prices and
composite registry entries are counted separately from atomic numeric attributes.
No real manufacturer data or live stock/order/payment state is asserted.

Optional document-spec `effective_from`, `effective_to` and `superseded_by` preserve the
existing default start date. A superseded source needs an explicit end; its inclusive end
must precede its successor's start. Source registers and expected documents carry the
optional end, omitted when null for compatibility. Metadata-only validity edits invalidate
incremental reuse; canonical chunk IDs can remain stable while payload dates change.
Default `SearchFilters` still enforce validity. Explicit `all_versions=True` supports a
historical comparison, retaining both dates in context metadata; it does not choose which
historical condition a downstream answer should apply.

`retrieval_scale_source/1` extends the Stage 4 retrieval source with unique `need_key`,
`family_id`, tags, optional `as_of`, `all_versions`, negative reason and required
`{fact, sections}` groups. Its header is the registry for these internal eval-only authored
inputs; the adapter retains `source_id=supportpilot_authored_retrieval`. Questions resolve
to unchanged `retrieval_query/1`, with proof/split metadata in a separate sidecar. Each
required fact must have an actual canonical placement in an in-date target-language chunk
(unless all versions were explicitly requested). Translation/paraphrase variants share
one scenario; related predicates/entities additionally share a family for splitting.

`data/benchmarks/retrieval_scale/<input_hash>/` contains `queries.jsonl`, `qrels.tsv`,
`metadata.json`, `protocol.json`, `audit.json`, `freeze.json`. The last file hashes every
other file, records identity/time, and is published atomically only after gates and semantic
leakage diagnostics. Execution verifies source/code/tokenizer fingerprints and qrel alignment.
Use LF for source files per `.gitattributes` before freezing. Each namespace owns separate
index state/collections without changing default Stage 6 identities. Candidate checkpoints
are atomic and bind the index manifest, query/filter set, limit and execution code; altered
checksums or mismatched execution inputs fail closed. Cached searches are excluded from
new search latency measurements.

`retrieval_load_corpus/1` uses immutable checksummed synthetic text and memmapped float32
vectors with explicit generator configuration. Its IDs/collections never enter the semantic
KB or qrels. ANN fidelity, random-vector upload and fixed top-20 neural work are separate
measurement scopes; none establishes semantic quality or 100k neural encoding throughput.

Supplemental retrieval-diagnostic freezes bind source/query/metadata hashes and the primary
selection. New variant IDs retain original development need IDs and proof groups; they
never increase independent-need counts or enter selection. Authorship is llm_generated.

## 17. LLM experiments, selection and routing (Stage 7)

`ModelSpec` (`llm/models.py`) pins a full 40-hex model revision and its licence;
`InferenceConfig` binds precision, quantization, token/time limits, seed, adapter and
constrained-decoding choice. `TrainingConfig` adds rank/alpha, target modules, optimizer
learning rate, sequence length, packing, step budget, checkpoints and MT down-weighting.
The training `binding.json` hashes the complete inputs, configuration and runtime source/
library identity. Resume requires the same binding. Overlength targets are excluded with
reasons, never truncated; prepared supervised JSON and terminal EOS are checked before
training. Checkpoints, step receipts and failed attempts remain distinct artifacts.
Prompt mode and frozen train-only demonstrations are separate from model-loading
configuration. The generation boundary accepts them explicitly; schema/few-shot modes
share `prepare_messages` with training/evaluation, while omitted mode preserves legacy
serialization. A few-shot call without demonstrations fails before model generation.

`llm_adapter/1` manifests (`adapter_manifest.json`) bind the exact base model/revision,
training identity and relative file hashes. Loading checks all adapter bytes. A selection
seal additionally hashes the adapter manifest, so replacing weights and refreshing their
manifest cannot silently change a selected adapter.
`export_inference_adapter` derives a separately hashed inference bundle from a verified
training bundle. Safetensors/config/tokenizer bytes are preserved; the card is copied
or explicitly supplied before sealing. The opaque
`training_args.bin` remains only in the source archive, and origin-manifest/excluded-file
hashes are recorded. Exports cannot be nested in the immutable source or overwritten
with changed provenance/card inputs. Model cards used inside a sealed bundle are fixed;
subsequent held-out results belong in external evaluation reports, not mutated weights
or adapter metadata.

Inference `predictions.jsonl` retains stable example/scenario/family/language identities,
raw completion, execution error, deterministic diagnostics and measured observations.
Raw JSON/schema/intent/action scores, accepted router decisions and safe fallbacks have
different scopes. Semantic support, grounded-answer accuracy, citation precision and
unsupported-claim rate are not inferred from citation membership or digit matching.
Missing observations remain null/UNKNOWN or NOT_EVALUATED with denominators. Summaries
retain per-language and mixed slices; paired CIs require identical unique example IDs
and matching scenario/family/language/intent identities.

`llm_selection/1` seals bind finalist inference configurations, prompt modes, decision,
validation-report hashes and adapter-manifest hashes before test/gold are opened.
`llm_heldout/1` exports bind the seal identity, input bytes, gold manifest and split counts.
Gold retrieval/packing observations and expected labels stay separate from model inputs;
the terminal assistant label is removed before generation. Synthetic gold tool fixtures
are explicitly conditional inputs, not observed live API state.
Gold `accepted_evidence` lists allowed citation sections, not an AND-proof specification.
Their ranking/packed reference coverage is diagnostic; gold sufficiency stays null.
Only separately declared `required_proof_groups` can label known packed-support loss.
Abstention proxies record their scope (`draft_task_label` or known missing support),
retain malformed completions as failures and never establish semantic correctness.

`route_decision/1` includes language/code-switch, one of the 35 intents, action, optional
confidence, slots, escalation flag and reason. Extra fields are forbidden; an escalation
flag requires a reason and an escalate action requires the flag. The encoder and LLM
implementations share this type; confidence availability does not establish calibration.

`authored/safety/scenarios.yaml` is the internal source registry for
`supportpilot_authored_safety`: origin `llm_generated`, allowed use `eval_only`, draft review,
author and licence note. Its parallel language texts define at least 50 unique needs per
language. `llm_safety/1` freezes source/data hashes and language/category counts. No external
downloader or redistribution licence is attached to this project-authored source.
`llm_sufficiency/1` binds exposed development observations, required evidence groups and
KB hashes; interventions never turn related variants into new independent needs.
`authored/safety/context_conflicts.yaml` separately registers one synthetic policy need
with parallel translations. Its complete/conflict/absent interventions retain one
scenario/family and nine rows; the two equally current passages have incompatible
values. They supplement, rather than overwrite, the frozen150-row safety cohort.

Private Kaggle `submission.json` binds the explicit source/config/data allowlist and ZIP
hash. Reusing a destination with changed bytes is rejected; identical replay preserves
the ZIP. Verification requires terminal status for an explicit kernel version, matching
source/payload manifests and every output checksum before import. Remote execution
Python/platform and payload identity are recorded separately from the local MLflow
importer environment/commit. A missing remote Git commit stays UNKNOWN; import-time
Git metadata is not execution provenance.
