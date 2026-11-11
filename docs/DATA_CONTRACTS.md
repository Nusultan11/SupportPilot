# DATA_CONTRACTS — schemas, identifiers, versioning, validation

Schemas are implemented as Pydantic models and Polars schemas in `supportpilot.common.schemas`
(from Stage 1). This doc is the specification. Any breaking change needs a new `schema_version`
and an entry in DECISIONS.md.

## 1. Conventions

- **Language codes**: ISO 639-1 `en`, `ru`, `kk`. Mixed text gets `lang` = the dominant language
  and `code_switch: true`, plus `langs: list[str]`.
  Kazakh is Cyrillic script (`kk-Cyrl`). Latin-script Kazakh is accepted as input, flagged
  `script: Latn`, and not generated.
- **IDs**: lowercase, deterministic, no PII.
  - `source_id`: slug, e.g. `bitext_retail_ecom`.
  - `record_id` = `{source_id}:{source_row_key}` (the source's own key or row index at a pinned revision).
  - `example_id` = first 16 hex chars of `sha256(source_id|record_id|lang|origin|text_normalised)`.
  - `scenario_id`: a group of examples that express the same request and response logic. Splits
    are by `scenario_id`.
  - `family_id`: a paraphrase family inside a scenario. All translations of an example share its `family_id`.
  - `doc_id`: slug of the KB document. `chunk_id` = `{doc_id}:{version[:8]}:{position:04d}`.
- **Timestamps**: UTC ISO-8601. **Money**: integer minor units + `currency` (ISO 4217, `KZT`).
- **Text**: UTF-8, NFC normalised, `\n` newlines. Original text is preserved in raw storage.

## 2. Source registry (`configs/sources.yaml`)

| Field | Type | Notes |
|---|---|---|
| source_id | str | unique slug |
| url | str | canonical page |
| revision | str | HF commit sha / release tag / file sha256 |
| licence | str | SPDX id or exact text id; `unknown` blocks use |
| allowed_use | enum | `train`, `eval_only`, `reference_only`, `blocked` |
| languages | list[str] | |
| size | object | rows / files / bytes, as published |
| retrieved_at | datetime | |
| purpose | str | what it is used for |
| notes | str | quality, overlaps, caveats |

## 3. Raw storage

`data/raw/{source_id}/{revision}/` holds the original files exactly as downloaded, plus
`manifest.json` = `{source_id, revision, url, retrieved_at, files:[{path, bytes, sha256}]}`.
Raw files are never modified. A re-download with a different hash creates a new revision directory.

## 4. Canonical example (`data/interim/`, Parquet, partitioned by `lang`)

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
| slots | map<str,str> | – | e.g. `order_id`. Placeholders replaced with synthetic values |
| origin | enum | ✓ | `original`, `machine_translated`, `llm_generated`, `post_edited`, `human_written` |
| translation_of | str? | – | `example_id` of the source example |
| translation_system | str? | – | model/version |
| quality | struct | ✓ | `{lid_conf, mt_qe?, intent_agree?, dup_cluster?, flags: list[str]}` |
| split | enum? | – | assigned in Stage 4 only |

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
**training KB variant**, whose policy values differ from the demo KB, so the model learns to
read context instead of memorising policies.

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

## 7. Intent taxonomy (draft v0, frozen in Stage 4)

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

## 8. KB document and chunk

**KBDocument** fields: `doc_id`, `version` (sha256 of source bytes), `title`, `lang`,
`doc_type` (`policy`, `faq`, `catalog`, `terms`), `format`, `source_path`,
`effective_from/to`, and `sections` (tree of `{heading, level, text, tables}`).

**KBChunk** fields: `chunk_id`, `doc_id`, `version`, `lang`, `position`, `section_path`
(list of headings), `text`, `locator` (`{page?, sheet?, row_range?, record_id?}`), `doc_type`,
`effective_from/to`, `token_count`, `translation_group` (links the same content across
languages), `checksum`.

Tables keep their header row in each chunk. XLSX chunks keep `sheet`, column headers and the
row key. A FAQ chunk is one Q+A pair. Small FAQ items are not split further.

## 9. Dataset version manifest

Each processed dataset or index has a `manifest.json` with these fields: `dataset_version`
(`{name}-{yyyymmdd}-{hash8}`), `schema_version`, `code_commit`, `inputs` (source_id@revision
or parent dataset_version), `params`, `row_counts` (by split × lang × task), `sha256` per file,
`created_at`, and `quality_report` (path).

## 10. Validation gates (enforced in code)

| Gate | Rule |
|---|---|
| schema | 100% of rows parse. Invalid rows go to `rejected/` with a reason. Post-cleaning validity ≥ 99.5% |
| language | LID(text) = `lang` (or `code_switch`) for ≥ 99% of rows. Mismatches are flagged |
| duplicates | exact dup = 0 within a split. Near-dup clusters never cross splits |
| leakage | 0 shared `scenario_id`/`family_id` across train/validation/test/gold. No n-gram (≥ 13-gram) overlap between gold and train |
| placeholders | no unresolved `{{…}}` in SFT targets |
| licence | every row's source has `allowed_use` ∈ {train} for training sets |
