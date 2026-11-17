# Human-authored corpus review (ADR-042)

Reader: the owner and whoever runs the next controlled training/router experiment.
Decision record: [ADR-042](../DECISIONS.md#adr-042--human-authored-active-training-corpus);
schema/gates: [DATA_CONTRACTS §20](../DATA_CONTRACTS.md); commands: [RUNBOOK §7f](../RUNBOOK.md).
Raw evidence: `data/processed/human/human-20261117-3f1ed9e4/{stats,quality_report,manifest}.json`.
This is a data-engineering result. No model was trained or evaluated; no quality gain is claimed.

## 1. Sources compared (2026-11-17)

| Source | Human authorship evidence | Rights | Fit | Decision |
|---|---|---|---|---|
| ABCD (asappresearch/abcd @6b8700ce) | 2020 Expert Live Chat crowd role-play, paper §3; no generator | MIT (repo LICENSE) | EN retail chats, 55 subflows, multi-turn | **Active** EN backbone |
| MASSIVE en-US / ru-RU | crowd-written EN; RU by professional translators | CC BY 4.0 | assistant commands | **Active** for out_of_scope/greeting only; RU marked `human_localised` |
| mangi kazakh-customer-support-qa | not documented ("appear manually written/templated") | CC BY 4.0 | KK other industries | Unverified only |
| Bitext retail/customer support | hybrid-synthetic templates | CDLA-Sharing | EN intents | Excluded from active corpus (historical Stage 4 only) |
| Stage 4 MT / authored rows | MADLAD MT / LLM-agent authored | — | RU/KK/mixed | Excluded from active corpus |
| MINT-E | LLM self-play generation | dataset licence not found | multilingual e-commerce | Rejected |
| MG-ShopDial | coached human-human | no licence | 64 product-search dialogues | Rejected |
| NatCS | human spoken role-play | not verified | insurance/banking/travel | Not used |
| wb-questions (Wildberries) | real users, scraped | curator CC0 ≠ authors' rights; PII | RU product Q&A | Not acquired (unchanged) |
| KazSAnDRA | real reviews, scraped | curator CC BY ≠ authors' rights; gated | KK sentiment, not support | Not acquired |

No openly licensed, documented human RU, KK or RU–KK customer-support request corpus was found.

## 2. Measured result (`human-20261117-3f1ed9e4`)

| Measure | Value |
|---|---|
| Records read | 42,424 (ABCD 10,042 dialogues; MASSIVE 31,977; mangi 405) |
| Active train / validation / test | 3,484 / 775 / 824 = 5,083 rows; 4,743 independent family/scenario groups |
| Languages | EN 4,356; RU 727 (all human-localised OOS/greeting); KK 0; RU–KK mixed 0 |
| Origins | human_roleplay 3,595; human_written 761; human_localised 727 |
| Intents covered | 20 of 35; validation and test contain all 20 |
| Actions | clarify 1,547; refuse_oos 1,480; answer_kb 1,469; escalate 546; call_tool 41 |
| Rows with earlier dialogue context | 3,328 |
| Unverified | 1,357: too_short 1,163, family_label_conflict 147, template_cluster 22, language_mismatch 15, prompt_echo 10 |
| Excluded | 35,984 (84.8% of input): label_cap 19,957 (MASSIVE balance), no v1 intent 12,046, cross_split_semantic 3,037, historical overlap 820, exact duplicate 121, unlabelled 3 |
| Historical decontamination | 32,413 unique earlier user texts from 45 files; removed exact 359, lexical 5, e5 semantic 456; after removal exact/lexical/13-gram overlap = 0 |
| SFT targets | none (`no_human_target` for every row) |

Missing everywhere: app_help, cancel_order, cart_help, contact_info, delivery_time, feedback,
human_agent, invoice, newsletter, order_history, payment_methods, personal_data, place_order,
product_defect, unclear. Thin: exchange_product 3, greeting 8, change_address 14.

Previous active corpus `sft-20261112-b0388c64`: 17,456 rows (machine_translated 10,273,
original-but-templated Bitext 4,774, llm_generated 2,409); only 4.3% came from MASSIVE.

## 3. Label audit (agent review, not human verification)

Two random samples were read by the implementing agent against the taxonomy definitions.
Sample 1 (60 ABCD rows, earlier build) found 2 clear errors; both rules were corrected.
Sample 2 (80 rows, build `a4d18e9a`, different seed) found 4 errors (5.0%, Wilson 95% about
2–12%): "pending refund" and "ordered boots don't fit" (rule defects, corrected) plus a cancel
request inside a shipping-cost dialogue and a vague "question about my account". The last two are
single text/source-label conflicts left in place. The post-correction rate is not measured;
treat roughly 5% residual label noise as plausible until independent review exists.

Observed source traits: workers paraphrase the same scenario prompt, so near-paraphrases are
frequent (cross-split semantic removal 3,037 train rows at the project e5 cosine 0.95); prompt
wording without second-person markers cannot be detected reliably; masked tokens (`<name>`,
`<order_id>`) replace the fictional customer data; ABCD agent turns state AcmeBrands policies
and are kept only as dialogue context, never as facts or targets.

## 4. Attempts and limitations

Development builds `human-20261117-42e33568` and `human-20261117-a4d18e9a` are retained,
superseded (no `langs` field / before two rule fixes). Build `3f1ed9e4` was rebuilt once to
make manifest history paths relative; data files were byte-identical.

The corpus supports intent/action routing, clarification/escalation decisions and dialogue
context in EN, and out-of-scope refusal in EN/RU. It does not support grounded generation,
KK or mixed RU–KK, or 15 intents. Completion of the original goal therefore remains
PARTIAL: closing it needs licensed human RU/KK/mixed requests, independent fluent label review
and human-written answers grounded in this store's KB/tools (owner decisions on any paid,
gated or external resource).
