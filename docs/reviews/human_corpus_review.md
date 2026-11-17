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

## 5. Extension (ADR-043, 2026-11-17)

Sections 1–4 record the original ADR-042 delivery and stay unchanged. Active corpus is now
`human-20261117-fedfa10a`; raw evidence in its `{stats,quality_report,manifest}.json`.

### 5.1 Sources examined in this round

| Source | Human authorship evidence | Rights | Decision |
|---|---|---|---|
| MINDS-14 ru-RU (PolyAI @40ce77cb) | native speakers spoke new requests (Prolific); Google-ASR 1-best text | CC BY 4.0 | **Active**, origin `human_spoken_asr`; bank-only intents → out_of_scope, app/address → technical_issue/change_address; card/bill/debit/history excluded |
| xSID 0.7 kk dev/test (mainlp/xsid @761d926c) | native-speaker translations of Snips/Facebook (paper §2.2); MT `projectedTrain` not fetched | CC BY-SA 4.0 | **Active, held-out only** (authors: evaluation data), out_of_scope |
| CLINC150 plus (clinc/clinc_oos @155b9c71) | crowd-written queries (Larson et al. 2019) | CC BY 3.0 | **Active**: greeting/thanks, card-declined, order-status, out-of-scope domains; ambiguous banking/meta labels excluded |
| doc2dial v1.0.1 (shared-task repo @7584ac0c) | ~70 appen.com contributors wrote all utterances from span-annotated scenes | CC BY 3.0 (IBM card; host repo has no licence file) | **Active grounded component**, namespace `doc2dial_external` |
| mangi kk support QA | card now says "hand-curated"; authorship of the questions still undocumented | CC BY 4.0 | Unchanged: unverified only |
| KRCS kk–ru code-switching | social-media posts + bilingual normalisations | curator Apache-2.0 ≠ authors' rights | Not acquired (style reference only) |
| TilQazyna til-kk-intent | LLM-teacher synthetic | Apache-2.0, gated | Rejected |
| Denn231 grocery complaints | apparently real customer messages with order numbers | none | Rejected (rights, personal data) |
| sfbalaba ru messages | undocumented | none | Rejected |
| Russian MultiWOZ | translation method undocumented; hotel/travel domains | MIT | Rejected |
| FutureBeeAI RU retail chats | 150 native speakers (vendor claim) | commercial licence only | Not contacted: paid/external resource needs owner approval |

No openly licensed corpus of natural Kazakh or mixed RU–KK store requests was found. The KK rows
are translations of assistant commands; the natural RU rows are spoken e-banking requests.

### 5.2 Measured result (`human-20261117-fedfa10a`)

| Measure | `3f1ed9e4` | `fedfa10a` |
|---|---|---|
| Active intent rows (train / validation / test) | 5,083 (3,484 / 775 / 824) | 6,512 (4,197 / 1,133 / 1,182) |
| Independent family/scenario groups | 4,743 | 6,173 |
| EN / RU / KK / RU–KK mixed | 4,356 / 727 / 0 / 0 | 5,103 / 896 / 513 (validation 256, test 257) / 0 |
| Origins | roleplay 3,595; written 761; localised 727 | roleplay 3,584; written 1,519; localised 1,242; spoken_asr 167 |
| Intents covered (all; RU; KK) | 20; 2; 0 | 20; 5 (out_of_scope 866, technical_issue 14, change_address 13, greeting 2, account_help 1); 1 (out_of_scope) |
| Unverified / excluded | 1,357 / 35,984 | 1,515 / 59,586 (label_cap 35,357; no v1 meaning 19,922) |
| Historical overlap removed | exact 359, lexical 5, e5 456 | exact 365, lexical 5, e5 506 |
| Grounded human responses | none | 17,710 EN (answer 14,292; clarify 3,418; 451 documents; train/val/test 14,023 / 1,887 / 1,800) |

Grounded component: 26,254 agent turns read; 1,851 without a preceding user turn and 125
historical overlaps excluded; unverified 6,568 (weak evidence overlap 5,723, contact pattern 833,
placeholder/corrupted 12). Evidence-recall quantiles of trusted rows: p10 0.571, p50 0.846.

Still missing everywhere (15): app_help, cancel_order, cart_help, contact_info, delivery_time,
feedback, human_agent, invoice, newsletter, order_history, payment_methods, personal_data,
place_order, product_defect, unclear. No open human source with these store intents in RU/KK
was found; none was synthesised. KK is evaluation-only out-of-scope; mixed RU–KK remains 0.

### 5.3 Audits (agent review, not human verification)

- Intent rows, sample A (80 new-source rows, seed 20261117): 2 disputable labels (a "joint
  account" and an "existing credit" request that need not be bank-only); rules now require bank
  cues. Sample B (60 rows, seed 2, after the fix): 1 disputable ("pay for things in America"
  may be a payment-method question); the `abroad` rule now requires a card/bank cue. Residual
  noise of the new sources after these fixes is not measured; ABCD's ~5% estimate (§3) stands.
- Spoken RU rows keep ASR errors (e.g. misrecognised words); they are natural phrasing, not
  clean text. Two self-introduced names and English-language transcripts went to unverified.
- Grounded, sample C (30 rows, threshold 0.3): about 2 of 24 answers added details absent from
  the cited spans and 3 of 6 clarifications were flow-driven rather than responsive; the
  threshold was raised to 0.5 for all behaviours. Sample D (24 rows, threshold 0.5): one
  placeholder query ("Null", now filtered) and one answer with an unsupported "Yes" lead-in.
  Treat roughly 5–10% of grounded targets as containing unsupported or loosely responsive text.

### 5.4 Attempts and limitations

Development builds `f71079a6` (name filter false positives, merchant leaks in card-declined),
`673d9304` (grounded threshold 0.3) and `fb63fc50` (before the `abroad`/placeholder fixes) are
retained, superseded. `ac402a4c` and `0db117f2` have identical rows and stats but differed in
the last float digit of one LID confidence between runs; `lid_conf` is now rounded to 6 digits, and
`fedfa10a` rebuilds byte-identically (second run `skipped`). Both are retained as the
reproducibility attempt. No model was trained or evaluated; no ML quality change is claimed.
Grounded targets teach the behaviour of answering from cited evidence in another domain; their
facts must never be served as store facts. Closing the remaining gaps needs licensed natural
RU/KK/mixed store requests, independent fluent label review and human-written answers grounded
in this store's KB/tools — each an owner decision on paid, gated or external resources.

### 5.5 Owner decision: external resources for the remaining gaps

No contact, purchase or data transfer was made. Prices and terms below are UNKNOWN until a
supplier is asked, and asking needs the owner's approval (AGENTS §4; DECISIONS open decisions).

| Option | What it would close | Verified so far | Needed from the owner |
|---|---|---|---|
| A. FutureBeeAI "Russian Retail & E-Commerce chat" (catalogue page) | natural RU store requests, many intents, human agent replies | vendor states 10,000+ chats by 150 native speakers; commercial licence, price not published | approve a quotation request; then terms, custody, PII and price review before any purchase |
| B. Custom KK and mixed RU–KK collection (same or another vendor) | KK and RU–KK store requests for all 35 intents | vendor advertises custom call-centre/chat collection in 100+ languages; Kazakh availability not confirmed | approve an inquiry with a generic brief (no project data) |
| C. mangi full ~10K KK QA | more KK support questions | public sample's authorship is undocumented; full set is a separate paid licence | only worth asking if the vendor can document human authorship in writing |
| D. Fluent native reviewers (paid or volunteer) | independent label review; human store-grounded answers from our KB | none recruited (prohibited without approval) | approve recruitment, budget and a review protocol |

Recommendation: decide B and D first; they are the only routes to KK/mixed coverage and to
answers grounded in this store's KB. A alone adds RU only.
