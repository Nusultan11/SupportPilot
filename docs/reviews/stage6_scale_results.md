# Stage 6.1 evidence and error review

Reviewer: Codex, LLM author self-review, 2026-11-13. No independent human or fluent KK
review. Judgements remain drafts. This review adds interpretation, never edits frozen
questions, qrels, the selection rule or holdout observations.

Inputs: KB `kb_demo_store-20261113-f3b204a6`, primary freeze `6fab32c16b3a6e05b29f`;
development run `1bdd4dd13dac4d85afdf8f9872210da8`, holdout
`8cb7cbf9088949f98f239397a09a6830`. Full metrics, clustered CIs, observations and source-bound
report live in `artifacts/reports/retrieval_scale/6fab32c16b3a6e05b29f/`.

## Selection and uncertainty

The registered procedure retains BGE-M3 weighted .75 + BGE rerank20, context 1024,
without a confidence gate. Every ranking alternative violates the cross-language recall
guard or has no significant positive nDCG difference. Context 512/2048 changes packing,
not ranking, and does not provide the required ranking improvement. No latency tie-break
was needed. This is not evidence that the larger corpus improves semantic quality.
Because selection retains the control itself, selected-minus-control holdout deltas are
identity comparisons (zero by construction), not independent evidence of improvement.
The nontrivial comparisons are the development alternatives and the explicitly exposed
matched historical growth diagnostic.

For example, E5-base hybrid + the same reranker has development cross Recall@5 .72748
versus control .89114. Gate .01 reduces development FPR only from 1.0 to .96730 while
cross recall falls to .83108; gate .5 gives FPR .67352, cross recall .49399 and answerable
FNR .07836. These negative alternatives stay recorded; no threshold is retuned on holdout.

Fresh holdout: 1,968 variants, 588 needs, 28 families; 1,362 answerable and 606 negative
queries. Scenario-cluster 95% CIs:

| Slice | Recall@5 | packed sufficiency |
|---|---|---|
| RU, 386 answerable needs | .90155 [.88083, .92098] | .88601 [.84974, .91710] |
| KK, 386 answerable needs | .93264 [.91580, .94822] | .88860 [.85492, .91969] |
| EN, 386 answerable needs | .97927 [.96891, .98834] | .99223 [.98187, 1.00000] |
| Cross, 162 variants / 27 needs | .89506 [.81481, .96914] | .91975 [.85802, .97531] |
| Mixed, 42 needs | 1.00000 [1, 1] | 1.00000 [1, 1] |

Family sensitivity is required: 303 multi-document variants cover 101 needs but only two
reasoning families; temporal holdout has 15 variants / three needs / one family. A
one-family bootstrap cannot identify between-family uncertainty. These counts restrict
generalization even when scenario intervals or empirical all-success intervals are narrow.
They do not certify native fluency, human relevance or independent natural-customer intents.

## Complete failure accounting and representative source checks

The selected holdout census includes 161 qrel misses at five, 18 cases missing a required
group at ten, and 103 insufficient packed contexts. Of these 103, **85 had all required
proof in top-10 and lost it during packing**; the other 18 already lacked proof in ranking.
Labels overlap. No answerable query had an empty context: FNR zero does not mean proof
sufficiency. The automated inspection sample traces 50 distinct failed needs by RU query,
50 by KK and all 21 available by EN. `report/qualitative_audit.json` records each sample's
actual missing source, rank, text and validity; this is a programmatic trace with LLM
representative inspection, not 121 independent human annotations.

| Representative | Observed failure and source evidence |
|---|---|
| `rq_catalog_backpack_sprayer_pressure_bar_en_kk` | Correct delivery-pressure attribute is behind other sprayer models. Same predicate and unit do not identify the requested product. |
| `rq_catalog_backpack_sprayer_pressure_bar_kk_en` | Packed confusers include speaker impedance, batteries and suction pressure; the requested backpack sprayer reaches the later ranking. Cross-language entity/predicate alignment fails despite numerical/unit cues. |
| `rq_catalog_bath_towel_iron_max_c_en_kk` | Washing limits for other textiles displace the towel's ironing limit. The actual towel source states 150 C for ironing; washing temperature is a different operation. |
| `rq_catalog_compact_camera_focal_max_mm_kk_ru` | Correct 120 mm fact is at rank eight but absent from the packed context. Other camera models' focal limits cannot replace it. |
| `rq_catalog_curtain_panel_spin_max_rpm_en_kk` | Curtain spin-speed proof at rank eight is dropped; tool spindle speeds share rpm but have different subjects and operations. |
| `rq_catalog_pickup_mass_air_washer_ru_ru` | Product mass 8,100 g is first; the separate pickup 20 kg parcel limit is at rank nine. Packing includes seven product attributes and omits the delivery source. Product mass is not packed-parcel mass or live pickup availability. |
| `rq_compare_payment_retry_dates_en_en` | Archive 10-minute proof is at rank two; current 15-minute proof is at rank eight and lost. Both dates were explicitly requested. All-versions retrieval preserves dates but does not guarantee both proofs survive packing. |
| `rq_catalog_missing_aa_battery_charger_repair_parts_price_en_en` | Packed material includes retail price and electrical specifications. Neither supplies the unstated replacement-part price. |

The dominant pickup limit has exactly one placement section per language, so this packing
finding is not an omitted alternative citation to the same canonical fact. Other qrels
remain non-exhaustive drafts; unjudged equivalent evidence can affect individual cases.
All 13 extraction gates pass with every original chunk retained: the observed missing
ranked/packed proof is not evidence that the canonical facts disappeared during ingestion.

Multi-document full-evidence@10 is .96700, yet packed sufficiency is .70297
[.64356, .75908]. High MRR (.99505) establishes early relevant evidence, not a complete
answer. The next generation/grounding layer must reject incomplete evidence; this review
does not tune the frozen context budget using these holdout errors.

## Abstention and blind coverage limits

FPR is **1.0 on all 606 holdout negatives**, representing 202 needs: 603 partial-evidence
variants and three missing-exception variants. The selected retrieval gate accepts related
background; this is not a measurement of a generated answer because no generation layer
is evaluated here. Live API, conflicting/clarification and outside-validity negatives are
represented in development, not established as fresh blind success. Cross and mixed
holdout contain no negatives, so their FPR is **UNKNOWN**, not zero. All-negative acceptance
and the development threshold trade-off leave semantic abstention unresolved.

## Supplemental wording diagnostic

After development exposed near-saturated label-echo monolingual performance, a separate
plan was recorded before inspecting holdout metrics. It samples four existing dev needs
per each of 20 domains: 80 existing needs / five families, 620 variants, **zero additional
independent needs**. Native paraphrases, one internal entity-letter deletion, cross-language
and mixed variants keep the original facts/proof groups. Source and generated text hashes
are checked against `diagnostic_plan.json`. SFT/gold exact, lexical and ngram overlap is
zero; semantic near-SFT count is zero. Run `fece36d08d3c41d98dcda4c217e1e2da`, diagnostic
freeze `1ad2bbc278800f1f260c`.

RU/KK/EN Recall@5 remains 1.0 on 160 variants per language. EN nDCG is .96309
[.93541, .98616], while RU/KK nDCG is 1.0. Both mono wording groups retain recall but have
overall nDCG .98770. Cross-language recall is .83333 [.73333, .91687] on 120 variants /
20 needs. This cross cohort differs from the main benchmark and is not a paired decline
claim. Paired mono deltas align the same original native query/need, language, corpus and
proof; cross/mixed variants are excluded from those deltas. There are no auxiliary negatives.
No configuration changes follow this post-selection diagnostic; it is not fresh holdout
confirmation or broad natural-dialogue acceptance.

## Performance and recovery interpretation

On identical 681 historical variants, control p50/p95 is 663.15/903.48 ms at 369 chunks
and 687.57/889.45 ms at 8,865 chunks. Paired recall growth deltas are RU/EN zero,
KK −.00613 [−.01840, 0], cross −.02174 [−.04710, 0]; these are exposed-cohort diagnostics,
not statistically confirmed improvement. The balanced new 80-variant selected dev probe
has p50/p95 579.72/629.21 ms, 1.758 sequential requests/s; its query mix differs.

Peak selected process RSS is 5,839,515,648 bytes, Torch VRAM allocated/reserved
2,301,659,136 / 2,334,130,176 bytes. Sampled Qdrant cgroup peaks include file cache:
2,144,235,520 bytes current, 139,792,384 anon, 1,980,297,216 file; these maxima occur in
their own sampling window and are not whole-VM or per-collection RSS. Apparent disk
sizes include sparse/preallocated files: allocated sizes are 11,112,448 bytes baseline,
97,677,312 expanded and 489,938,944 synthetic100k. `measurement_scope_audit.json` separates
`du -sb` apparent and `du -s -B1` allocated measurements. Cached/uncached evidence agrees
on all 80 sampled requests; this does not establish bitwise model reproduction everywhere.

Full uncached document encoding: BGE-M3 55.61 units/s, E5-base 164.10, on all 8,865 units
per model, warm weights and no vector cache. Minimum cosine to stored vectors is .999901
and .999968, respectively; max absolute drift .001618/.001104 reflects numerical/batch
padding sensitivity. Index restore verifies stored vectors separately: all 8,865 dense and
sparse vectors/payloads match (max dense delta zero), six ordinary serving evidence results
agree, and synchronization without inference takes 12.945 s. Random 100k ANN fidelity,
real snapshot restore and vector-update/rollback evidence remain separate from semantic
quality and from neural encoding throughput. No production SLO or concurrency is inferred.
