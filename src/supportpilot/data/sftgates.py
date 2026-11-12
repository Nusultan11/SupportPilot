"""Quality gates for the SFT dataset (DATA_CONTRACTS §10, §11). Re-read the written files.

schema, ids, language (user text and answer), provenance and licence, grounding (re-run
from the stored messages), evidence completeness per task, duplicates, leakage (ids, exact,
lexical; semantic from the build report), placeholders, PII, training-KB distinctness and
held-out coverage (every intent in validation and test in en, ru and kk; ADR-029).
The language mix and per-language Kazakh indicators are reported, not gated.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from supportpilot.common.config import Settings
from supportpilot.common.grounding import Evidence, check_response
from supportpilot.common.schemas import (
    INTENTS,
    AssistantResponse,
    SFTExample,
    SourceEntry,
    Task,
)
from supportpilot.data.kbtrain import distinct_violations, load_variants
from supportpilot.data.langid import KK_LETTERS, LanguageIdentifier
from supportpilot.data.mtrun import systems
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.qe import kk_report
from supportpilot.data.scenarios import AUTHORED_SOURCE
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.textnorm import EMAIL_RE, PHONE_RE

_TEMPLATE_RE = re.compile(r"\{\{|\{[a-z_][a-z0-9_.]*(?::[a-z_@]+)?\}")
_SYNTH_PHONE = re.compile(r"\+7 700 555 01\d\d|\+1 202 555 01\d\d")
SPLITS = ("train", "validation", "test")


def _gate(name: str, ok: bool, metric: float, threshold: float, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if ok else "fail",
        "metric": metric,
        "threshold": threshold,
        "details": details,
    }


def _load(build: Path) -> pl.DataFrame:
    return pl.concat([pl.read_parquet(build / f"{s}.parquet") for s in SPLITS], how="vertical")


def evidence_of(messages: list[dict[str, Any]]) -> tuple[Evidence, list[str]]:
    chunks: dict[str, str] = {}
    tools: list[dict[str, Any]] = []
    users: list[str] = []
    for m in messages:
        if m["role"] == "user":
            users.append(m["content"])
        elif m["role"] == "assistant" and m is not messages[-1]:
            users.append(m["content"])  # earlier assistant turns are part of the dialogue
        elif m["role"] == "tool":
            payload = json.loads(m["content"])
            if m["name"] == "kb_search":
                for r in payload["results"]:
                    dates = " ".join(d for d in (r["effective_from"], r["effective_to"]) if d)
                    chunks[r["chunk_id"]] = f"{r['text']} {dates}".strip()
            else:
                tools.append(payload)
    return Evidence(chunks=chunks, tool_results=tools, user_texts=users), list(chunks)


def run_sft_gates(
    build: Path,
    settings: Settings,
    entries: Sequence[SourceEntry],
    chunks: pl.DataFrame,
    leak_report: dict[str, Any],
) -> dict[str, Any]:
    cfg = settings.stage4
    df = _load(build)
    rows = df.to_dicts()
    gates: list[dict[str, Any]] = []

    # schema
    bad_schema: Counter[str] = Counter()
    parsed: list[SFTExample] = []
    for r in rows:
        try:
            ex = SFTExample.model_validate(
                {
                    **{
                        k: r[k]
                        for k in SFTExample.model_fields
                        if k in r and k not in ("messages", "target")
                    },
                    "messages": json.loads(r["messages_json"]),
                    "target": json.loads(r["target_json"]),
                }
            )
            parsed.append(ex)
        except Exception as exc:
            bad_schema[type(exc).__name__] += 1
    gates.append(
        _gate("schema", not bad_schema, sum(bad_schema.values()), 0, errors=dict(bad_schema))
    )

    # ids
    ids = Counter(r["example_id"] for r in rows)
    dup_ids = sum(1 for v in ids.values() if v > 1)
    gates.append(_gate("ids", dup_ids == 0, dup_ids, 0, examples=len(rows)))

    # language: user text and answer
    lid = LanguageIdentifier()
    users = lid.detect_many([r["user_text"] for r in rows])
    answers = [AssistantResponse.model_validate_json(r["target_json"]).answer for r in rows]
    ans_lid = lid.detect_many(answers)
    user_bad: Counter[str] = Counter()
    ans_bad: Counter[str] = Counter()
    for r, u, a in zip(rows, users, ans_lid, strict=True):
        if not (u.lang == r["lang"] or r["code_switch"] or u.code_switch or u.lang == "und"):
            user_bad[f"{r['lang']}->{u.lang}|{r['origin']}"] += 1
        if a.lang != r["lang"]:
            ans_bad[f"{r['lang']}->{a.lang}"] += 1
    n = max(1, len(rows))
    user_ok = 1 - sum(user_bad.values()) / n
    ans_ok = 1 - sum(ans_bad.values()) / n
    gates.append(
        _gate(
            "language",
            user_ok >= settings.clean.min_lid_agreement and ans_ok >= 0.999,
            round(min(user_ok, ans_ok), 5),
            settings.clean.min_lid_agreement,
            user_text_agreement=round(user_ok, 5),
            answer_agreement=round(ans_ok, 5),
            user_mismatch=dict(user_bad.most_common(10)),
            answer_mismatch=dict(ans_bad.most_common(10)),
        )
    )

    # provenance + licence
    train_sources = set()
    for e in entries:
        try:
            check_use(e, Purpose.TRAINING)
            train_sources.add(e.source_id)
        except Exception:  # noqa: S112
            continue
    mt_ok = {sid for sid, s in systems(settings).items() if s.training_use}
    lic_bad = Counter(
        r["source_id"] for r in rows if r["source_id"] not in train_sources | {AUTHORED_SOURCE}
    )
    mt_bad = Counter(
        r["translation_system"]
        for r in rows
        if r["origin"] == "machine_translated" and r["translation_system"] not in mt_ok
    )
    prov_bad = sum(
        1
        for r in rows
        if not r["record_id"].startswith(r["source_id"] + ":")
        or (r["origin"] == "machine_translated" and not r["translation_of"])
    )
    gates.append(
        _gate(
            "licence_provenance",
            not lic_bad and not mt_bad and prov_bad == 0,
            sum(lic_bad.values()) + sum(mt_bad.values()) + prov_bad,
            0,
            bad_sources=dict(lic_bad),
            bad_mt_systems=dict(mt_bad),
            lineage_errors=prov_bad,
            train_sources=sorted(train_sources | {AUTHORED_SOURCE}),
        )
    )

    # grounding re-run + evidence completeness per task
    g_bad: Counter[str] = Counter()
    e_bad: Counter[str] = Counter()
    for ex in parsed:
        ev, shown = evidence_of([m.model_dump() for m in ex.messages])
        for p in check_response(ex.target, ev):
            g_bad[p.split(":")[0]] += 1
        t = ex.target
        task = ex.task
        if task in (Task.GROUNDED_ANSWER, Task.INJECTION_RESIST) and not t.citations:
            e_bad["grounded_without_citation"] += 1
        if task is Task.TOOL_ANSWER and not ev.tool_results:
            e_bad["tool_answer_without_tool"] += 1
        if task is Task.TOOL_ANSWER and not t.facts_from_tools:
            e_bad["tool_answer_without_facts"] += 1
        if task is Task.ABSTAIN_NO_EVIDENCE and (t.citations or not t.missing_information):
            e_bad["abstain_shape"] += 1
        if task is Task.ESCALATE and not t.needs_human:
            e_bad["escalate_without_needs_human"] += 1
        if task is Task.CLARIFY and not t.clarification_question:
            e_bad["clarify_without_question"] += 1
        if task is Task.REFUSE_OOS and (t.citations or t.intent != "out_of_scope"):
            e_bad["refuse_shape"] += 1
        if task is Task.CONFLICTING_CONTEXT and len(shown) < 2:
            e_bad["conflict_without_two_sources"] += 1
        if not set(t.citations) <= set(shown):
            e_bad["citation_not_shown"] += 1
    gates.append(_gate("grounding", not g_bad, sum(g_bad.values()), 0, problems=dict(g_bad)))
    gates.append(_gate("evidence", not e_bad, sum(e_bad.values()), 0, problems=dict(e_bad)))

    # duplicates within split
    dup = 0
    for s in SPLITS:
        msgs = [r["messages_json"] for r in rows if r["split"] == s]
        dup += len(msgs) - len(set(msgs))
    gates.append(_gate("duplicates", dup == 0, dup, 0))

    # leakage: ids, exact, lexical (recomputed); semantic from the build
    train = [r for r in rows if r["split"] == "train"]
    held = [r for r in rows if r["split"] != "train"]
    tr_ids = {r["scenario_id"] for r in train} | {r["family_id"] for r in train}
    id_leak = sum(1 for r in held if r["scenario_id"] in tr_ids or r["family_id"] in tr_ids)
    cross: Counter[str] = Counter()
    val_s = {r["scenario_id"] for r in rows if r["split"] == "validation"}
    test_s = {r["scenario_id"] for r in rows if r["split"] == "test"}
    cross["test-validation"] = len(val_s & test_s)
    validation = sorted({r["user_text"] for r in rows if r["split"] == "validation"})
    test = sorted({r["user_text"] for r in rows if r["split"] == "test"})
    cross["test-validation-exact"] = len(exact_leaks(validation, test))
    cross["test-validation-lexical"] = len(
        lexical_leaks(validation, test, threshold=cfg.leak_jaccard, seed=cfg.seed)
    )
    cross["test-validation-ngram"] = len(ngram_leaks(validation, test, cfg.leak_ngram))
    tt = sorted({r["user_text"] for r in train})
    ht = sorted({r["user_text"] for r in held})
    ex_leak = len(exact_leaks(tt, ht))
    lex_leak = len(lexical_leaks(tt, ht, threshold=cfg.leak_jaccard, seed=cfg.seed))
    total = id_leak + sum(cross.values()) + ex_leak + lex_leak
    gates.append(
        _gate(
            "leakage",
            total == 0,
            total,
            0,
            scenario_family_overlap=id_leak,
            heldout_scenario_overlap=dict(cross),
            exact_text=ex_leak,
            lexical_near_copy=lex_leak,
            semantic_filtered_at_build=leak_report.get("semantic", 0),
            semantic_checked=leak_report.get("semantic_checked", False),
        )
    )

    # placeholders and PII in what the model reads and writes
    ph = sum(
        1
        for r, ans in zip(rows, answers, strict=True)
        if _TEMPLATE_RE.search(r["user_text"] + " " + ans)
    )
    pii = 0
    for r, ans in zip(rows, answers, strict=True):
        text = r["user_text"] + " " + ans
        for m in EMAIL_RE.finditer(text):
            if not m.group(0).lower().endswith((".example", "@example.com")):
                pii += 1
        for m in PHONE_RE.finditer(text):
            if not _SYNTH_PHONE.search(m.group(0)):
                pii += 1
    gates.append(_gate("placeholders", ph == 0, ph, 0))
    gates.append(_gate("pii", pii == 0, pii, 0))

    # training KB distinct from the demo store
    authored = settings.resolve(cfg.authored_dir)
    demo = yaml.safe_load(settings.resolve(cfg.demo_facts).read_text(encoding="utf-8"))
    viol = distinct_violations(load_variants(authored / "kb_train" / "variants.yaml"), demo)
    gates.append(_gate("kb_variant_distinct", not viol, len(viol), 0, violations=viol[:10]))

    # held-out coverage (ADR-029): every intent in validation and test, in every language
    intents = sorted(INTENTS)
    have = {(r["split"], r["lang"], r["intent"]) for r in rows}
    gaps = [
        f"{split}|{lang}|{intent}"
        for split in SPLITS
        for lang in ("en", "ru", "kk")
        for intent in intents
        if (split, lang, intent) not in have
    ]
    gates.append(
        _gate(
            "heldout_coverage",
            not gaps,
            len(gaps),
            0,
            intents=len(intents),
            missing=gaps[:40],
        )
    )

    # report: language mix and Kazakh indicators (not gated)
    kk_rows = [(r, ans) for r, ans in zip(rows, answers, strict=True) if r["lang"] == "kk"]
    kk_user = [kk_report(r["user_text"]) for r, _ in kk_rows]
    kk_ans = [kk_report(a) for _, a in kk_rows]
    kk_info = {
        "examples": len(kk_rows),
        "user_text_with_kk_letters": round(
            sum(k.has_kk_letters for k in kk_user) / max(1, len(kk_user)), 4
        ),
        "answer_with_kk_letters": round(
            sum(k.has_kk_letters for k in kk_ans) / max(1, len(kk_ans)), 4
        ),
        "user_text_ru_function_words": sum(k.ru_function_words > 0 for k in kk_user),
        "kk_letters": "".join(sorted(KK_LETTERS)),
    }
    mix = {s: dict(Counter(r["lang"] for r in rows if r["split"] == s)) for s in SPLITS}
    passed = all(g["status"] == "pass" for g in gates)
    return {"passed": passed, "gates": gates, "language_mix": mix, "kazakh": kk_info}
