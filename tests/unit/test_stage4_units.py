"""Stage 4 unit tests: formatting, taxonomy v1, locale, QE and glossary, grounding, contracts,
training KB, scenarios, SFT example building and split assignment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from supportpilot.common.config import find_project_root
from supportpilot.common.grounding import Evidence, check_response, numbers_in
from supportpilot.common.schemas import (
    INTENTS,
    INTENTS_V0,
    AssistantResponse,
    GoldItem,
    SFTExample,
    Split,
    Task,
)
from supportpilot.data.gold import load_gold_sources, to_items
from supportpilot.data.kbtrain import (
    distinct_violations,
    load_docs,
    load_variants,
    product_price,
    render_all,
)
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.locale import load_locale, localize_en, target_forms
from supportpilot.data.qe import (
    QEThresholds,
    assess,
    chrf,
    kk_report,
    numbers,
    post_edit,
    repetition,
)
from supportpilot.data.scenarios import expand, load_scenarios
from supportpilot.data.sft import Resources, build_all, tasks_for
from supportpilot.data.split import (
    assign_authored,
    exact_leaks,
    lexical_leaks,
    ngram_leaks,
    semantic_leaks,
)
from supportpilot.data.taxonomy import load_taxonomy, relabel
from supportpilot.data.textfmt import FormatError, count, fmt_date, kzt, render, ru_plural

ROOT = find_project_root(Path(__file__).parent)
AUTHORED = ROOT / "authored"
LOCALE = load_locale(ROOT / "configs" / "locale_kz.yaml")
TAX = load_taxonomy(ROOT / "configs" / "taxonomy.yaml")


# ---------------------------------------------------------------- authored YAML hygiene


def _non_str_keys(node: Any, path: str = "") -> list[str]:
    bad: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if not isinstance(k, str):
                bad.append(f"{path}/{k}")
            bad += _non_str_keys(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            bad += _non_str_keys(v, f"{path}[{i}]")
    return bad


@pytest.mark.parametrize(
    "path",
    sorted(
        [
            *AUTHORED.rglob("*.yaml"),
            ROOT / "kb_sources" / "demo_store" / "facts.yaml",
            ROOT / "configs" / "taxonomy.yaml",
            ROOT / "configs" / "locale_kz.yaml",
        ]
    ),
    ids=lambda p: p.name,
)
def test_authored_yaml_has_no_split_flow_values(path: Path) -> None:
    """An unquoted comma inside a flow mapping ({ru: ул. X, 24}) silently creates an int key."""
    assert _non_str_keys(yaml.safe_load(path.read_text(encoding="utf-8"))) == []


# ---------------------------------------------------------------- textfmt


@pytest.mark.parametrize(
    ("n", "form"), [(1, "день"), (2, "дня"), (5, "дней"), (11, "дней"), (21, "день"), (24, "дня")]
)
def test_ru_plural(n: int, form: str) -> None:
    assert ru_plural(n, ("день", "дня", "дней")) == form


def test_counts_and_genitive() -> None:
    assert count(21, "day", "ru", genitive=True) == "21 дня"
    assert count(7, "day", "ru", genitive=True) == "7 дней"
    assert count(1, "wday", "ru", genitive=True) == "1 рабочего дня"
    assert count(3, "month", "en") == "3 months"
    assert count(1, "day", "en") == "1 day"
    assert count(24, "month", "kk", case="dat") == "24 айға"


def test_money_and_dates() -> None:
    assert kzt(20000, "en") == "KZT 20,000"
    assert kzt(20000, "ru") == "20 000 тенге"
    assert kzt(20000, "kk", "abl") == "20 000 теңгеден"
    assert fmt_date("2026-11-25", "ru") == "25 ноября"
    assert fmt_date("2026-11-25", "kk", "abl") == "25 қарашадан"
    assert fmt_date("2026-11-30", "kk", "dat") == "30 қарашаға"


def test_render_formats_and_errors() -> None:
    vals = {"d": 30, "c": 0, "r": [1, 3]}
    assert render("{d:days_gen} / {c:kzt_free} / {r:range_days}", vals, "ru") == (
        "30 дней / бесплатно / 1–3 дня"
    )
    assert render("{c:kzt_free}", {"c": 990}, "en") == "KZT 990"
    with pytest.raises(FormatError):
        render("{missing}", {}, "en")
    with pytest.raises(FormatError):
        render("{d:nope}", vals, "en")


# ---------------------------------------------------------------- taxonomy


def test_taxonomy_matches_schema_and_supersets_v0() -> None:
    assert set(TAX.intents) == INTENTS
    assert INTENTS_V0 <= INTENTS
    assert len(INTENTS) == 35


def test_relabel_rules_excludes_and_reference() -> None:
    contact = relabel(TAX, "bitext_retail_ecom", "customer_service", "how can I call support?", {})
    human = relabel(TAX, "bitext_retail_ecom", "customer_service", "i need to talk to support", {})
    neither = relabel(TAX, "bitext_retail_ecom", "customer_service", "customer support pls", {})
    assert (contact.intent, contact.action) == ("contact_info", "answer_kb")
    assert (human.intent, human.action, human.escalation_reason) == (
        "human_agent",
        "escalate",
        "customer_requested_human",
    )
    assert neither.status == "unlabelled"
    transfer = relabel(TAX, "bitext_retail_ecom", "pay", "how could i make a transfer", {})
    assert transfer.status == "excluded"
    fee = relabel(TAX, "bitext_customer_support", "check_cancellation_fee", "fee?", {})
    assert fee.status == "excluded"
    with_ref = relabel(
        TAX, "bitext_customer_support", "track_order", "track 123456", {"order_id": "123456"}
    )
    no_ref = relabel(TAX, "bitext_customer_support", "track_order", "track my order", {})
    assert (with_ref.action, no_ref.action) == ("call_tool", "clarify")
    reg = relabel(TAX, "bitext_customer_support", "registration_problems", "cannot sign up", {})
    assert reg.action == "clarify"
    assert relabel(TAX, "whileai_ecommerce_intent", "send", "x", {}).status == "excluded"
    assert relabel(TAX, "nope", "x", "x", {}).status == "not_in_map"


# ---------------------------------------------------------------- locale and QE


def test_localize_en_converts_dollars_to_kzt() -> None:
    out = localize_en("I want a refund of $45 for order 123456", LOCALE)
    assert "$" not in out.text
    assert "22,500 KZT" in out.text
    assert "123456" in out.text
    assert out.amounts_kzt == [22500]
    assert target_forms("deliver to Almaty for Aigerim Sadykova", "kk", LOCALE) == {
        "Aigerim Sadykova": "Әйгерім Садықова",
        "Almaty": "Алматы",
    }


def test_chrf_numbers_repetition() -> None:
    assert chrf("hello world", "hello world") == 1.0
    assert chrf("abc", "xyz") == 0.0
    assert numbers("refund 15,000 KZT and 15 000 тенге, order 482913") == {"15000", "482913"}
    assert repetition("қайтадан қайтадан қайтадан қайтадан")
    assert not repetition("where is my order please")


def test_post_edit_repairs_inflected_order_word() -> None:
    text, fixes, unresolved = post_edit("Мен 482913 тапсырмасын болдырмағым келеді", "kk")
    assert "тапсырысын" in text
    assert fixes == ["glossary:kk_order_task"]
    assert unresolved == []
    ru, fixes_ru, _ = post_edit("Статус приказа 123456 и приказов", "ru")
    assert "заказа" in ru
    assert "заказов" in ru
    assert fixes_ru == ["glossary:ru_order_decree"]
    _, _, flag = post_edit("Мен адам агентімен сөйлескім келеді", "kk")
    assert flag == ["glossary:kk_human_agent_calque"]


def test_assess_flags() -> None:
    lid = LanguageIdentifier()
    th = QEThresholds(min_bt_chrf=0.3, min_emb_sim=0.8)
    src = "where is my order 482913?"
    good = assess(
        src, "Менің 482913 тапсырысым қайда?", "kk", lid=lid, back=src, emb_sim=0.9, th=th
    )
    assert good.accept, good.hard
    lost = assess(src, "Менің тапсырысым қайда?", "kk", lid=lid, back=src, emb_sim=0.9, th=th)
    assert "numbers_changed" in lost.hard
    latin = assess(
        src, "мой uhelp cancelin order 482913", "ru", lid=lid, back=src, emb_sim=0.9, th=th
    )
    assert not latin.accept
    assert "latin_word" in latin.hard
    sms = assess(
        src,
        "Менің 482913 тапсырысым қайда? SMS келмеді",
        "kk",
        lid=lid,
        back=src,
        emb_sim=0.9,
        th=th,
    )
    assert "latin_word" not in sms.hard
    ru_in_kk = assess(
        src, "Где мой заказ 482913, я жду", "kk", lid=lid, back=src, emb_sim=0.9, th=th
    )
    assert not ru_in_kk.accept
    low = assess(src, "Менің 482913 тапсырысым қайда?", "kk", lid=lid, back="x", emb_sim=0.2, th=th)
    assert {"bt_chrf_low", "emb_sim_low"} <= set(low.hard)
    rep = kk_report("Сәлеметсіз бе, тапсырысым қайда?")
    assert rep.has_kk_letters
    assert rep.ru_function_words == 0


# ---------------------------------------------------------------- grounding and contracts


def _resp(**kw: Any) -> AssistantResponse:
    base: dict[str, Any] = {
        "language": "en",
        "intent": "return_policy",
        "action": "answer_kb",
        "answer": "You can return it within 30 days.",
        "citations": ["returns_policy:0123abcd:0000"],
    }
    base.update(kw)
    return AssistantResponse.model_validate(base)


def test_numbers_in() -> None:
    assert numbers_in("KZT 20,000 and 1 490 тенге on 2026-11-14") == {
        "20000",
        "1490",
        "2026",
        "11",
        "14",
    }


def test_check_response_grounding() -> None:
    ev = Evidence(chunks={"returns_policy:0123abcd:0000": "Returns within 30 days."})
    assert check_response(_resp(), ev) == []
    bad = check_response(_resp(answer="You can return it within 45 days."), ev)
    assert bad == ["unsupported_number:45"]
    assert check_response(_resp(citations=["other_doc:0123abcd:0001"]), ev)[0].startswith(
        "citation_not_in_context"
    )
    tool_ev = Evidence(
        chunks={}, tool_results=[{"result": {"status": "shipped", "eta": "2026-11-14"}}]
    )
    ok = _resp(
        intent="track_order",
        action="call_tool",
        answer="It arrives on 14 November.",
        citations=[],
        facts_from_tools=["eta=2026-11-14"],
    )
    assert check_response(ok, tool_ev) == []
    fake = ok.model_copy(update={"facts_from_tools": ["eta=2026-12-01"]})
    assert check_response(fake, tool_ev)[0].startswith("tool_fact_not_in_results")
    empty = _resp(citations=[], answer="Returns are fine.")
    assert check_response(empty, Evidence(chunks={})) == ["answer_without_evidence"]


def test_assistant_response_validators() -> None:
    with pytest.raises(ValidationError):
        _resp(needs_human=True)  # no reason
    with pytest.raises(ValidationError):
        _resp(action="clarify")  # no question
    with pytest.raises(ValidationError):
        _resp(citations=["not a chunk id"])
    with pytest.raises(ValidationError):
        _resp(action="escalate", needs_human=False)
    with pytest.raises(ValidationError):
        _resp(intent="not_an_intent")


def test_gold_item_review_status_rules() -> None:
    items = to_items(load_gold_sources(AUTHORED / "gold"))
    assert len(items) >= 900
    assert Counter(i.lang for i in items) == {
        "en": len(items) // 3,
        "ru": len(items) // 3,
        "kk": len(items) // 3,
    }
    assert {i.review.status.value for i in items} == {"draft"}
    data = items[0].model_dump(mode="json")
    data["review"] = {"status": "verified", "author": "x", "reviewers": []}
    with pytest.raises(ValidationError):
        GoldItem.model_validate(data)


# ---------------------------------------------------------------- training KB


def test_training_kb_renders_and_differs_from_demo() -> None:
    variants = load_variants(AUTHORED / "kb_train" / "variants.yaml")
    docs = load_docs(AUTHORED / "kb_train" / "docs.yaml")
    chunks = render_all(variants, docs, LOCALE)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert not any("{" in c["text"] or "}" in c["text"] for c in chunks)
    assert {c["lang"] for c in chunks} == {"en", "ru", "kk"}
    demo = yaml.safe_load((ROOT / "kb_sources" / "demo_store" / "facts.yaml").read_text("utf-8"))
    assert distinct_violations(variants, demo) == []
    clash = dict(demo, return_window_days=variants.variants[0]["return_window_days"])
    assert distinct_violations(variants, clash)
    # the same section of two variants renders different values
    win = {
        c["variant"]: c["text"]
        for c in chunks
        if c["lang"] == "en" and c["section_id"] == "window" and c["status"] == "current"
    }
    assert len(set(win.values())) == len(win)
    assert product_price("dala", "TV-5501", 189000, 289000) == product_price(
        "dala", "TV-5501", 189000, 289000
    )


# ---------------------------------------------------------------- scenarios and SFT


@pytest.fixture(scope="module")
def resources() -> Resources:
    variants = load_variants(AUTHORED / "kb_train" / "variants.yaml")
    chunks = pl.DataFrame(
        render_all(variants, load_docs(AUTHORED / "kb_train" / "docs.yaml"), LOCALE)
    )
    return Resources(AUTHORED, chunks, variants.variants, LOCALE, 20261111)


def test_scenarios_expand_deterministically() -> None:
    sc = load_scenarios(AUTHORED / "scenarios")
    a, b = expand(sc, LOCALE, 1), expand(sc, LOCALE, 1)
    assert a == b
    assert {u["lang"] for u in a} == {"en", "ru", "kk"}
    assert any(u["code_switch"] for u in a)
    assert any(u["history"] for u in a)
    assert all("{" not in u["text"] for u in a)
    assert len({u["record_id"] for u in a}) == len(a)


def test_build_all_authored_examples_are_valid_and_grounded(resources: Resources) -> None:
    utts = expand(load_scenarios(AUTHORED / "scenarios"), LOCALE, 20261111)
    for u in utts:
        u["utterance_key"] = u["record_id"]
    built = build_all(utts, resources, {u["scenario_id"]: Split.TRAIN for u in utts})
    assert not built.rejected, built.rejected
    tasks = Counter(e["task"] for e in built.examples)
    assert set(tasks) == {t.value for t in Task}
    for e in built.examples[:300]:
        ex = SFTExample.model_validate({k: v for k, v in e.items() if k in SFTExample.model_fields})
        assert ex.target.language == ex.lang
        target = json.loads(e["messages"][-1]["content"])
        if e["task"] == "abstain_no_evidence":
            assert target["citations"] == []
            assert target["missing_information"]
        if e["task"] == "refuse_oos":
            assert target["action"] == "refuse_oos"


def test_task_mix_is_deterministic(resources: Resources) -> None:
    utt = {"utterance_key": "k1", "action": "answer_kb", "task": None}
    assert tasks_for(utt, resources) == tasks_for(dict(utt), resources)
    assert tasks_for({"utterance_key": "k", "action": "escalate"}, resources) == [Task.ESCALATE]


# ---------------------------------------------------------------- splits and leakage


def test_assign_authored_is_stratified_and_deterministic() -> None:
    scn = [(f"s{i}", "return_policy", "answer_kb", True) for i in range(20)] + [
        (f"t{i}", "complaint", "escalate", True) for i in range(4)
    ]
    a = assign_authored(scn, {"train": 0.7, "validation": 0.15, "test": 0.15}, 7)
    assert a == assign_authored(
        list(reversed(scn)), {"train": 0.7, "validation": 0.15, "test": 0.15}, 7
    )
    esc = Counter(a[f"t{i}"] for i in range(4))
    assert esc[Split.VALIDATION] >= 1
    assert esc[Split.TEST] >= 1


def test_leak_detectors() -> None:
    train = ["where is my order please", "how do I return an item bought online"]
    held = [
        "Where is my order please!",
        "can I pay with kaspi",
        "how do I return an item bought online now",
    ]
    assert exact_leaks(train, held) == {0}
    assert set(lexical_leaks(train, held, threshold=0.8, seed=1)) == {0, 2}
    long_train = [
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen"
    ]
    assert ngram_leaks(
        long_train,
        ["x one two three four five six seven eight nine ten eleven twelve thirteen"],
        13,
    ) == {0}
    import numpy as np

    tv = np.array([[1.0, 0.0]], dtype=np.float32)
    hv = np.array([[0.99, 0.141], [0.0, 1.0]], dtype=np.float32)
    assert set(semantic_leaks(tv, hv, 0.95)) == {0}
