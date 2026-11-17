"""Commissioned human delivery checks (humanintake, DATA_CONTRACTS §20.2)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from supportpilot.common.config import find_project_root
from supportpilot.data.acquire import verify_raw
from supportpilot.data.humanintake import (
    Adjudication,
    Answer,
    AnswerReview,
    Batch,
    Contributor,
    Delivery,
    LabelReview,
    Request,
    assess,
    cohen_kappa,
    load_delivery,
    seal_delivery,
)
from supportpilot.data.taxonomy import Taxonomy, load_taxonomy
from tests.data_helpers import entry

ROOT = find_project_root(Path(__file__).parent)
KB = frozenset({"delivery_policy:abcd1234:0000", "returns_policy:abcd1234:0001"})
ALL_ROLES = ["author", "reviewer", "adjudicator", "answer_author", "answer_reviewer"]


@pytest.fixture(scope="module")
def tax() -> Taxonomy:
    return load_taxonomy(ROOT / "configs" / "taxonomy.yaml")


def person(pid: str, langs: dict[str, str], **kw: Any) -> Contributor:
    data: dict[str, Any] = {
        "contributor_id": pid,
        "roles": ALL_ROLES,
        "languages": langs,
        "competence_evidence": "supplier screening test 2026-11",
        "identity_verified": True,
        "independence_attested": True,
        "no_llm_attested": True,
    }
    data.update(kw)
    return Contributor.model_validate(data)


BILINGUAL = {"ru": "native", "kk": "native"}
PEOPLE = [
    person("p-author", BILINGUAL),
    person("p-rev1", BILINGUAL),
    person("p-rev2", BILINGUAL),
    person("p-adj", BILINGUAL),
    person("p-ru-only", {"ru": "native"}),
    person("p-unattested", BILINGUAL, no_llm_attested=False),
]


def batch() -> Batch:
    return Batch(
        schema_version="human_intake/1",
        batch_id="pilot-01",
        supplier="test supplier",
        collection_method="written by screened native speakers from need briefs",
        rights_evidence="assignment of rights in the signed statement of work",
        consent_evidence="written contributor consent forms held by the supplier",
        kb_version="kb_demo_store-20261113-f3b204a6",
        delivered_at=date(2026, 11, 17),
    )


def req(item: str, text: str, variant: str = "ru", **kw: Any) -> Request:
    data: dict[str, Any] = {
        "item_id": item,
        "author_id": "p-author",
        "scenario_id": f"need-{item}",
        "variant": variant,
        "preferred_reply_lang": "kk" if variant == "mixed" else variant,
        "method": "written",
        "text": text,
    }
    data.update(kw)
    return Request.model_validate(data)


def lab(item: str, reviewer: str, intent: str = "delivery_time", **kw: Any) -> LabelReview:
    data: dict[str, Any] = {
        "item_id": item,
        "reviewer_id": reviewer,
        "natural": True,
        "variant": "ru",
        "intent": intent,
        "action": "answer_kb",
    }
    data.update(kw)
    return LabelReview.model_validate(data)


def good_review(item: str, reviewer: str, **kw: Any) -> AnswerReview:
    data: dict[str, Any] = {
        "item_id": item,
        "reviewer_id": reviewer,
        "responsive": True,
        "complete": True,
        "claims_supported": True,
        "citations_correct": True,
        "abstains_when_unsupported": True,
        "language_ok": True,
    }
    data.update(kw)
    return AnswerReview.model_validate(data)


def test_agreed_and_adjudicated_labels_are_admitted(tax: Taxonomy) -> None:
    d = Delivery(
        batch(),
        PEOPLE,
        [req("a1", "Когда привезут заказ в Шымкент?"), req("a2", "Можно вернуть кроссовки?")],
        [
            lab("a1", "p-rev1"),
            lab("a1", "p-rev2"),
            lab("a2", "p-rev1", "return_policy"),
            lab("a2", "p-rev2", "return_product"),
        ],
        [
            Adjudication(
                item_id="a2",
                adjudicator_id="p-adj",
                accept=True,
                variant="ru",
                intent="return_policy",
                action="answer_kb",
            )
        ],
    )
    res = assess(d, tax, KB)
    status = {a.request.item_id: (a.label_status, a.intent) for a in res.admitted}
    assert status == {
        "a1": ("human_agreed", "delivery_time"),
        "a2": ("human_adjudicated", "return_policy"),
    }
    assert res.report["labels"] == {"human_agreed": 1, "human_adjudicated": 1}
    assert res.report["agreement"]["ru"]["intent"]["raw"] == 0.5
    assert res.report["independent_scenarios"] == 2
    assert "delivery_time" not in res.report["intents_missing"]


def test_independence_competence_and_attestation_are_enforced(tax: Taxonomy) -> None:
    mixed = "Заказым әлі келмеді, где мой заказ?"
    d = Delivery(
        batch(),
        PEOPLE,
        [
            req("b1", "Когда доставка?"),  # one reviewer is the author
            req("b2", mixed, "mixed"),  # one reviewer lacks Kazakh
            req("b3", "Когда доставка в Алматы?", author_id="p-unattested"),
            req("b4", "Доставка в Тараз сколько дней?"),  # reviewers disagree, no adjudicator
            req("b5", "ааа ббб ввв"),  # judged unnatural
        ],
        [
            lab("b1", "p-author"),
            lab("b1", "p-rev1"),
            lab("b2", "p-rev1", "track_order", variant="mixed", action="clarify"),
            lab("b2", "p-ru-only", "track_order", variant="mixed", action="clarify"),
            lab("b3", "p-rev1"),
            lab("b3", "p-rev2"),
            lab("b4", "p-rev1"),
            lab("b4", "p-rev2", "delivery_options_cost"),
            lab("b5", "p-rev1", natural=False),
            lab("b5", "p-rev2", natural=False),
        ],
    )
    res = assess(d, tax, KB)
    assert not res.admitted
    reasons = {x["item_id"]: x["reason"] for x in res.excluded}
    assert reasons == {
        "b1": "insufficient_independent_review",
        "b2": "insufficient_independent_review",
        "b3": "author_unattested_contributor",
        "b4": "unresolved_disagreement",
        "b5": "not_natural",
    }
    assert res.report["rejected_label_reviews"] == {
        "language_not_competent": 1,
        "not_independent": 1,
    }


def test_mixed_variant_needs_bilingual_reviewers_and_matching_variant(tax: Taxonomy) -> None:
    text = "Заказым әлі келмеді, где мой заказ?"
    reviews = [
        lab(i, r, "track_order", variant=v, action="clarify")
        for i, v in (("c1", "mixed"), ("c2", "kk"))
        for r in ("p-rev1", "p-rev2")
    ]
    d = Delivery(batch(), PEOPLE, [req("c1", text, "mixed"), req("c2", text, "mixed")], reviews)
    res = assess(d, tax, KB)
    assert [a.request.item_id for a in res.admitted] == ["c1"]
    assert res.excluded == [
        {"item_id": "c2", "reason": "variant_mismatch", "detail": "declared mixed, reviewed kk"}
    ]
    assert res.report["contributors"]["ru_kk_competent"] == 5


def test_store_grounded_answers_need_kb_citations_and_two_passing_reviews(
    tax: Taxonomy,
) -> None:
    items = ["d1", "d2", "d3", "d4"]
    reqs = [req(i, f"Сколько дней доставка в город номер {n}?") for n, i in enumerate(items)]
    reviews = [lab(i, r) for i in items for r in ("p-rev1", "p-rev2")]

    def ans(item: str, **kw: Any) -> Answer:
        data: dict[str, Any] = {
            "item_id": item,
            "author_id": "p-adj",
            "lang": "ru",
            "behaviour": "answer_kb",
            "text": "Доставка по городу занимает 1–2 дня.",
            "citations": ["delivery_policy:abcd1234:0000"],
        }
        data.update(kw)
        return Answer.model_validate(data)

    answers = [
        ans("d1"),
        ans("d2", citations=["invented_doc:00000000:0000"]),
        ans("d3"),
        ans("d4", lang="kk"),
    ]
    a_reviews = [
        good_review("d1", "p-rev1"),
        good_review("d1", "p-rev2"),
        good_review("d2", "p-rev1"),
        good_review("d2", "p-rev2"),
        good_review("d3", "p-rev1"),
        good_review("d3", "p-rev2", claims_supported=False),
    ]
    d = Delivery(batch(), PEOPLE, reqs, reviews, [], answers, a_reviews)
    res = assess(d, tax, KB)
    by_id = {a.request.item_id: a for a in res.admitted}
    assert set(by_id) == set(items)  # the request rows stay admitted
    assert by_id["d1"].answer is not None
    assert all(by_id[i].answer is None for i in ("d2", "d3", "d4"))
    assert res.report["answers"]["outcome"] == {
        "admitted": 1,
        "answer_failed_review": 1,
        "answer_language_mismatch": 1,
        "citation_not_in_kb": 1,
        "delivered": 4,
    }
    assert res.report["answers"]["rubric_pass_rate"]["ru"]["claims_supported"] == 0.8333


def test_cohen_kappa() -> None:
    assert cohen_kappa([]) is None
    assert cohen_kappa([("a", "a"), ("b", "b")]) == 1.0
    assert cohen_kappa([("a", "a"), ("a", "a")]) is None  # one class: undefined
    assert cohen_kappa([("a", "b"), ("b", "a")]) == -1.0


def _write(root: Path, d: Delivery) -> None:
    root.mkdir(parents=True)
    (root / "batch.json").write_text(d.batch.model_dump_json(), encoding="utf-8")
    for name, rows in (
        ("contributors", d.contributors),
        ("requests", d.requests),
        ("label_reviews", d.label_reviews),
    ):
        lines = [r.model_dump_json() for r in rows]
        (root / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_delivery_files_are_strict_and_sealed(tmp_path: Path, tax: Taxonomy) -> None:
    d = Delivery(batch(), PEOPLE, [req("e1", "Когда доставка?")], [lab("e1", "p-rev1")])
    e = entry("intake_pilot_01", fetch=None, status="candidate", languages=["ru", "kk"])
    root = tmp_path / "raw" / e.source_id / e.revision
    _write(root, d)
    loaded = load_delivery(root)
    assert loaded.requests == d.requests
    seal_delivery(root, e)
    assert verify_raw(root) == []
    with pytest.raises(ValueError, match="already sealed"):
        seal_delivery(root, e)
    with (root / "requests.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(req("e2", "Ещё вопрос").model_dump_json() + "\n")
    assert verify_raw(root)  # a later change is detected

    (root / "notes.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected files"):
        load_delivery(root)
    (root / "notes.txt").unlink()
    (root / "requests.jsonl").write_text(
        req("e1", "x").model_dump_json() + "\n" + req("e1", "y").model_dump_json() + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate request"):
        load_delivery(root)


def test_records_for_unknown_items_and_machine_origins_are_rejected(tax: Taxonomy) -> None:
    d = Delivery(batch(), PEOPLE, [req("f1", "Когда доставка?")], [lab("zz", "p-rev1")])
    with pytest.raises(ValueError, match="unknown items"):
        assess(d, tax, KB)
    with pytest.raises(ValueError, match="method"):
        Request.model_validate({**json.loads(req("f2", "x").model_dump_json()), "method": "mt"})
