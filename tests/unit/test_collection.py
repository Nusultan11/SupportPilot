"""Commissioned collection: briefs, delivery contract and intake checks (ADR-049)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from supportpilot.common.schemas import INTENTS, FetchSpec
from supportpilot.data.collection import (
    BRIEFS_FILE,
    CollectionError,
    check_delivery,
    make_briefs,
)
from supportpilot.data.taxonomy import NEEDS_REFERENCE
from tests.collection_helpers import ROWS, contributor, plan, taxonomy, write_delivery
from tests.data_helpers import entry


def test_briefs_cover_every_intent_with_distinct_deterministic_scenarios() -> None:
    p, tax = plan(scenarios_per_intent=10), taxonomy()
    briefs = make_briefs(p, tax)
    assert briefs == make_briefs(p, tax)
    assert len(briefs) == len(INTENTS) * 10 == len({b.brief_id for b in briefs})
    for intent in INTENTS:
        mine = [b for b in briefs if b.intent == intent]
        assert len({tuple(sorted(b.facets.items())) for b in mine}) == 10
        needs_ref = tax.intents[intent].action == NEEDS_REFERENCE
        assert all(("reference" in b.facets) == needs_ref for b in mine)
        assert all(("topic" in b.facets) == (intent == "out_of_scope") for b in mine)


def test_valid_delivery_passes_and_counts_rows_scenarios_writers(tmp_path: Path) -> None:
    rep = check_delivery(write_delivery(tmp_path, plan()), plan(), taxonomy())
    assert rep.ok, rep.problems
    assert rep.rows == len(ROWS)
    assert len(rep.content_sha256) == 64
    ru = rep.slices["ru"]
    assert (ru.rows, ru.scenarios, ru.writers, ru.intents, ru.scenarios_full) == (4, 2, 2, 2, 2)
    assert rep.slices["ru_kk"].flagged.get("code_switch_mismatch", 0) == 0
    assert any("incomplete" in w for w in rep.warnings)  # 2 of 35 scenarios per slice


@pytest.mark.parametrize(
    ("row", "fragment"),
    [
        (("w01", "sc-nope-01", "ru", "ru", "Где мой заказ, подскажите"), "unknown brief"),
        (("w09", "sc-place_order-01", "ru", "ru", "Как оформить заказ на сайте?"), "not a writer"),
        (("w01", "sc-place_order-01", "kk", "ru", "Как оформить заказ на сайте?"), "!= slice"),
        (("w01", "sc-place_order-01", "ru", "ru", "Позвоните мне +7 701 234 56 78"), "personal"),
        (("w01", "sc-track_order-01", "ru", "ru", "Где посылка? Жду очень долго"), "wrote 2 rows"),
        (
            ("w02", "sc-place_order-01", "ru", "ru", "Где мой заказ <order_id>, жду уже неделю!"),
            "exact",
        ),
    ],
)
def test_blocking_row_problems(
    tmp_path: Path, row: tuple[str, str, str, str, str], fragment: str
) -> None:
    rep = check_delivery(write_delivery(tmp_path, plan(), [*ROWS, row]), plan(), taxonomy())
    assert any(fragment in p for p in rep.problems), rep.problems


def test_contributor_rights_and_language_competence_are_enforced(tmp_path: Path) -> None:
    bad = contributor("w01")
    bad["attestation"]["no_llm"] = False
    ru_only = contributor("w02", native=["ru"])
    rep = check_delivery(
        write_delivery(tmp_path, plan(), contributors=[bad, ru_only]), plan(), taxonomy()
    )
    assert any("w01: attestation incomplete" in p for p in rep.problems)
    assert any("writer w02 lacks ['kk']" in p for p in rep.problems)


def test_tampered_briefs_and_fast_rows_block_intake(tmp_path: Path) -> None:
    path = write_delivery(tmp_path, plan(), duration=3)
    (path / BRIEFS_FILE).write_text("brief_id,intent,description,facets\n", encoding="utf-8")
    rep = check_delivery(path, plan(), taxonomy())
    assert any("differs from the plan" in p for p in rep.problems)
    assert any("rows flagged" in p for p in rep.problems)
    assert rep.slices["ru"].flagged["too_fast"] == 4


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CollectionError, match="missing"):
        check_delivery(tmp_path, plan(), taxonomy())


def test_delivery_fetch_contract() -> None:
    FetchSpec(type="delivery", include=["*"])
    with pytest.raises(ValidationError):
        FetchSpec(type="delivery", repo_id="org/x", include=["*"])
    with pytest.raises(ValidationError):
        FetchSpec(type="hf_dataset", include=["*"])
    fetch = {"type": "delivery", "include": ["*"]}
    with pytest.raises(ValidationError, match="content sha256"):
        entry("commissioned_kz_2026", licence="commissioned-assignment", fetch=fetch)
    e = entry(
        "commissioned_kz_2026", licence="commissioned-assignment", fetch=fetch, revision="a" * 64
    )
    assert e.allowed_use.value == "train"
