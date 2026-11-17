"""Commissioned delivery -> raw intake -> corpus build -> blind review -> scores (ADR-049)."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

import pytest
import yaml

from supportpilot.data.acquire import verify_raw
from supportpilot.data.collection import (
    CollectionError,
    delivery_content_sha256,
    intake_delivery,
    writers_by_example,
)
from supportpilot.data.corpus import build_corpus, load_examples, load_inputs
from supportpilot.data.review import (
    SHEET_COLUMNS,
    ReviewError,
    cohen_kappa,
    make_package,
    score_review,
    wilson,
)
from tests.collection_helpers import SOURCE, plan, taxonomy, write_delivery
from tests.data_helpers import entry
from tests.integration.test_corpus_build import RECIPE, project  # noqa: F401  (fixture)

FETCH = {"type": "delivery", "include": ["*"]}


def _entry(revision: str):  # type: ignore[no-untyped-def]
    return entry(
        SOURCE,
        licence="commissioned-assignment",
        languages=["ru", "kk"],
        fetch=FETCH,
        revision=revision,
        url="contract:test",
    )


def test_intake_requires_matching_revision_and_is_immutable(tmp_path: Path) -> None:
    p = plan()
    delivery = write_delivery(tmp_path / "delivery", p)
    with pytest.raises(CollectionError, match="registry revision"):
        intake_delivery(delivery, tmp_path / "data", _entry("b" * 64), p, taxonomy())
    e = _entry(delivery_content_sha256(delivery))
    rep = intake_delivery(delivery, tmp_path / "data", e, p, taxonomy())
    raw = tmp_path / "data" / "raw" / SOURCE / e.revision
    assert rep.ok
    assert verify_raw(raw) == []
    assert intake_delivery(delivery, tmp_path / "data", e, p, taxonomy()).ok  # reuse
    (raw / "rows.csv").write_text("tampered", encoding="utf-8")
    with pytest.raises(CollectionError, match="fails verification"):
        intake_delivery(delivery, tmp_path / "data", e, p, taxonomy())


def _fill(text: str, labels: dict[str, str]) -> str:
    rows = list(csv.DictReader(io.StringIO(text)))
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        r["intent"] = labels.get(r["example_id"], "")
        r["language_ok(y/n)"] = "y"
        r["natural_request(y/n)"] = "y"
        w.writerow(r)
    return buf.getvalue()


def test_commissioned_rows_build_split_by_scenario_and_review_scores(
    project: dict[str, Path],  # noqa: F811
) -> None:
    p = plan(
        review={
            "double_label_fraction": 0.5,
            "min_kappa": 0.6,
            "strata": [
                {"name": "ru_commissioned", "source": SOURCE, "slice": "ru", "size": 10},
                {"name": "en_existing", "lang": "en", "size": 10},
            ],
        }
    )
    delivery = write_delivery(project["root"] / "delivery", p)
    e = _entry(delivery_content_sha256(delivery))
    intake_delivery(delivery, project["data"], e, p, taxonomy())
    reg = yaml.safe_load(project["registry"].read_text("utf-8"))
    reg["sources"].append(e.model_dump(mode="json"))
    project["registry"].write_text(yaml.safe_dump(reg, allow_unicode=True), "utf-8")
    recipe: dict[str, Any] = {**RECIPE, "sources": dict(RECIPE["sources"])}
    recipe["sources"][SOURCE] = {
        "reader": "commissioned",
        "domain": "retail_support",
        "origin": {"ru": "human_written", "kk": "human_written"},
        "evidence": "test",
        "min_words": 2,
        "labels": {i: {"intent": i} for i in ("track_order", "cancel_order")},
    }
    project["recipe"].write_text(yaml.safe_dump(recipe, allow_unicode=True), "utf-8")

    inputs = load_inputs(project["recipe"], project["taxonomy"], project["registry"])
    res = build_corpus(project["data"], project["data"] / "corpus", inputs, "abc")
    assert all(v == 0 for v in res.leakage.values())
    mine = [x for x in load_examples(res.path) if x.source_id == SOURCE]
    assert len(mine) == 8
    assert {x.origin.value for x in mine} == {"human_written"}
    for intent in ("track_order", "cancel_order"):  # one brief each in the fixture
        scenario = [x for x in mine if x.intent == intent]
        # every paraphrase of one scenario, any slice or writer, is one split unit
        assert len({x.group_id for x in scenario}) == 1
        assert len({x.split for x in scenario}) == 1
    assert {x.source_split for x in mine} == {"ru", "kk", "ru_kk"}

    files = make_package(load_examples(res.path), p)
    assert "track_order" not in files["sheet_a.csv"]  # blind: no label shown
    pkg = project["root"] / "review"
    pkg.mkdir()
    key = list(csv.DictReader(io.StringIO(files["key.csv"])))
    truth = {r["example_id"]: r["intent"] for r in key}
    wrong = key[0]["example_id"]
    labels_a = {**truth, wrong: "complaint"}
    (pkg / "key.csv").write_text(files["key.csv"], "utf-8")
    (pkg / "sheet_a.csv").write_text(_fill(files["sheet_a.csv"], labels_a), "utf-8")
    (pkg / "sheet_b.csv").write_text(_fill(files["sheet_b.csv"], truth), "utf-8")
    raw = project["data"] / "raw" / SOURCE / e.revision
    writers = writers_by_example(raw, SOURCE)
    rep = score_review(pkg, "r01", "r02", p, frozenset(inputs.taxonomy.intents), writers)
    assert rep.problems == []
    total = sum(s.reviewed for s in rep.strata)
    assert total == len(key)
    assert len(rep.disagreements) == 1
    assert all(s.double_fraction >= 0.1 for s in rep.strata)

    own = score_review(pkg, "w01", "r02", p, frozenset(inputs.taxonomy.intents), writers)
    assert any("reviewed by its own writer w01" in x for x in own.problems)


def test_kappa_and_wilson_known_values() -> None:
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "b"]) == 1.0
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"]) == 0.0
    lo, hi = wilson(90, 100)
    assert (lo, hi) == (0.8256, 0.9448)


def test_package_can_be_limited_to_named_strata(project: dict[str, Path]) -> None:  # noqa: F811
    inputs = load_inputs(project["recipe"], project["taxonomy"], project["registry"])
    res = build_corpus(project["data"], project["data"] / "corpus", inputs, "abc")
    p = plan(
        review={
            "double_label_fraction": 0.2,
            "min_kappa": 0.6,
            "strata": [
                {"name": "en_existing", "lang": "en", "size": 10},
                {"name": "ru_commissioned", "source": SOURCE, "slice": "ru", "size": 10},
            ],
        }
    )
    examples = load_examples(res.path)
    with pytest.raises(ReviewError, match="no core rows"):
        make_package(examples, p)  # commissioned data not delivered yet
    files = make_package(examples, p, frozenset({"en_existing"}))
    assert {r["stratum"] for r in csv.DictReader(io.StringIO(files["key.csv"]))} == {"en_existing"}
    with pytest.raises(ReviewError, match="unknown strata"):
        make_package(examples, p, frozenset({"nope"}))
