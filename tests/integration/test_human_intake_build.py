"""A sealed commissioned delivery flows through the human corpus build (DATA_CONTRACTS §20.2)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from supportpilot.common.config import find_project_root
from supportpilot.data.human import BuildInputs, HumanRecipe, historical_sources, load_recipe
from supportpilot.data.humangates import build_human, verify_human
from supportpilot.data.humanintake import Delivery, LabelReview, seal_delivery
from supportpilot.data.taxonomy import load_taxonomy
from tests.data_helpers import entry
from tests.integration.test_human_extension import _encode, _lid
from tests.unit.test_human_intake import PEOPLE, _write, batch, lab, req

ROOT = find_project_root(Path(__file__).parent)

# Two needs, each written in several variants; a need's variants must share one family.
REQUESTS = [
    req("n1-ru", "Подскажите, сколько дней идёт доставка в Караганду?", scenario_id="need-1"),
    req("n1-kk", "Қарағандыға жеткізу неше күн алады?", "kk", scenario_id="need-1"),
    req("n1-mx", "Қарағандыға доставка сколько дней алады?", "mixed", scenario_id="need-1"),
    req("n2-ru", "Хочу поговорить с живым оператором, а не с ботом", scenario_id="need-2"),
    req(
        "n2-kk",
        "Маған тірі операторды қосыңызшы, ботпен сөйлескім келмейді",
        "kk",
        scenario_id="need-2",
    ),
    req("n3-ru", "Привет", scenario_id="need-3"),  # short but natural: no word minimum
    req("n4-ru", "Где мой заказ?", scenario_id="need-4"),  # reviewers disagree: excluded
]


def _reviews() -> list[LabelReview]:
    out = []
    labels = {
        "n1": ("delivery_time", "answer_kb", None),
        "n2": ("human_agent", "escalate", "customer_requested_human"),
        "n3": ("greeting", "answer_kb", None),
    }
    for r in REQUESTS[:-1]:
        intent, action, reason = labels[r.item_id[:2]]
        for who in ("p-rev1", "p-rev2"):
            out.append(
                lab(
                    r.item_id,
                    who,
                    intent,
                    variant=r.variant,
                    action=action,
                    escalation_reason=reason,
                )
            )
    out += [
        lab("n4-ru", "p-rev1", "track_order", action="clarify"),
        lab("n4-ru", "p-rev2", "order_history", action="answer_kb"),
    ]
    return out


def test_reviewed_delivery_enters_corpus_with_human_label_status(tmp_path: Path) -> None:
    tax = load_taxonomy(ROOT / "configs" / "taxonomy.yaml")
    full = load_recipe(ROOT / "configs" / "human_corpus.yaml", tax)
    data = full.model_dump()
    data["authorship"] = {
        "intake_pilot_01": {
            "origin": {"ru": "human_written", "kk": "human_written"},
            "active": True,
            "intake": True,
            "evidence": "Commissioned pilot: screened native authors, two blind reviewers.",
        }
    }
    data["sources"] = {}
    data["grounded"] = None
    data["split_ratios"] = {"train": 0.5, "validation": 0.25, "test": 0.25}
    recipe = HumanRecipe.model_validate(data)

    e = entry(
        "intake_pilot_01",
        fetch=None,
        status="candidate",
        languages=["ru", "kk"],
        pipelines=["human"],
    )
    root = tmp_path / "raw" / e.source_id / e.revision

    _write(root, Delivery(batch(), PEOPLE, REQUESTS, _reviews()))
    seal_delivery(root, e)
    kb = tmp_path / "kb" / "demo_store" / "versions" / batch().kb_version
    kb.mkdir(parents=True)
    pl.DataFrame({"chunk_id": ["delivery_policy:abcd1234:0000"]}).write_parquet(
        kb / "chunks.parquet"
    )

    inp = BuildInputs(
        entries=[e],
        data_dir=tmp_path,
        recipe=recipe,
        tax=tax,
        lid=_lid,
        encode=_encode,
        history_paths=historical_sources(tmp_path),
    )
    res = build_human(inp, code_commit="abc", recipe_sha256="r", taxonomy_sha256="t", previous=None)
    assert res["status"] == "built", res
    assert res["gates"] == "pass"
    out = Path(res["path"])
    active = pl.concat(
        [pl.read_parquet(out / f"{s}.parquet") for s in ("train", "validation", "test")]
    )
    assert active.height == 6
    assert set(active["label_status"]) == {"human_agreed"}
    assert set(active["review_status"]) == {"independently_reviewed"}
    mixed = active.filter(pl.col("record_id") == "intake_pilot_01:n1-mx").row(0, named=True)
    assert mixed["code_switch"]
    assert {"ru", "kk"} <= set(mixed["langs"])
    fam = active.group_by("family_id").agg(pl.col("record_id").count().alias("n"))
    assert sorted(fam["n"].to_list()) == [1, 2, 3]  # need-1 / need-2 variants stay together
    esc = active.filter(pl.col("intent") == "human_agent")
    assert set(esc["escalation_reason"]) == {"customer_requested_human"}

    exc = pl.read_parquet(out / "excluded.parquet")
    assert dict(zip(exc["record_id"], exc["reason"], strict=True)) == {
        "intake_pilot_01:n4-ru": "unresolved_disagreement"
    }
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["independently_reviewed_share"] == 1.0
    assert stats["mixed_ru_kk"] == 1
    assert stats["build"]["intake"]["intake_pilot_01"]["delivered"] == len(REQUESTS)
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert {"review_provenance", "labels"} <= {g["name"] for g in report["gates"]}
    assert verify_human(out)["checksums_ok"]
