"""Stage 3 end to end on small synthetic sources: clean -> dedup/group -> canonical/pending/
rejected -> gates, plus idempotency, determinism, crash safety, lineage and the CLI."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq
import pytest
import yaml

from supportpilot.cli import main
from supportpilot.common.config import CleanSettings, EdaSettings, Settings, find_project_root
from supportpilot.common.hashing import sha256_file
from supportpilot.common.schemas import SourceEntry
from supportpilot.data import build as build_mod
from supportpilot.data import gates as gates_mod
from supportpilot.data.build import BuildError, canonical_dir
from supportpilot.data.clean import clean_dir
from supportpilot.data.convert import RECORDS, convert_source, records_dir
from supportpilot.data.gates import Gate
from supportpilot.data.labels import load_intent_map
from supportpilot.data.neardup import build_shingles, exact_jaccard
from supportpilot.data.pipeline import prepare, regate
from supportpilot.data.placeholders import fill_text, load_catalog
from supportpilot.data.textnorm import sim_key
from tests.data_helpers import entry, make_raw

ROOT = find_project_root(Path(__file__).parent)
CATALOG = ROOT / "configs" / "placeholders.yaml"
IMAP = ROOT / "configs" / "intent_map.yaml"
LONG_A = (
    "hello there I would like some help because I need to close my account today "
    "if possible thank you"
)
LONG_B = LONG_A.replace("close", "edit")


def _csv(rows: list[tuple[str, str, str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["instruction", "intent", "category", "tags", "response"])
    for text, intent, resp in rows:
        w.writerow([text, intent, "X", "B", resp])
    return buf.getvalue()


def _cs_literal_twin() -> str:
    """A literal text equal to what row 6 of the CS file becomes after slot filling."""
    filled = fill_text(
        "switch to {{Account Type}} account", load_catalog(CATALOG), "bitext_customer_support:csv.6"
    )
    return filled.text


RETAIL_ROWS = [
    ("I want to cancel my order please", "cancel_order", "Sure, here is how."),  # 0
    ("i want to cancel my order please!", "cancel_order", "Sure."),  # 1 exact dup of 0
    ("I want to cancel my fucking order please", "cancel_order", "Sure."),  # 2 same family
    ("can i return a product online", "return_policy", "See {{Return Policy}}."),  # 3 conflict
    ("can I return a fucking product online?", "return_product_online", "Ok."),  # 4 conflict
    (LONG_A, "close_account", "Ok."),  # 5 guard keeps apart
    (LONG_B, "change_account", "Ok."),  # 6
    ("the app does not work on my phone", "technical_issue", "Sorry."),  # 7 unmapped
    (
        "where can I find information about a product in store",
        "return_product_in_store",
        "x",
    ),  # 8 suspect
    ("how do I return an item in store", "return_product_in_store", "Steps."),  # 9 mapped
]
CS_ROWS = [
    ("cancel order {{Order Number}}", "cancel_order", "Cancelling {{Order Number}}."),  # 0
    (
        "cancel order {{Order Number}}",
        "cancel_order",
        "Cancelling {{Order Number}}!",
    ),  # 1 dup (template)
    ("I want to cancel purchase {{Order Number}} now", "cancel_order", "Ok {{Order Number}}."),  # 2
    ("cancel my order", "cancel_order", "Which order?"),  # 3 different action -> other scenario
    ("talk to {{Store Manager}}", "contact_human_agent", "Ok."),  # 4 unknown placeholder
    (
        "i want a refund of {{Currency Symbol}}{{Refund Amount}}",
        "get_refund",
        "Refund ${{Refund Amount}}.",
    ),  # 5
    ("switch to {{Account Type}} account", "switch_account", "Use {{Account Type}}."),  # 6
    ("PLACEHOLDER_LITERAL", "switch_account", "Ok."),  # 7 literal twin of 6
    ("mail me at jane.doe2004@gmail.com about my invoice #12588", "get_invoice", "Ok."),  # 8 PII
]


def _dialogue(i: int, user: str, extra_turn: str = "fine thanks") -> str:
    msgs = [
        {"role": "user", "content": user, "seq": 0},
        {"role": "assistant", "content": "How can I help?", "seq": 1},
        {"role": "user", "content": extra_turn, "seq": 2},
    ]
    return json.dumps({"conv_id": f"c{i}", "messages": msgs, "target": {"core_type": "none"}})


WHILEAI = (
    "\n".join(
        [
            _dialogue(0, "hi, I need to return a blender that arrived cracked"),
            _dialogue(
                1, "Hola, necesito ayuda con un pago que aparece dos veces en mi cuenta por favor"
            ),
            _dialogue(
                2, "hello", "payment 我找设施 контроль פתרון मदद help with the order please now"
            ),
            _dialogue(3, "hi, I need to return a blender that arrived cracked"),  # same first turn,
        ]
    )
    + "\n"
)  # but the dialogue is identical too -> exact duplicate


def _belebele(lang: str, question: str, passage: str) -> str:
    row = {
        "link": "https://example.org/a",
        "question_number": 1,
        "flores_passage": passage,
        "question": question,
        "mc_answer1": "a",
        "mc_answer2": "b",
        "mc_answer3": "c",
        "mc_answer4": "d",
        "correct_answer_num": "1",
        "dialect": lang,
    }
    return json.dumps(row, ensure_ascii=False) + "\n"


@pytest.fixture
def stage3(tmp_path: Path) -> tuple[Settings, list[SourceEntry]]:
    root = tmp_path / "repo"
    data = root / "data"
    retail = entry("bitext_retail_ecom", licence="cdla-sharing-1.0")
    cs = entry("bitext_customer_support", licence="cdla-sharing-1.0")
    wai = entry("whileai_ecommerce_intent", fetch={"repo_id": "o/w", "include": ["*.jsonl"]})
    bel = entry(
        "belebele",
        licence="cc-by-sa-4.0",
        allowed_use="eval_only",
        languages=["en", "ru", "kk"],
        fetch={"repo_id": "o/b", "include": ["data/*.jsonl"]},
    )
    cs_rows = [
        (t if t != "PLACEHOLDER_LITERAL" else _cs_literal_twin(), i, r) for t, i, r in CS_ROWS
    ]
    make_raw(data, retail, {"retail.csv": _csv(RETAIL_ROWS)})
    make_raw(data, cs, {"cs.csv": _csv(cs_rows)})
    make_raw(data, wai, {"train.jsonl": WHILEAI})
    make_raw(
        data,
        bel,
        {
            "data/eng_Latn.jsonl": _belebele(
                "eng_Latn",
                "Where did the travellers stay?",
                "The travellers stayed in a small hostel near the station.",
            ),
            "data/rus_Cyrl.jsonl": _belebele(
                "rus_Cyrl",
                "Где остановились путешественники?",
                "Путешественники остановились в маленьком хостеле у вокзала.",
            ),
            "data/kaz_Cyrl.jsonl": _belebele(
                "kaz_Cyrl",
                "Саяхатшылар қайда тоқтады?",
                "Саяхатшылар вокзал жанындағы шағын хостелде тоқтады.",
            ),
        },
    )
    entries = [retail, cs, wai, bel]
    for e in entries:
        convert_source(e, data, max_reject_ratio=0.5)
    registry = {
        "schema_version": "sources/1",
        "sources": [e.model_dump(mode="json") for e in entries],
    }
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "sources.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    settings = Settings(
        env="test",
        root=root,
        clean=CleanSettings(placeholders=CATALOG, intent_map=IMAP, minhash_perms=64, lsh_bands=16),
        eda=EdaSettings(review_random=3, review_flagged=3, review_annotations=root / "none.yaml"),
    )
    return settings, entries


def _run(settings: Settings, entries: list[SourceEntry], force: bool = False) -> Any:
    return prepare(settings, entries, build=True, force=force, code_commit="test")


def _out(settings: Settings) -> Path:
    return canonical_dir(settings.resolve(settings.paths.data_dir))


def _canonical(settings: Settings) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(f) for f in sorted((_out(settings) / "examples").glob("*/*.parquet"))]
    )


def _by_record(df: pl.DataFrame) -> dict[str, dict[str, Any]]:
    return {r["record_id"]: r for r in df.iter_rows(named=True)}


def test_every_record_lands_once_and_gates_pass(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    res = _run(settings, entries)
    assert res.gates is not None
    assert res.gates.passed, res.gates
    statuses = {g.name: g.status for g in res.gates.gates}
    assert statuses["splits_and_leakage"] == "not_applicable"
    assert all(s == "pass" for k, s in statuses.items() if k != "splits_and_leakage")
    can, pen, rej = (
        _canonical(settings),
        pl.read_parquet(_out(settings) / "pending.parquet"),
        pl.read_parquet(_out(settings) / "rejected.parquet"),
    )
    ids = can["record_id"].to_list() + pen["record_id"].to_list() + rej["record_id"].to_list()
    assert len(ids) == len(set(ids)) == len(RETAIL_ROWS) + len(CS_ROWS) + 4 + 3
    manifest = json.loads((_out(settings) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_version"].startswith("canonical-")
    assert manifest["row_counts"]["canonical"] == can.height


def test_routing_reasons(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    _run(settings, entries)
    pen = _by_record(pl.read_parquet(_out(settings) / "pending.parquet"))
    rej = _by_record(pl.read_parquet(_out(settings) / "rejected.parquet"))
    can = _by_record(_canonical(settings))
    r, c, w = (
        "bitext_retail_ecom:csv.",
        "bitext_customer_support:csv.",
        "whileai_ecommerce_intent:train.",
    )
    assert rej[f"{r}1"]["reason"] == "exact_duplicate"
    assert rej[f"{r}1"]["duplicate_of"] == can[f"{r}0"]["example_id"]
    assert rej[f"{c}1"]["reason"] == "exact_duplicate"  # same template, other synthetic number
    assert rej[f"{c}7"]["reason"] == "exact_duplicate"  # literal text equal to the filled row 6
    assert rej[f"{c}4"]["reason"] == "unknown_placeholder"
    assert pen[f"{r}3"]["reason"] == pen[f"{r}4"]["reason"] == "label_conflict"
    assert pen[f"{r}7"]["reason"] == "intent_unmapped"
    assert pen[f"{r}8"]["reason"] == "label_suspect"
    assert f"{r}9" in can
    assert rej[f"{w}1"]["reason"] == "unsupported_language"
    assert rej[f"{w}2"]["reason"] == "garbled_text"
    assert rej[f"{w}3"]["reason"] == "exact_duplicate"  # whole dialogue identical
    assert pen[f"{w}0"]["reason"] == "record_type:dialogue"
    assert all(
        p["reason"] == "allowed_use:eval_only" for k, p in pen.items() if k.startswith("belebele")
    )


def test_families_scenarios_and_label_guard(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    _run(settings, entries)
    can = _by_record(_canonical(settings))
    r, c = "bitext_retail_ecom:csv.", "bitext_customer_support:csv."
    # profanity variant: same family and scenario as the plain text
    assert can[f"{r}2"]["family_id"] == can[f"{r}0"]["family_id"]
    # near-identical texts with different labels stay apart (and the pair really is near-identical)
    sm = build_shingles([sim_key(LONG_A), sim_key(LONG_B)], 5)
    import numpy as np

    assert exact_jaccard(sm, np.array([[0, 1]]))[0] >= settings.clean.family_threshold
    assert can[f"{r}5"]["family_id"] != can[f"{r}6"]["family_id"]
    assert can[f"{r}5"]["scenario_id"] != can[f"{r}6"]["scenario_id"]
    # same request + same action -> same scenario; another action -> another scenario
    assert can[f"{c}0"]["action"] == can[f"{c}2"]["action"] == "call_tool"
    assert can[f"{c}3"]["action"] == "clarify"
    assert can[f"{c}0"]["scenario_id"] == can[f"{c}2"]["scenario_id"]
    assert can[f"{c}0"]["scenario_id"] != can[f"{c}3"]["scenario_id"]
    # belebele translations share a family via the source alignment
    pen = pl.read_parquet(_out(settings) / "pending.parquet").filter(
        pl.col("source_id") == "belebele"
    )
    assert pen["family_id"].n_unique() == 1
    assert set(pen["lang"].to_list()) == {"en", "ru", "kk"}


def test_cleaned_text_slots_pii_and_lineage(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    data = settings.resolve(settings.paths.data_dir)
    before = {e.source_id: sha256_file(records_dir(data, e) / RECORDS) for e in entries}
    _run(settings, entries)
    can = _by_record(_canonical(settings))
    c = "bitext_customer_support:csv."
    refund = can[f"{c}5"]
    slots = (
        dict(refund["slots"])
        if isinstance(refund["slots"], dict)
        else {kv["key"]: kv["value"] for kv in refund["slots"]}
    )
    assert slots["currency"] == "USD"
    assert slots["refund_amount"] in refund["text"]
    assert (
        f"${slots['refund_amount']}" in refund["reference_response"]
    )  # same value in the response
    assert "{{" not in refund["text"]
    pii = can[f"{c}8"]
    assert "jane.doe2004" not in pii["text"]
    assert "@example.com" in pii["text"]
    assert "pii:email" in pii["quality"]["flags"]
    assert pii["action"] == "call_tool"  # literal invoice reference
    # lineage: raw records untouched, every canonical row traces to a record
    assert before == {e.source_id: sha256_file(records_dir(data, e) / RECORDS) for e in entries}
    rec_ids = set()
    for e in entries:
        rec_ids |= set(
            pq.read_table(records_dir(data, e) / RECORDS, columns=["record_id"])
            .column(0)
            .to_pylist()
        )
    assert set(can) <= rec_ids
    assert all("labels:intent_map_draft_v0" in r["quality"]["flags"] for r in can.values())


def test_rerun_is_skipped_and_force_reproduces_identical_content(
    stage3: tuple[Settings, list[SourceEntry]],
) -> None:
    settings, entries = stage3
    first = _run(settings, entries)
    files = {
        p.relative_to(_out(settings)).as_posix(): sha256_file(p)
        for p in _out(settings).rglob("*.parquet")
    }
    stale = _out(settings).parent / ".partial-canonical-deadbeef"
    stale.mkdir()  # what a killed run leaves behind
    second = _run(settings, entries)
    assert second.build.status == "skipped"
    assert not stale.exists()
    assert all(c.status == "skipped" for c in second.cleaned)
    third = _run(settings, entries, force=True)
    assert third.build.status == "built"
    assert third.build.dataset_version.split("-")[-1] == first.build.dataset_version.split("-")[-1]
    assert files == {
        p.relative_to(_out(settings)).as_posix(): sha256_file(p)
        for p in _out(settings).rglob("*.parquet")
    }


def test_crash_during_build_keeps_previous_output(
    stage3: tuple[Settings, list[SourceEntry]], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, entries = stage3
    _run(settings, entries)
    manifest = (_out(settings) / "manifest.json").read_text(encoding="utf-8")

    def boom(*_a: Any, **_k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(build_mod, "_write", boom)
    with pytest.raises(KeyboardInterrupt):
        _run(settings, entries, force=True)
    assert (_out(settings) / "manifest.json").read_text(encoding="utf-8") == manifest
    assert not list(_out(settings).parent.glob(".partial-canonical-*"))
    monkeypatch.undo()
    assert _run(settings, entries, force=True).gates.passed


def test_failed_gate_keeps_previous_build_and_saves_failed_one(
    stage3: tuple[Settings, list[SourceEntry]], monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, entries = stage3
    _run(settings, entries)
    manifest = (_out(settings) / "manifest.json").read_text(encoding="utf-8")
    monkeypatch.setattr(gates_mod, "g_pii", lambda _b: Gate(name="pii", status="fail", metric=1))
    with pytest.raises(BuildError, match="quality gates failed"):
        _run(settings, entries, force=True)
    assert (_out(settings) / "manifest.json").read_text(encoding="utf-8") == manifest
    failed = _out(settings).parent / "canonical.failed"
    report = json.loads((failed / "quality_report.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    monkeypatch.undo()
    assert regate(settings, entries).passed  # the kept build is still valid


def test_clean_step_on_corrupt_records_fails(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    data = settings.resolve(settings.paths.data_dir)
    (records_dir(data, entries[0]) / RECORDS).write_bytes(b"corrupt")
    from supportpilot.data.converters import ConversionError

    with pytest.raises(ConversionError, match="verification"):
        prepare(settings, entries[:1], build=False, force=True, code_commit=None)
    assert not clean_dir(data, entries[0]).exists()


def test_build_needs_every_source_cleaned(stage3: tuple[Settings, list[SourceEntry]]) -> None:
    settings, entries = stage3
    prepare(settings, entries[:1], build=False, force=False, code_commit=None)
    with pytest.raises(BuildError, match="not cleaned"):
        build_mod.build_canonical(
            entries,
            settings.resolve(settings.paths.data_dir),
            settings.clean,
            load_intent_map(IMAP),
            {},
            code_commit=None,
        )


def test_cli_clean_gates_and_eda(
    stage3: tuple[Settings, list[SourceEntry]],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings, _ = stage3
    root = settings.root
    cfg = {
        "env": "test",
        "root": str(root),  # never the real project root
        "clean": {
            "placeholders": str(CATALOG),
            "intent_map": str(IMAP),
            "minhash_perms": 64,
            "lsh_bands": 16,
        },
        "eda": {
            "review_random": 2,
            "review_flagged": 2,
            "review_annotations": str(root / "none.yaml"),
        },
    }
    path = root / "configs" / "test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.chdir(root)
    assert main(["--config", str(path), "data", "clean", "--all"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gates"]["passed"] is True
    assert main(["--config", str(path), "data", "gates"]) == 0
    capsys.readouterr()
    assert main(["--config", str(path), "data", "eda", "--no-tokens"]) == 0
    summary = json.loads(capsys.readouterr().out)
    report = Path(summary["report"]).read_text(encoding="utf-8")
    assert "Stage 3 EDA" in report
    assert "Quality gates" in report
    profile = json.loads(Path(summary["profile"]).read_text(encoding="utf-8"))
    assert (
        profile["sources"]["bitext_customer_support"]["placeholders"]["text_inventory"][
            "Order Number"
        ]
        == 3
    )
    assert (root / "artifacts" / "reports" / "eda" / "review" / "bitext_retail_ecom.md").is_file()
    assert Path(summary["report"]).is_relative_to(root)
