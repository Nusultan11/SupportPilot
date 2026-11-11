"""Converters (exact values, lineage, rejections) and the conversion orchestrator
(idempotency, determinism, quality gate, crash safety, policy-checked loading)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from supportpilot.common.manifest import read_model_json
from supportpilot.common.schemas import RecordsManifest, SourceEntry, SourceRegistry
from supportpilot.data import converters
from supportpilot.data.convert import RECORDS, REJECTED, convert_source, records_dir
from supportpilot.data.converters import ConversionError, Converter
from supportpilot.data.policy import PolicyError, Purpose
from supportpilot.data.records import scan_records
from tests.data_helpers import entry, make_raw

RETAIL = (
    "instruction,intent,category,tags,response\n"
    '"I need to return  {{Product}}",return_product,RETURNS,BLQ,'
    '"Sure.\nStep 1: go to {{Website}}"\n'
    "where is my parcel?,track_order,ORDER,BI,Let me check.\n"
    '"  Тапсырысым қайда?  ",track_order,ORDER,,\n'
)
SUPPORT = "flags,instruction,category,intent,response\nB,cancel order 1,ORDER,cancel_order,Ok\n"


def _rows(data_dir: Path, e: SourceEntry) -> list[dict[str, object]]:
    return pq.read_table(records_dir(data_dir, e) / RECORDS).to_pylist()  # type: ignore[no-any-return]


def _convert(data_dir: Path, e: SourceEntry, **kw: object) -> object:
    return convert_source(e, data_dir, max_reject_ratio=kw.pop("ratio", 0.5), **kw)  # type: ignore[arg-type]


@pytest.fixture
def retail(tmp_path: Path) -> SourceEntry:
    e = entry("bitext_retail_ecom")
    make_raw(tmp_path, e, {"retail.csv": RETAIL, "README.md": "card"})
    return e


def test_bitext_values_are_copied_exactly(tmp_path: Path, retail: SourceEntry) -> None:
    result = _convert(tmp_path, retail)
    assert result.rows == 3  # type: ignore[attr-defined]
    rows = _rows(tmp_path, retail)
    first = rows[0]
    assert first["record_id"] == "bitext_retail_ecom:csv.0"
    assert first["text"] == "I need to return  {{Product}}"  # double space + placeholder kept
    assert first["response"] == "Sure.\nStep 1: go to {{Website}}"  # multi-line field intact
    assert (first["source_intent"], first["source_category"], first["source_tags"]) == (
        "return_product",
        "RETURNS",
        "BLQ",
    )
    assert first["source_file"] == "retail.csv"
    assert first["source_row"] == 0
    assert first["source_split"] is None  # Bitext has no splits; none invented
    assert first["lang"] == "en"  # declared language of the source
    assert rows[2]["text"] == "  Тапсырысым қайда?  "  # untouched, including spaces
    assert rows[2]["source_tags"] is None  # empty cell -> None, not ""
    assert rows[2]["response"] is None


def test_bitext_flags_column_and_lineage(tmp_path: Path) -> None:
    e = entry("bitext_customer_support")
    make_raw(tmp_path, e, {"cs.csv": SUPPORT})
    _convert(tmp_path, e)
    row = _rows(tmp_path, e)[0]
    assert row["source_tags"] == "B"
    m = read_model_json(records_dir(tmp_path, e) / "manifest.json", RecordsManifest)
    assert m.converter == "bitext_csv"
    assert m.allowed_use.value == "train"
    assert m.counts["lang=en"] == 1


def test_bitext_missing_column_is_a_structural_error(tmp_path: Path) -> None:
    e = entry("bitext_retail_ecom")
    make_raw(tmp_path, e, {"bad.csv": "instruction,intent\nhi,x\n"})
    with pytest.raises(ConversionError, match="missing columns"):
        _convert(tmp_path, e)
    assert not records_dir(tmp_path, e).exists()
    assert not list((tmp_path / "interim" / "records" / e.source_id).glob(".partial-*"))


def test_rejected_rows_are_logged_with_reasons(tmp_path: Path) -> None:
    e = entry("bitext_retail_ecom")
    body = "instruction,intent,category,tags,response\n" + "ok,a,B,C,D\n" * 9 + '"   ",a,B,C,D\n'
    make_raw(tmp_path, e, {"r.csv": body})
    result = _convert(tmp_path, e, ratio=0.2)
    assert (result.rows, result.rejected) == (9, 1)  # type: ignore[attr-defined]
    rejected = [
        json.loads(x)
        for x in (records_dir(tmp_path, e) / REJECTED).read_text(encoding="utf-8").splitlines()
    ]
    assert rejected == [
        {"source_file": "r.csv", "source_row": 9, "reason": "empty_text", "snippet": ""}
    ]
    m = read_model_json(records_dir(tmp_path, e) / "manifest.json", RecordsManifest)
    assert m.rejected_reasons == {"empty_text": 1}


def test_reject_ratio_gate_fails_conversion(tmp_path: Path) -> None:
    e = entry("bitext_retail_ecom")
    body = "instruction,intent,category,tags,response\nok,a,B,C,D\n" + '"",a,B,C,D\n' * 3
    make_raw(tmp_path, e, {"r.csv": body})
    with pytest.raises(ConversionError, match="rows rejected"):
        _convert(tmp_path, e, ratio=0.01)
    assert not records_dir(tmp_path, e).exists()


def test_whileai_dialogues(tmp_path: Path) -> None:
    e = entry("whileai_ecommerce_intent")
    lines = [
        {"conv_id": "c1", "messages": [{"seq": 0, "role": "assistant", "content": "Hi!"},
                                       {"seq": 1, "role": "user", "content": "refund pls"},
                                       {"seq": 2, "role": "assistant", "content": "Which order?"}],
         "target": {"core_type": "reverse", "details": {"action": "refund"}}},
        {"messages": [{"seq": 0, "role": "user", "content": "just looking"}],
         "target": {"core_type": "none"}},
    ]  # fmt: skip
    train = "\n".join(json.dumps(x) for x in lines) + "\n{broken json\n"
    make_raw(tmp_path, e, {"train.jsonl": train})
    result = _convert(tmp_path, e, ratio=0.5)
    assert (result.rows, result.rejected) == (2, 1)  # type: ignore[attr-defined]
    a, b = _rows(tmp_path, e)
    assert a["text"] == "refund pls"  # first *user* turn, not the opening assistant turn
    assert a["response"] == "Which order?"
    assert a["source_intent"] == "reverse"
    assert a["source_split"] == "train"
    assert json.loads(str(a["extra_json"]))["conv_id"] == "c1"
    assert b["response"] is None  # no assistant reply -> None
    assert a["record_type"] == "dialogue"


def test_belebele_language_mapping(tmp_path: Path) -> None:
    e = entry("belebele", allowed_use="eval_only", languages=["kk", "ru", "en"])
    q = {"question": " Сұрақ?", "flores_passage": "Мәтін", "correct_answer_num": "2",
         **{f"mc_answer{i}": f"a{i}" for i in range(1, 5)}}  # fmt: skip
    make_raw(tmp_path, e, {"data/kaz_Cyrl.jsonl": json.dumps(q, ensure_ascii=False) + "\n"})
    _convert(tmp_path, e)
    row = _rows(tmp_path, e)[0]
    assert (row["lang"], row["text"], row["record_type"]) == (
        "kk",
        " Сұрақ?",
        "reading_comprehension",
    )
    assert row["record_id"] == "belebele:kaz_Cyrl.0"
    assert json.loads(str(row["extra_json"]))["correct_answer_num"] == "2"
    assert row["source_intent"] is None


def test_belebele_unmapped_language_fails(tmp_path: Path) -> None:
    e = entry("belebele", allowed_use="eval_only")
    make_raw(tmp_path, e, {"data/deu_Latn.jsonl": "{}\n"})
    with pytest.raises(ConversionError, match="not mapped"):
        _convert(tmp_path, e)


def test_support_json_ru_parquet(tmp_path: Path) -> None:
    e = entry("support_json_ru", allowed_use="reference_only", languages=["ru"])
    table = pa.Table.from_pylist(
        [{"id": "x1", "split": "train", "message": "Хочу вернуть оплату", "category": "refund",
          "reply": "Возврат возможен.", "action": "propose_action", "human_escalation": False}]
    )  # fmt: skip
    raw = make_raw(tmp_path, e, {"README.md": "card"})
    (raw / "data" / "selected").mkdir(parents=True)
    pq.write_table(table, raw / "data" / "selected" / "train.parquet")
    make_raw(tmp_path, e, {})  # refresh the manifest to include the parquet file
    _convert(tmp_path, e)
    row = _rows(tmp_path, e)[0]
    assert (row["text"], row["response"], row["source_category"]) == (
        "Хочу вернуть оплату",
        "Возврат возможен.",
        "refund",
    )
    assert row["source_intent"] is None  # the source has categories, not intents
    extra = json.loads(str(row["extra_json"]))
    assert extra["action"] == "propose_action"
    assert extra["human_escalation"] is False


def test_rerun_skips_and_force_rebuild_is_byte_identical(
    tmp_path: Path, retail: SourceEntry
) -> None:
    first = _convert(tmp_path, retail)
    out = records_dir(tmp_path, retail)
    before = (out / RECORDS).read_bytes()
    m1 = read_model_json(out / "manifest.json", RecordsManifest)
    assert _convert(tmp_path, retail).status == "skipped"  # type: ignore[attr-defined]
    _convert(tmp_path, retail, force=True)
    m2 = read_model_json(out / "manifest.json", RecordsManifest)
    assert (out / RECORDS).read_bytes() == before
    assert m1.content_sha256 == m2.content_sha256
    assert first.status == "converted"  # type: ignore[attr-defined]


def test_converter_version_bump_triggers_rebuild(
    tmp_path: Path, retail: SourceEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _convert(tmp_path, retail)
    bumped = Converter("bitext_csv", 2, converters.convert_bitext)
    monkeypatch.setitem(converters.CONVERTERS, "bitext_retail_ecom", bumped)
    assert _convert(tmp_path, retail).status == "converted"  # type: ignore[attr-defined]


def test_tampered_raw_data_blocks_conversion(tmp_path: Path, retail: SourceEntry) -> None:
    raw = tmp_path / "raw" / retail.source_id / retail.revision
    with (raw / "retail.csv").open("a", encoding="utf-8") as fh:
        fh.write("extra,row,added,later,!\n")
    with pytest.raises(ConversionError, match="fails verification"):
        _convert(tmp_path, retail)


def test_crash_mid_conversion_keeps_previous_output(
    tmp_path: Path, retail: SourceEntry, monkeypatch: pytest.MonkeyPatch
) -> None:
    _convert(tmp_path, retail)
    good = (records_dir(tmp_path, retail) / RECORDS).read_bytes()

    def exploding(raw: Path, e: SourceEntry) -> Iterator[converters.Row]:
        yield from list(converters.convert_bitext(raw, e))[:1]
        raise RuntimeError("disk full")

    monkeypatch.setitem(
        converters.CONVERTERS, "bitext_retail_ecom", Converter("bitext_csv", 1, exploding)
    )
    with pytest.raises(RuntimeError, match="disk full"):
        _convert(tmp_path, retail, force=True)
    assert (records_dir(tmp_path, retail) / RECORDS).read_bytes() == good
    assert not list(records_dir(tmp_path, retail).parent.glob(".partial-*"))


def test_unknown_source_and_missing_raw(tmp_path: Path) -> None:
    with pytest.raises(ConversionError, match="no converter"):
        _convert(tmp_path, entry("not_a_known_source"))
    with pytest.raises(ConversionError, match="not acquired"):
        _convert(tmp_path, entry("bitext_retail_ecom"))


def test_loader_enforces_allowed_use(tmp_path: Path, retail: SourceEntry) -> None:
    bele = entry("belebele", allowed_use="eval_only", languages=["en"])
    q = {"question": "Q?", "flores_passage": "P", "correct_answer_num": "1",
         **{f"mc_answer{i}": "a" for i in range(1, 5)}}  # fmt: skip
    make_raw(tmp_path, bele, {"data/eng_Latn.jsonl": json.dumps(q) + "\n"})
    _convert(tmp_path, retail)
    _convert(tmp_path, bele)
    reg = SourceRegistry(sources=[retail, bele])
    assert (
        scan_records(reg, tmp_path, ["bitext_retail_ecom"], Purpose.TRAINING).collect().height == 3
    )
    assert (
        scan_records(reg, tmp_path, ["belebele"], Purpose.EVALUATION, verify=True).collect().height
        == 1
    )
    with pytest.raises(PolicyError, match="does not permit training"):
        scan_records(reg, tmp_path, ["bitext_retail_ecom", "belebele"], Purpose.TRAINING)
    with pytest.raises(ValueError, match="no sources"):
        scan_records(reg, tmp_path, [], Purpose.ANALYSIS)
