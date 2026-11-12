"""Stage 4 end to end on a small synthetic Stage 3 build: select -> training KB -> translate
(fake MT, resumable cache) -> QE -> SFT (splits, leakage filter, gates) -> gold, plus re-runs,
determinism, crash leftovers and the CLI. No GPU and no model download: the MT backend and the
embedder are replaced by deterministic fakes; the real taxonomy, locale, authored scenarios,
templates, training KB and gold sources are used."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from supportpilot.cli import main
from supportpilot.common.config import PathsSettings, Settings, Stage4Settings, find_project_root
from supportpilot.common.ids import make_dataset_version, make_example_id, make_record_id
from supportpilot.common.manifest import write_model_json
from supportpilot.common.schemas import DatasetManifest, SourceEntry
from supportpilot.data import embed as embed_mod
from supportpilot.data import mtrun
from supportpilot.data.build import CANONICAL_SCHEMA, PENDING_SCHEMA, canonical_dir
from supportpilot.data.gold import build_gold
from supportpilot.data.locale import load_locale, localize_en
from supportpilot.data.scenarios import AUTHORED_SOURCE
from supportpilot.data.sftbuild import build_sft, sft_root
from supportpilot.data.stage4 import run_kb, run_native, run_select
from tests.data_helpers import entry

ROOT = find_project_root(Path(__file__).parent)
LOCALE = load_locale(ROOT / "configs" / "locale_kz.yaml")

# (source, label, text, slots, canonical?, ru, kk)
ROWS: list[tuple[str, str, str, dict[str, str], bool, str | None, str | None]] = [
    (
        "bitext_retail_ecom",
        "return_policy",
        "how many days do I have to return an item?",
        {},
        True,
        "Сколько дней у меня есть, чтобы вернуть товар?",
        "Тауарды қайтаруға менде неше күн бар?",
    ),
    (
        "bitext_retail_ecom",
        "customer_service",
        "i want to talk to customer service",
        {},
        False,
        "Я хочу поговорить со службой поддержки клиентов",
        "Мен қолдау қызметімен сөйлескім келеді",
    ),
    (
        "bitext_retail_ecom",
        "customer_service",
        "how can I call customer service?",
        {},
        False,
        "Как позвонить в службу поддержки?",
        "Қолдау қызметіне қалай қоңырау шалуға болады?",
    ),
    (
        "bitext_retail_ecom",
        "delivery_time",
        "can you tell me how long delivery takes?",
        {},
        True,
        "Скажите, пожалуйста, сколько длится доставка?",
        "Жеткізу қанша уақыт алатынын айтыңызшы?",
    ),
    (
        "bitext_retail_ecom",
        "cancel_order",
        "I want to cancel my order",
        {},
        True,
        "Я хочу отменить свой заказ",
        "Мен тапсырысымды болдырмағым келеді",
    ),
    (
        "bitext_customer_support",
        "track_order",
        "where is my order 482913 now?",
        {"order_id": "482913"},
        True,
        "Подскажите, где находится заказ 482913?",
        "Менің 482913 тапсырысым қазір қайда?",
    ),
    (
        "bitext_customer_support",
        "check_cancellation_fee",
        "check the cancellation fee",
        {},
        False,
        None,
        None,
    ),
    (
        "bitext_customer_support",
        "change_shipping_address",
        "I need to change my shipping address",
        {},
        False,
        "Мне нужно изменить адрес доставки",
        "Жеткізу мекенжайымды өзгертуім керек",
    ),
    (
        "bitext_customer_support",
        "check_payment_methods",
        "what payment methods do you accept?",
        {},
        True,
        "Какие способы оплаты вы принимаете?",
        "Қандай төлем тәсілдерін қабылдайсыздар?",
    ),
    (
        "bitext_customer_support",
        "get_refund",
        "refund of $45 for order 382910 please",
        {"order_id": "382910"},
        True,
        "Верните, пожалуйста, 22 500 тенге за заказ 382910",
        "382910 тапсырысы үшін 22 500 теңгені қайтарыңызшы",
    ),
]

TRANSLATIONS: dict[tuple[str, str], str] = {}
ALIAS: dict[str, str] = {}
for _src, _lab, _text, _slots, _can, _ru, _kk in ROWS:
    _loc = localize_en(_text, LOCALE).text
    for _lang, _t in (("ru", _ru), ("kk", _kk)):
        if _t:
            TRANSLATIONS[(_lang, _loc)] = _t
            ALIAS[_t] = _loc


class FakeTranslator:
    calls: ClassVar[list[int]] = []

    def __init__(self, spec: Any) -> None:
        self.spec = spec

    def __call__(self, texts: Sequence[str], src: str, tgt: str) -> list[str]:
        FakeTranslator.calls.append(len(texts))
        if tgt == "en":
            return [ALIAS.get(t, t) for t in texts]
        return [TRANSLATIONS.get((tgt, t), t) for t in texts]

    def close(self) -> None:
        pass


class FakeEmbedder:
    """Char-trigram hashing; a translation embeds like its source (ALIAS)."""

    def __init__(self, spec: Any) -> None:
        self.spec = spec

    def encode(self, texts: Sequence[str]) -> np.ndarray[Any, Any]:
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, t in enumerate(texts):
            t = ALIAS.get(t, t).lower()
            for j in range(len(t) - 2):
                h = int(hashlib.sha256(t[j : j + 3].encode()).hexdigest()[:8], 16) % 256
                out[i, h] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


def _entries() -> list[SourceEntry]:
    return [
        entry("bitext_retail_ecom", licence="cdla-sharing-1.0"),
        entry("bitext_customer_support", licence="cdla-sharing-1.0"),
        entry("belebele", licence="cc-by-sa-4.0", allowed_use="eval_only"),
    ]


def _quality() -> dict[str, Any]:
    return {"lid_conf": 0.9, "mt_qe": None, "intent_agree": None, "dup_cluster": None, "flags": []}


def _make_canonical(data_dir: Path) -> None:
    out = canonical_dir(data_dir)
    can, pend = [], []
    for i, (src, label, text, slots, is_can, _ru, _kk) in enumerate(ROWS):
        rid = make_record_id(src, f"csv.{i}")
        row = {
            "example_id": make_example_id(src, rid, "en", "original", text),
            "record_id": rid,
            "source_id": src,
            "scenario_id": "scn_" + hashlib.sha256(f"{src}|{label}".encode()).hexdigest()[:16],
            "family_id": "fam_" + hashlib.sha256(rid.encode()).hexdigest()[:16],
            "lang": "en",
            "code_switch": False,
            "text": text,
            "slots": list(slots.items()),
            "source_intent": label,
            "quality": _quality(),
        }
        if is_can:
            can.append(
                {
                    **row,
                    "schema_version": "canonical/1",
                    "intent": "return_policy",
                    "action": "answer_kb",
                    "escalate": False,
                    "origin": "original",
                }
            )
        else:
            pend.append(
                {
                    **row,
                    "schema_version": "pending/1",
                    "revision": "r",
                    "record_type": "single_turn",
                    "allowed_use": "train",
                    "reason": "intent_unmapped",
                    "langs": ["en"],
                }
            )
    (out / "examples" / "lang=en").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(can, schema=CANONICAL_SCHEMA),
        out / "examples" / "lang=en" / "part-00000.parquet",
    )
    pq.write_table(pa.Table.from_pylist(pend, schema=PENDING_SCHEMA), out / "pending.parquet")
    write_model_json(
        out / "manifest.json",
        DatasetManifest(
            dataset_version=make_dataset_version("canonical", datetime.now(UTC).date(), "a" * 64),
            data_schema_version="canonical/1",
            code_commit="test",
            inputs=["test"],
            row_counts={"canonical": len(can)},
            files=[{"path": "pending.parquet", "bytes": 1, "sha256": "a" * 64}],
            created_at=datetime.now(UTC),
        ),
    )


@pytest.fixture
def s4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setattr(mtrun, "HFTranslator", FakeTranslator)
    monkeypatch.setattr(mtrun, "Embedder", FakeEmbedder)
    monkeypatch.setattr(embed_mod, "Embedder", FakeEmbedder)
    base = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))["stage4"]
    for key in ("taxonomy", "locale", "authored_dir", "demo_facts"):
        base[key] = ROOT / base[key]
    base["native_sources"] = []  # This fixture deliberately contains only Bitext.
    data_dir = tmp_path / "data"
    _make_canonical(data_dir)
    return Settings(
        env="test",
        root=tmp_path,
        paths=PathsSettings(
            data_dir=data_dir, artifacts_dir=tmp_path / "art", reports_dir=tmp_path / "art" / "rep"
        ),
        stage4=Stage4Settings.model_validate(base),
    )


@pytest.mark.slow
def test_stage4_end_to_end(s4: Settings) -> None:
    entries = _entries()
    data_dir = s4.resolve(s4.paths.data_dir)
    sel = run_select(s4, entries, code_commit="t", force=False)
    assert sel["status"] == "built"
    rows = sel["rows"]
    assert rows["excluded|reason=excluded"] == 1  # check_cancellation_fee
    bb = pl.read_parquet(data_dir / "interim" / "stage4" / "selected" / "backbone.parquet")
    by_text = {r["text"]: r for r in bb.iter_rows(named=True)}
    assert by_text["how can I call customer service?"]["intent"] == "contact_info"
    assert by_text["i want to talk to customer service"]["intent"] == "human_agent"
    assert by_text["where is my order 482913 now?"]["action"] == "call_tool"
    assert by_text["I want to cancel my order"]["action"] == "clarify"
    assert by_text["I need to change my shipping address"]["intent"] == "change_address"
    assert run_select(s4, entries, code_commit="t", force=False)["status"] == "skipped"

    assert run_kb(s4, code_commit="t", force=False)["status"] == "built"
    assert run_native(s4, entries, code_commit="t", force=False)["status"] == "built"

    # translation is resumable: a limited run, then the rest from the cache
    first = mtrun.run_translate(s4, "kk", limit=4, back=False)
    assert first[0].translated == 4
    rest = mtrun.run_translate(s4, "kk")
    assert rest[0].cached == 4
    assert rest[0].translated == len(ROWS) - 1 - 4
    mtrun.run_translate(s4, "ru")
    again = mtrun.run_translate(s4, "ru")
    assert all(r.translated == 0 for r in again)
    score = mtrun.run_score(s4, code_commit="t")
    assert score["status"] == "built"
    tr = pl.read_parquet(data_dir / "interim" / "stage4" / "translated" / "translations.parquet")
    assert tr.height == 2 * (len(ROWS) - 1)
    assert tr["accept"].all(), (
        tr.filter(~pl.col("accept")).select("lang", "text", "hard").to_dicts()
    )
    assert mtrun.run_score(s4, code_commit="t")["status"] == "skipped"

    stale = sft_root(data_dir) / ".partial-current-dead"
    stale.mkdir(parents=True)
    res = build_sft(s4, entries, code_commit="t")
    assert res["status"] == "built"
    assert res["gates_passed"], json.loads(
        (Path(res["path"]) / "quality_report.json").read_text("utf-8")
    )
    assert not stale.exists()
    out = Path(res["path"])
    report = json.loads((out / "quality_report.json").read_text(encoding="utf-8"))
    assert {g["name"] for g in report["gates"]} >= {
        "schema",
        "language",
        "grounding",
        "leakage",
        "licence_provenance",
    }
    df = pl.concat(
        [pl.read_parquet(out / f"{sp}.parquet") for sp in ("train", "validation", "test")]
    )
    held = df.filter(pl.col("split") != "train")
    assert held.height > 0
    assert set(held["source_id"].to_list()) == {AUTHORED_SOURCE}
    train_scn = set(df.filter(pl.col("split") == "train")["scenario_id"].to_list())
    assert not train_scn & set(held["scenario_id"].to_list())
    mt = df.filter(pl.col("origin") == "machine_translated")
    assert mt.height > 0
    assert mt["translation_of"].is_not_null().all()
    assert set(df["lang"].to_list()) == {"en", "ru", "kk"}
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["train"]["lang"]["kk"] > 0

    # re-run is skipped; a forced rebuild reproduces the same content (dataset_version)
    assert build_sft(s4, entries, code_commit="t")["status"] == "skipped"
    forced = build_sft(s4, entries, code_commit="t", force=True)
    assert forced["dataset_version"] == res["dataset_version"]

    gold = build_gold(s4, code_commit="t")
    assert gold["passed"], gold
    assert gold["per_lang"] == {
        "en": gold["scenarios"],
        "ru": gold["scenarios"],
        "kk": gold["scenarios"],
    }
    assert gold["review_status"] == {"draft": gold["items"]}
    lines = (Path(gold["path"]) / "gold.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == gold["items"]


def test_cli_taxonomy_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["data", "taxonomy"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intents"] == 35


def test_native_cache_tracks_source_language_policy(s4: Settings) -> None:
    from supportpilot.data.native import native_dir
    from supportpilot.data.policy import PolicyError

    cfg = s4.stage4.model_copy(
        update={
            "native_sources": ["bitext_retail_ecom"],
            "native_max_per_label": {"bitext_retail_ecom": 1},
        }
    )
    settings = s4.model_copy(update={"stage4": cfg})
    entries = _entries()
    first = run_native(settings, entries, code_commit="t", force=False)
    assert first["rows"]["kept"] == 1
    assert run_native(settings, entries, code_commit="t", force=False)["status"] == "skipped"
    changed = [
        e.model_copy(update={"languages": ["ru"]}) if e.source_id == "bitext_retail_ecom" else e
        for e in entries
    ]
    second = run_native(settings, changed, code_commit="t", force=False)
    assert second["status"] == "built"
    assert second["rows"]["kept"] == 0
    assert second["dataset_version"] != first["dataset_version"]
    out = native_dir(settings.resolve(settings.paths.data_dir))
    assert pl.read_parquet(out / "excluded.parquet")["reason"].to_list() == ["language"] * 2
    from supportpilot.common.schemas import AllowedUse

    blocked = [
        e.model_copy(update={"allowed_use": AllowedUse.EVAL_ONLY})
        if e.source_id == "bitext_retail_ecom"
        else e
        for e in entries
    ]
    with pytest.raises(PolicyError):
        run_native(settings, blocked, code_commit="t", force=False)
