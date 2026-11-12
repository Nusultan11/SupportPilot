"""Regression checks for expanded sources, protected evaluation and retrieval qrels."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_chunk_id
from supportpilot.common.schemas import KBChunk, LanguageCode, RawRecord, Split
from supportpilot.data.converters import ConversionError, convert_mangi, convert_massive
from supportpilot.data.locale import load_locale
from supportpilot.data.retrievalset import QueryFile, QuerySource, resolve_queries
from supportpilot.data.scenarios import fill_values
from supportpilot.data.sftbuild import _leak_filter
from supportpilot.data.split import assign_authored
from supportpilot.data.steps import StepError
from supportpilot.data.taxonomy import load_taxonomy, relabel
from tests.data_helpers import entry

ROOT = Path(__file__).resolve().parents[2]


def test_move_between_cities_never_renders_the_same_city() -> None:
    locale = load_locale(ROOT / "configs/locale_kz.yaml")
    for lang in ("en", "ru", "kk"):
        for seed in range(100):
            vals = fill_values(locale, lang, "change_address_city_change", seed)
            assert vals["city"] != vals["city_from"]
            assert vals["city_in"] != vals["city_from_in"]


def test_split_covers_each_intent_and_keeps_single_language_in_train() -> None:
    sources = [
        (f"{intent}-{i}", intent, "clarify", True)
        for intent in ("greeting", "unclear", "change_address")
        for i in range(4)
    ]
    sources.append(("mixed_only", "greeting", "clarify", False))
    result = assign_authored(sources, {"validation": 0.15, "test": 0.15}, 7)
    assert result == assign_authored(list(reversed(sources)), {"validation": 0.15, "test": 0.15}, 7)
    assert result["mixed_only"] == Split.TRAIN
    for intent in ("greeting", "unclear", "change_address"):
        assert {result[f"{intent}-{i}"] for i in range(4)} == set(Split)


def test_evaluation_examples_survive_copy_filter(settings: Settings) -> None:
    def row(key: str, text: str, split: str) -> dict[str, Any]:
        return {"scenario_id": key, "family_id": key, "user_text": text, "split": split}

    held = row("held", "Explain how to remove an item from the shopping basket", "test")
    copied = row("source", held["user_text"], "train")
    unrelated = row("other", "A bank mortgage is outside the online shop domain", "train")
    kept, report = _leak_filter([copied, unrelated, held], settings, None)
    assert kept == [unrelated, held]
    assert report["removed_examples"] == {"train": 1}
    assert report["policy"] == "preserve_evaluation_remove_training_copies"


def test_mangi_missing_domain_fails_instead_of_guessing(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    rec = {"id": "k1", "domain": "banking", "subdomain": "loans", "question": "Несие алу керек"}
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf8")
    result = list(convert_mangi(tmp_path, entry("mangi_kk_support_qa", languages=["kk"])))
    assert len(result) == 1
    assert isinstance(result[0], RawRecord)
    assert result[0].lang == "kk"
    assert result[0].source_intent == "banking.loans"
    del rec["domain"]
    path.write_text(json.dumps(rec), encoding="utf8")
    with pytest.raises(ConversionError, match="missing fields"):
        list(convert_mangi(tmp_path, entry("mangi_kk_support_qa")))


def test_massive_uses_class_metadata_and_keeps_partition(tmp_path: Path) -> None:
    path = tmp_path / "ru-RU" / "test" / "0000.parquet"
    path.parent.mkdir(parents=True)
    table = pa.Table.from_pylist(
        [
            {
                "id": "7",
                "locale": "ru-RU",
                "partition": "test",
                "scenario": 0,
                "intent": 0,
                "utt": "включи будильник утром",
            }
        ]
    )
    metadata = {
        b"huggingface": json.dumps(
            {
                "info": {
                    "features": {
                        "intent": {"names": ["alarm_set"]},
                        "scenario": {"names": ["alarm"]},
                    }
                }
            }
        ).encode()
    }
    pq.write_table(table.replace_schema_metadata(metadata), path)
    result = list(convert_massive(tmp_path, entry("massive_en_ru")))
    assert len(result) == 1
    assert isinstance(result[0], RawRecord)
    assert result[0].source_intent == "alarm_set"
    assert result[0].lang == "ru"
    assert result[0].source_split == "test"
    pq.write_table(table, path)
    with pytest.raises(ConversionError, match="metadata"):
        list(convert_massive(tmp_path, entry("massive_en_ru")))


@pytest.mark.parametrize(
    ("label", "text", "allowed"),
    [
        ("banking.loans", "Неге бас тартылды?", False),
        ("banking.loans", "Несиеге өтініш беру үшін не қажет?", True),
        ("banking.cards", "Картаның мерзімі бітсе не болады?", False),
        ("banking.cards", "Банкомат картамды жұтып қойды.", True),
        ("telecom.internet", "Роутерді өзім баптай аламын ба?", False),
        ("telecom.internet", "Үй интернетіне қанша мерзімге келісімшарт жасалады?", True),
        ("telecom.terminology", "Dual SIM телефон деген не?", False),
    ],
)
def test_external_domain_needs_a_cue_in_the_question(label: str, text: str, allowed: bool) -> None:
    tax = load_taxonomy(ROOT / "configs/taxonomy.yaml")
    result = relabel(tax, "mangi_kk_support_qa", label, text, {})
    assert (result.status == "mapped") is allowed


def _source(**updates: Any) -> dict[str, Any]:
    return dict(
        id="example",
        category="simple",
        en="A distinct original question",
        ru="Отдельный оригинальный вопрос",
        kk="Бөлек жазылған жаңа сұрақ",
        relevant=[{"section": "store_info.contact", "grade": 3}],
        rationale="Direct support in the explicitly named section",
        **updates,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"category": "unanswerable"},
        {"relevant": [{"section": "store_info.contact", "grade": 0}]},
        {"category": "multi_fragment"},
        {"relevant": [{"section": "store_info.contact", "grade": 4}]},
    ],
)
def test_relevance_cannot_invent_answerability(updates: dict[str, Any]) -> None:
    data = _source()
    data.update(updates)
    with pytest.raises(ValidationError):
        QuerySource.model_validate(data)


def _chunk(lang: LanguageCode, version: str) -> KBChunk:
    text = {"en": "Support contacts", "ru": "Контакты поддержки", "kk": "Қолдау байланыстары"}[lang]
    return KBChunk(
        chunk_id=make_chunk_id("store_info", version, 0),
        doc_id="store_info",
        version=version,
        lang=lang,
        position=0,
        section_id="contact",
        section_path=["Contact"],
        title="Store",
        text=text,
        locator={"line_range": [1, 2]},
        doc_type="faq",
        format="txt",
        source_path=f"{lang}/store_info.txt",
        effective_from=date(2026, 11, 12),
        token_count=3,
        translation_group="store_info#contact",
        checksum=sha256_text(text),
    )


def test_cross_language_qrels_resolve_only_to_target_language_and_current_ids() -> None:
    data = _source()
    data["cross_lingual"] = True
    source = QueryFile.model_validate(
        {
            "schema_version": "retrieval_source/1",
            "source_id": "supportpilot_authored_retrieval",
            "origin": "llm_generated",
            "review_status": "draft",
            "author": "test-author",
            "allowed_use": "eval_only",
            "licence_note": "Original synthetic test content",
            "items": [data],
        }
    )
    variants: list[tuple[LanguageCode, str]] = [("en", "a"), ("ru", "b"), ("kk", "c")]
    chunks = [_chunk(lang, char * 64) for lang, char in variants]
    items = resolve_queries([source], chunks, "kb_test")
    assert len(items) == 9
    for item in items:
        assert item.relevant[0].chunk_id == next(
            c.chunk_id for c in chunks if c.lang == item.corpus_lang
        )
        assert item.cross_lingual == (item.lang != item.corpus_lang)
    with pytest.raises(StepError, match="missing KB section"):
        resolve_queries([source], chunks[:2], "kb_test")
    replacement = _chunk("kk", "d" * 64)
    updated = resolve_queries([source], [*chunks[:2], replacement], "kb_changed")
    assert chunks[2].chunk_id not in {r.chunk_id for q in updated for r in q.relevant}
