"""Stage 5 unit tests: specs, facts, writers, parsers per format, structure, chunking, gates."""

from __future__ import annotations

import dataclasses
import io
import json
import zipfile
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
import yaml

from supportpilot.common.schemas import (
    LANGUAGES,
    DocFormat,
    DocType,
    KBChunk,
    LanguageCode,
    Locator,
)
from supportpilot.data.locale import load_locale
from supportpilot.kb import checks
from supportpilot.kb.blocks import ExtractionError, parse_effective
from supportpilot.kb.build import _register_entry, parse_file
from supportpilot.kb.chunk import chunk_document
from supportpilot.kb.locate import SourceReader
from supportpilot.kb.parse_office import parse_csv, parse_docx, parse_xlsx
from supportpilot.kb.parse_pdf import parse_pdf
from supportpilot.kb.parse_text import parse_json, parse_txt
from supportpilot.kb.register import Expected, ExpectedDoc, ExpectedPlacement
from supportpilot.kb.render import RenderedDoc, RSection, RTable, render_doc
from supportpilot.kb.spec import (
    DocSpec,
    Filler,
    KBSources,
    SpecError,
    effective_line,
    fact_registry,
    load_sources,
    malformed_facts,
)
from supportpilot.kb.structure import build_document
from supportpilot.kb.writers import normalize_zip, write


@pytest.mark.parametrize(
    ("value", "surface", "body", "found"),
    [
        (-10, "-10", "Ambient temperature: -10 C.", True),
        (10, "10", "Ambient temperature: -10 C.", False),
        (-10, "-10", "Ambient temperature: -100 C.", False),
        (2.5, "2.5", "Output: 22.5 W.", False),
        (2.5, "2.5", "Output: 2.5 W.", True),
        (1, "1", "SKU AX-0001.", False),
        (1000, "1 000", "Price: 1 000 KZT.", True),
        (100000, "KZT 100,000", "Limit: KZT 100,000.", True),
    ],
)
def test_numeric_fact_matching_preserves_sign_and_token_boundaries(
    value: int | float,
    surface: str,
    body: str,
    found: bool,
) -> None:
    from tests.unit.test_retrieval import chunk

    placement = ExpectedPlacement(
        key="test.value", value=value, surface=surface, section_id="terms", where="paragraph"
    )
    assert (checks._find(placement, [chunk(text=body)]) is not None) is found


ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "kb_sources" / "demo_store"
LOCALE = load_locale(ROOT / "configs" / "locale_kz.yaml")


def words_count(texts: Sequence[str]) -> list[int]:
    return [max(1, len(t.split())) for t in texts]


@pytest.fixture(scope="module")
def sources() -> KBSources:
    return load_sources(SOURCES)


def spec(sources: KBSources, doc_id: str) -> DocSpec:
    return next(d for d in sources.docs if d.doc_id == doc_id)


def rendered(
    sources: KBSources, doc_id: str, lang: LanguageCode, fmt: DocFormat | None = None
) -> RenderedDoc:
    doc = render_doc(spec(sources, doc_id), sources, lang, LOCALE.lookup)
    return dataclasses.replace(doc, format=fmt) if fmt else doc


def extract(doc: RenderedDoc, max_tokens: int = 400) -> tuple[list[KBChunk], bytes]:
    data = write(doc)
    entry = _register_entry(doc, data)
    kb_doc = build_document(parse_file(data, entry), entry)
    return chunk_document(kb_doc, max_tokens=max_tokens, count_tokens=words_count), data


def expected_of(doc: RenderedDoc) -> Expected:
    return Expected(
        facts={},
        documents={
            doc.path: ExpectedDoc(
                title=doc.title,
                effective_from=doc.effective_from,
                placements=[ExpectedPlacement(**p.__dict__) for p in doc.placements],
                content=doc.content_strings(),
            )
        },
    )


# ---------------------------------------------------------------- specs and facts


def test_sources_load_and_cover_every_format_and_type(sources: KBSources) -> None:
    assert {d.format for d in sources.docs} == set(DocFormat)
    assert {d.doc_type for d in sources.docs} == set(DocType)
    assert len(sources.catalog.products) >= 30


def test_malformed_facts_catch_split_flow_values() -> None:
    bad = yaml.safe_load("a: {ru: пр. Достык, 77, kk: x}")
    assert malformed_facts(bad) == ["a.77 (non-string key)", "a.77 (empty)"]
    assert malformed_facts({"a": {"ru": "пр. Достык, 77"}, "b": [1, 2]}) == []


def test_doc_spec_validation() -> None:
    base = {"doc_id": "x", "doc_type": "faq", "format": "txt",
            "title": {"en": "a", "ru": "б", "kk": "ә"}}  # fmt: skip
    sec = {"id": "q", "heading": {"en": "a", "ru": "б", "kk": "ә"},
           "paragraphs": [{"en": "a", "ru": "б", "kk": "ә"}]}  # fmt: skip
    DocSpec.model_validate({**base, "sections": [sec]})
    with pytest.raises(ValueError, match="either sections or records"):
        DocSpec.model_validate(base)
    with pytest.raises(ValueError, match="FAQ items"):
        DocSpec.model_validate({**base, "sections": [sec, {**sec, "id": "r", "level": 2}]})
    with pytest.raises(ValueError, match="record documents"):
        DocSpec.model_validate({**base, "format": "xlsx", "sections": [sec]})
    with pytest.raises(ValueError, match="duplicate section"):
        DocSpec.model_validate({**base, "sections": [sec, sec]})


def test_filler_surfaces_and_keys(sources: KBSources) -> None:
    facts = {**sources.facts, "x_sec": 60, "x_min": 21, "x_days": 21}
    f = Filler(facts, LOCALE.lookup)
    assert f.fill("{x_sec:seconds@abl}", "kk")[0] == "60 секундтан"
    assert f.fill("{x_min:minutes}", "ru")[0] == "21 минута"
    assert f.fill("{x_days:days_gen}", "ru")[0] == "21 дня"
    assert f.fill("{x_days:days}", "ru")[0] == "21 день"
    text, hits = f.fill("{sale:sale_name}: {sale:sale_from@abl}", "kk")
    assert [h.key for h in hits] == ["sale.name", "sale.from"]
    assert text.startswith(sources.facts["sale"]["kk"])
    _, hits = f.fill("{showroom:address}", "ru")
    assert hits[0].key == "showroom.address"
    assert "77" in hits[0].surface
    with pytest.raises(SpecError, match="unknown fact"):
        f.fill("{nope}", "en")
    with pytest.raises(SpecError, match="outside a record"):
        f.fill("{rec.sku}", "en")


def test_fact_registry_is_flat_and_complete(sources: KBSources) -> None:
    reg = fact_registry(sources)
    assert "showroom.address" in reg
    assert "sale.name" in reg
    assert "sale.from" in reg
    assert reg["product.TV-5501.price"] == 239000
    assert "zone.almaty.zone_name" in reg
    assert "schema_version" not in reg
    assert "showroom" not in reg


@pytest.mark.parametrize("lang", LANGUAGES)
def test_effective_line_round_trip(lang: LanguageCode) -> None:
    for m in range(1, 13):
        d = date(2026, m, 1 + m)
        assert parse_effective(effective_line(d, lang)) == d
    assert parse_effective("Effective from 1 Smarch 2026") is None
    assert parse_effective("some text") is None


# ---------------------------------------------------------------- writers


@pytest.mark.parametrize(
    ("doc_id", "fmt"),
    [("returns_policy", DocFormat.PDF), ("delivery_policy", DocFormat.DOCX),
     ("catalog", DocFormat.XLSX), ("delivery_zones", DocFormat.CSV),
     ("store_info", DocFormat.JSON), ("account_faq", DocFormat.TXT)],
)  # fmt: skip
def test_writers_are_byte_deterministic(sources: KBSources, doc_id: str, fmt: DocFormat) -> None:
    for lang in LANGUAGES:
        doc = rendered(sources, doc_id, lang, fmt)
        assert write(doc) == write(doc)


def test_normalize_zip_fixes_times_and_dates() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "docProps/core.xml", "<dcterms:created x='1'>2099-01-01T00:00:00Z</dcterms:created>"
        )
        z.writestr("a.xml", "<a/>")
    from datetime import datetime

    out = normalize_zip(buf.getvalue(), datetime(2026, 11, 1))
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        assert {i.date_time for i in z.infolist()} == {(2026, 1, 1, 0, 0, 0)}
        assert b"2026-11-01T00:00:00Z" in z.read("docProps/core.xml")


# ---------------------------------------------------------------- parsers


def test_pdf_structure_tables_and_furniture(sources: KBSources) -> None:
    doc = rendered(sources, "returns_policy", "kk")
    blocks = parse_pdf(write(doc)).blocks
    kinds = [b.kind for b in blocks]
    assert kinds[0] == "title"
    assert kinds[1] == "meta"
    headings = [b.text for b in blocks if b.kind == "heading"]
    assert headings == [s.heading for s in doc.sections]
    table = next(b for b in blocks if b.kind == "table")
    expected = next(s.table for s in doc.sections if s.table)
    assert table.header == expected.header
    assert len(table.rows) == len(expected.rows)
    assert not any("бет" in b.text and b.text[0].isdigit() for b in blocks)  # page label dropped
    assert not any(doc.store_name + " ·" in b.text for b in blocks)  # running header dropped


def test_pdf_multipage_paragraphs_and_split_tables(sources: KBSources) -> None:
    doc = rendered(sources, "returns_policy", "ru")
    secs = [dataclasses.replace(s, id=f"{s.id}-{k}", heading=f"{k}{s.heading}")
            for k in range(1, 6) for s in doc.sections]  # fmt: skip
    rows: list[list[str | int]] = [[str(i), f"строка {i}", f"{1000 * i} тенге"] for i in range(70)]
    big = RTable(header=["№", "Описание", "Сумма"], rows=rows)
    secs.append(RSection("big", "99. Большая таблица", 1, ["Таблица ниже."], big))
    long = dataclasses.replace(doc, sections=secs)
    parsed = parse_pdf(write(long))
    pages = {b.locator.page for b in parsed.blocks}
    assert len(pages) >= 3
    tables = [b for b in parsed.blocks if b.kind == "table" and b.header == big.header]
    assert len(tables) == 1, "a table continued on the next page is merged"
    t = tables[0]
    assert [r[0] for r in t.rows] == [str(i) for i in range(70)]
    assert t.locator.page_end is not None
    assert t.locator.page_end > (t.locator.page or 0)
    paras = {b.text for b in parsed.blocks if b.kind == "paragraph"}
    for s in secs:
        for p in s.paragraphs:
            assert p in paras, "every paragraph is recovered whole, also across page breaks"


def test_pdf_without_text_layer_is_an_error() -> None:
    from PIL import Image
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    img = io.BytesIO()
    Image.new("RGB", (40, 40), (200, 10, 10)).save(img, format="PNG")
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4, invariant=1)
    c.drawImage(ImageReader(io.BytesIO(img.getvalue())), 100, 500, 200, 200)
    c.showPage()
    c.save()
    with pytest.raises(ExtractionError, match="OCR"):
        parse_pdf(buf.getvalue())


def test_docx_styles_levels_and_tables(sources: KBSources) -> None:
    doc = rendered(sources, "terms", "en")
    blocks = parse_docx(write(doc)).blocks
    levels = [(b.text, b.level) for b in blocks if b.kind == "heading"]
    assert levels == [(s.heading, s.level) for s in doc.sections]
    assert blocks[0].kind == "title"
    assert blocks[1].kind == "meta"
    assert all(b.locator.block_range is not None for b in blocks)
    table_doc = rendered(sources, "delivery_policy", "ru")
    t = next(b for b in parse_docx(write(table_doc)).blocks if b.kind == "table")
    assert t.header == next(s.table for s in table_doc.sections if s.table).header


def test_xlsx_header_detection_records_and_ints(sources: KBSources) -> None:
    doc = rendered(sources, "catalog", "ru")
    blocks = parse_xlsx(write(doc)).blocks
    assert [b.kind for b in blocks[:2]] == ["title", "meta"]
    recs = [b for b in blocks if b.kind == "record"]
    assert [b.key for b in recs] == [r.key for r in doc.records.rows]  # type: ignore[union-attr]
    first = recs[0]
    assert first.locator.sheet == "Товары"
    assert first.locator.row_range == (5, 5)
    assert first.rows[0][3] == "239000", "numeric cells come back as integers"
    assert first.header == doc.records.header  # type: ignore[union-attr]


def test_csv_sniffs_delimiter_and_counts_lines() -> None:
    data = b'code;name;note\nA1;Alpha;"multi\nline"\nB2;Beta;x\n'
    blocks = parse_csv(data).blocks
    assert [b.key for b in blocks] == ["A1", "B2"]
    assert blocks[0].locator.row_range == (2, 3)
    assert blocks[1].locator.row_range == (4, 4)
    assert blocks[0].rows[0][2] == "multi line"
    with pytest.raises(ExtractionError, match="no key"):
        parse_csv(b"code,name\n,Alpha\n")


def test_txt_headings_faq_and_pipe_tables(sources: KBSources) -> None:
    doc = rendered(sources, "returns_policy", "en", DocFormat.TXT)
    blocks = parse_txt(write(doc)).blocks
    assert [b.text for b in blocks if b.kind == "heading"] == [s.heading for s in doc.sections]
    t = next(b for b in blocks if b.kind == "table")
    assert t.header == ["Action", "Period or cost"]
    assert len(t.rows) == 5
    faq = rendered(sources, "account_faq", "kk")
    fb = parse_txt(write(faq)).blocks
    qs = [b.text for b in fb if b.kind == "heading"]
    assert qs == [s.heading for s in faq.sections]
    answers = [b.text for b in fb if b.kind == "paragraph"]
    assert answers == [" ".join(s.paragraphs) for s in faq.sections]


def test_json_faq_ids_and_generic_fallback(sources: KBSources) -> None:
    doc = rendered(sources, "store_info", "ru")
    blocks = parse_json(write(doc)).blocks
    assert [b.section_id for b in blocks if b.kind == "heading"] == [s.id for s in doc.sections]
    assert blocks[1].kind == "meta"
    assert blocks[1].text == "2026-11-01"
    generic = parse_json(json.dumps({"a": {"b": [1, "x"]}, "c": "y/z"}).encode()).blocks
    assert [(b.text, b.locator.json_pointer) for b in generic] == [
        ("a.b.0: 1", "/a/b/0"), ("a.b.1: x", "/a/b/1"), ("c: y/z", "/c")]  # fmt: skip
    with pytest.raises(ExtractionError):
        parse_json(b"{not json")


# ---------------------------------------------------------------- structure and chunks


def test_structure_links_sections_and_nests_level2(sources: KBSources) -> None:
    doc = rendered(sources, "terms", "kk")
    data = write(doc)
    entry = _register_entry(doc, data)
    kb = build_document(parse_file(data, entry), entry)
    assert kb.sections[0].level == 0
    assert kb.sections[0].section_id is None
    pdata = next(s for s in kb.sections if s.section_id == "personal_data")
    assert [c.section_id for c in pdata.children] == ["data_stored", "data_deletion"]
    assert kb.effective_from == date(2026, 11, 1)
    assert kb.title == doc.title
    renamed = entry.model_copy(update={"sections": entry.sections[1:]})
    kb2 = build_document(parse_file(data, renamed), renamed)
    assert kb2.sections[1].section_id is None  # unknown heading is kept, not guessed


def test_faq_chunk_is_one_question_and_answer(sources: KBSources) -> None:
    doc = rendered(sources, "account_faq", "ru")
    chunks, _ = extract(doc)
    assert [c.section_id for c in chunks] == [s.id for s in doc.sections]
    for c, s in zip(chunks, doc.sections, strict=True):
        assert c.text.split("\n")[0] == s.heading
        assert all(p in c.text for p in s.paragraphs)
        assert c.translation_group == f"account_faq#{s.id}"


def test_record_chunks_keep_header_key_and_sheet(sources: KBSources) -> None:
    chunks, _ = extract(rendered(sources, "catalog", "en"))
    c = next(c for c in chunks if c.section_id == "cf-0812")
    lines = c.text.split("\n")
    assert lines[0] == "Product catalogue · Products"
    assert "SKU: CF-0812" in lines
    assert "Price, KZT: 119,000" in lines
    assert not any(line.startswith("Warranty") for line in lines)  # home category, no warranty
    assert c.locator.record_id == "CF-0812"
    assert c.locator.sheet == "Products"


def test_long_sections_split_with_heading_and_table_header(sources: KBSources) -> None:
    doc = rendered(sources, "returns_policy", "en")
    chunks, _ = extract(doc, max_tokens=40)
    assert all(c.token_count <= 40 for c in chunks)
    by: dict[str | None, list[KBChunk]] = {}
    for c in chunks:
        by.setdefault(c.section_id, []).append(c)
    window = by["window"]
    assert len(window) == 2
    assert [c.part for c in window] == [0, 1]
    assert all(c.text.startswith("1. Return period\n") for c in window)
    table_parts = [c for c in by["periods_table"] if "|---|" in c.text]
    assert len(table_parts) >= 2
    assert all("| Action | Period or cost |" in c.text for c in table_parts)
    expected = expected_of(doc)
    gate, _ = checks.gate_facts(chunks, expected)
    assert gate["passed"], gate["failures"]
    assert checks.gate_completeness(chunks, [], expected)["passed"]


# ---------------------------------------------------------------- every document x format


def _compatible(d: DocSpec) -> list[DocFormat]:
    if d.records is not None:
        return [DocFormat.XLSX, DocFormat.CSV]
    fmts = [DocFormat.PDF, DocFormat.DOCX, DocFormat.TXT]
    return [*fmts, DocFormat.JSON] if d.doc_type is DocType.FAQ else fmts


def test_every_document_in_every_format_and_language_keeps_all_facts(sources: KBSources) -> None:
    """The extraction contract on real generated documents: 100% of placed facts found in
    the chunk of their section, every content word kept, every locator verified."""
    combos = 0
    for d in sources.docs:
        for fmt in _compatible(d):
            for lang in LANGUAGES:
                doc = rendered(sources, d.doc_id, lang, fmt)
                chunks, data = extract(doc)
                expected = expected_of(doc)
                gate, found = checks.gate_facts(chunks, expected)
                assert gate["passed"], (d.doc_id, fmt, lang, gate["failures"])
                comp = checks.gate_completeness(chunks, [], expected)
                assert comp["passed"], (d.doc_id, fmt, lang, comp["failures"])
                tmp = {doc.path: data}
                readers = SourceReader(tmp[doc.path], fmt, lang)
                for p, c in found:
                    assert checks.norm(p.surface) in checks.norm(readers.text_at(c.locator))
                combos += 1
    assert combos == 3 * sum(len(_compatible(d)) for d in sources.docs)


# ---------------------------------------------------------------- gates


def test_numbers_normalise_grouping() -> None:
    assert checks.numbers("KZT 1,200 and 25,000; 2–3 days") == checks.numbers(
        "1 200 тенге и 25 000; 2–3 дня"
    )
    assert checks.numbers("20, 24 and 28 cm") == {"20": 1, "24": 1, "28": 1}
    assert checks.numbers("1.7 l") == {"1.7": 1}


def _chunk(text: str, lang: LanguageCode = "en", group: str = "d#s", sid: str = "s") -> KBChunk:
    from supportpilot.common.hashing import sha256_text
    from supportpilot.common.ids import make_chunk_id

    version = sha256_text(text + lang)
    return KBChunk(
        chunk_id=make_chunk_id("d", version, 0), doc_id="d", version=version, lang=lang,
        position=0, section_id=sid, section_path=["s"], title="t", text=text,
        locator=Locator(page=1), doc_type=DocType.POLICY, format=DocFormat.PDF,
        source_path=f"{lang}/d.pdf", effective_from=date(2026, 11, 1), token_count=3,
        translation_group=group, checksum=sha256_text(text),
    )  # fmt: skip


def test_gate_facts_reports_a_missing_or_changed_value() -> None:
    p = ExpectedPlacement(key="return_window_days", value=14, surface="14 days",
                          section_id="s", where="paragraph")  # fmt: skip
    doc = ExpectedDoc(title="t", effective_from=date(2026, 11, 1), placements=[p],
                      content=["14 days"])  # fmt: skip
    exp = Expected(facts={}, documents={"en/d.pdf": doc})
    assert checks.gate_facts([_chunk("within 14 days")], exp)[0]["passed"]
    bad = checks.gate_facts([_chunk("within 41 days")], exp)[0]
    assert not bad["passed"]
    assert bad["recall"] == 0.0
    wrong_section = checks.gate_facts([_chunk("within 14 days", sid="other")], exp)[0]
    assert not wrong_section["passed"]


def test_gate_completeness_reports_lost_and_extra_words() -> None:
    exp = Expected(facts={}, documents={"en/d.pdf": ExpectedDoc(
        title="t", effective_from=date(2026, 11, 1), placements=[],
        content=["alpha beta gamma"])})  # fmt: skip
    assert checks.gate_completeness([_chunk("alpha beta gamma")], [], exp)["passed"]
    lost = checks.gate_completeness([_chunk("alpha gamma")], [], exp)
    assert not lost["passed"]
    assert "lost" in lost["failures"][0]
    extra = checks.gate_completeness([_chunk("alpha beta gamma Page 2")], [], exp)
    assert not extra["passed"]
    assert "not in the document" in extra["failures"][0]


def test_gate_cross_lingual_and_pii() -> None:
    exp = Expected(facts={}, documents={})
    ok = [_chunk("KZT 1,200", "en"), _chunk("1 200 тенге", "ru"), _chunk("1 200 теңге", "kk")]
    assert checks.gate_cross_lingual(ok, exp)["passed"]
    bad = [*ok[:2], _chunk("1 300 теңге", "kk")]
    assert not checks.gate_cross_lingual(bad, exp)["passed"]
    assert not checks.gate_cross_lingual(ok[:2], exp)["passed"]  # a language missing
    facts = {"support_email": "a@x.example", "support_phone": "+7 700 555 0120"}
    assert checks.gate_pii([_chunk("mail a@x.example or +7 700 555 0120")], facts)["passed"]
    assert not checks.gate_pii([_chunk("mail b@y.example")], facts)["passed"]
    assert not checks.gate_pii([_chunk("call +7 701 123 4567")], facts)["passed"]


def test_gate_evidence_resolves_gold_references(tmp_path: Path) -> None:
    (tmp_path / "g.yaml").write_text("items:\n  - ev: [d.s]\n  - ev: [d.missing]\n", "utf-8")
    chunks = [_chunk("x", lang) for lang in LANGUAGES]
    gate = checks.gate_evidence(chunks, tmp_path)
    assert not gate["passed"]
    assert gate["failures"][0].startswith("d.missing")
    assert checks.evidence_map(chunks)["d.s"]["kk"] == [chunks[2].chunk_id]
    assert checks.gate_evidence(chunks, None)["status"] == "not_applicable"
