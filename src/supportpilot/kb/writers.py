"""Rendered document -> file bytes, one writer per format. Output is byte-deterministic.

- PDF: reportlab platypus (`invariant=1`: fixed dates and ids), DejaVu Sans from matplotlib's
  bundled fonts (covers Kazakh Cyrillic), running header and page-number footer.
- DOCX: python-docx with Title / Subtitle / Heading N styles and "Table Grid" tables.
- XLSX: openpyxl, a title row, the effective line, a header row and one row per record.
- CSV, JSON (FAQ export), TXT (setext headings, Q:/A: items, pipe tables, wrapped lines).
Office files are zip containers whose entry times and core-property dates are rewritten to
fixed values, so the same input gives the same bytes and the same document version.
"""

from __future__ import annotations

import csv
import io
import json
import re
import textwrap
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from supportpilot.common.hashing import sha256_file
from supportpilot.common.schemas import DocFormat, LanguageCode
from supportpilot.kb.render import RenderedDoc, RRecords, RTable, cell_text

FONT_REGULAR = "SPSans"
FONT_BOLD = "SPSans-Bold"
TXT_WIDTH = 88
ZIP_TIME = (2026, 1, 1, 0, 0, 0)
PAGE_LABEL = {"en": "Page {n}", "ru": "Стр. {n}", "kk": "{n}-бет"}


@cache
def font_files() -> dict[str, Path]:
    """DejaVu Sans regular/bold shipped with matplotlib (a pinned dependency)."""
    import matplotlib

    ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    files = {FONT_REGULAR: ttf / "DejaVuSans.ttf", FONT_BOLD: ttf / "DejaVuSans-Bold.ttf"}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"font not found: {path}")
    return files


def writer_params() -> dict[str, Any]:
    """Everything besides the content that changes the bytes (part of the render manifest)."""
    import docx
    import openpyxl
    import reportlab

    return {
        "reportlab": reportlab.Version,
        "python_docx": docx.__version__ if hasattr(docx, "__version__") else "unknown",
        "openpyxl": openpyxl.__version__,
        "fonts": {name: sha256_file(p) for name, p in font_files().items()},
        "txt_width": TXT_WIDTH,
    }


def write(doc: RenderedDoc) -> bytes:
    writers: dict[DocFormat, Callable[[RenderedDoc], bytes]] = {
        DocFormat.PDF: write_pdf,
        DocFormat.DOCX: write_docx,
        DocFormat.XLSX: write_xlsx,
        DocFormat.CSV: write_csv,
        DocFormat.JSON: write_json,
        DocFormat.TXT: write_txt,
    }
    return writers[doc.format](doc)


# ---------------------------------------------------------------- PDF


@cache
def _register_fonts() -> None:
    from reportlab.lib.fonts import addMapping
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    files = font_files()
    for name, path in files.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    addMapping(FONT_REGULAR, 0, 0, FONT_REGULAR)
    addMapping(FONT_REGULAR, 1, 0, FONT_BOLD)


def write_pdf(doc: RenderedDoc) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    _register_fonts()
    grey = colors.HexColor("#555555")
    st = {
        "title": ParagraphStyle("t", fontName=FONT_BOLD, fontSize=18, leading=22, spaceAfter=4),
        "meta": ParagraphStyle("m", fontName=FONT_REGULAR, fontSize=9.5, leading=12,
                               textColor=grey, spaceAfter=10),
        "body": ParagraphStyle("b", fontName=FONT_REGULAR, fontSize=10.5, leading=14.5,
                               spaceAfter=6),
        1: ParagraphStyle("h1", fontName=FONT_BOLD, fontSize=13.5, leading=17, spaceBefore=10,
                          spaceAfter=4, keepWithNext=1),
        2: ParagraphStyle("h2", fontName=FONT_BOLD, fontSize=11.5, leading=15, spaceBefore=6,
                          spaceAfter=3, keepWithNext=1),
        "cell": ParagraphStyle("c", fontName=FONT_REGULAR, fontSize=9.5, leading=12),
        "cellh": ParagraphStyle("ch", fontName=FONT_BOLD, fontSize=9.5, leading=12),
    }  # fmt: skip
    width = A4[0] - 4 * cm

    def para(text: str, style: Any) -> Any:
        return Paragraph(escape(text), style)

    def table(t: RTable) -> Any:
        data = [[para(h, st["cellh"]) for h in t.header]]
        data += [[para(cell_text(c, doc.lang), st["cell"]) for c in row] for row in t.rows]
        tbl = Table(data, colWidths=[width / len(t.header)] * len(t.header), repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#888888")),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        return tbl

    story: list[Any] = [para(doc.title, st["title"]), para(doc.effective_text, st["meta"])]
    story += [para(p, st["body"]) for p in doc.intro]
    for s in doc.sections:
        story.append(para(s.heading, st[s.level]))
        story += [para(p, st["body"]) for p in s.paragraphs]
        if s.table is not None:
            story += [Spacer(1, 2), table(s.table), Spacer(1, 8)]

    header = f"{doc.store_name} · {doc.title}"
    label = PAGE_LABEL[doc.lang]

    def decorate(canvas: Any, tpl: Any) -> None:
        canvas.saveState()
        canvas.setFont(FONT_REGULAR, 8)
        canvas.setFillColor(grey)
        canvas.drawString(2 * cm, A4[1] - 1.2 * cm, header)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, label.format(n=tpl.page))
        canvas.restoreState()

    buf = io.BytesIO()
    tpl = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2 * cm,
        title=doc.title,
        author=doc.store_name,
        subject=doc.doc_id,
        creator="supportpilot kb render",
        invariant=1,
    )
    tpl.build(story, onFirstPage=decorate, onLaterPages=decorate)
    return buf.getvalue()


# ---------------------------------------------------------------- DOCX / XLSX


def _fixed_time(doc: RenderedDoc) -> datetime:
    return datetime(doc.effective_from.year, doc.effective_from.month, doc.effective_from.day)


_CORE_DATE_RE = re.compile(rb"(<dcterms:(created|modified)\b[^>]*>)[^<]*(</dcterms:\2>)")


def normalize_zip(data: bytes, when: datetime) -> bytes:
    """Rewrite an OOXML zip with fixed entry times and fixed core-property dates."""
    stamp = when.replace(tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%SZ").encode()
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(data)) as zin,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout,
    ):
        for info in zin.infolist():
            content = zin.read(info.filename)
            if info.filename == "docProps/core.xml":
                content = _CORE_DATE_RE.sub(rb"\g<1>" + stamp + rb"\g<3>", content)
            zi = zipfile.ZipInfo(info.filename, date_time=ZIP_TIME)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.create_system = 0
            zi.external_attr = 0o600 << 16
            zout.writestr(zi, content, compresslevel=6)
    return out.getvalue()


def write_docx(doc: RenderedDoc) -> bytes:
    import docx
    from docx.shared import Pt

    d = docx.Document()
    when = _fixed_time(doc)
    props = d.core_properties
    props.author = doc.store_name
    props.last_modified_by = doc.store_name
    props.title = doc.title
    props.subject = doc.doc_id
    props.language = doc.lang
    props.created = when
    props.modified = when
    props.revision = 1
    d.add_heading(doc.title, 0)
    d.add_paragraph(doc.effective_text, style="Subtitle")
    for p in doc.intro:
        d.add_paragraph(p)
    for s in doc.sections:
        d.add_heading(s.heading, s.level)
        for p in s.paragraphs:
            d.add_paragraph(p)
        if s.table is not None:
            t = d.add_table(rows=1 + len(s.table.rows), cols=len(s.table.header))
            t.style = "Table Grid"
            for j, h in enumerate(s.table.header):
                run = t.rows[0].cells[j].paragraphs[0].add_run(h)
                run.bold = True
            for i, row in enumerate(s.table.rows, start=1):
                for j, c in enumerate(row):
                    t.rows[i].cells[j].text = cell_text(c, doc.lang)
    header = d.sections[0].header.paragraphs[0]
    header.text = f"{doc.store_name} · {doc.title}"
    header.runs[0].font.size = Pt(8)
    buf = io.BytesIO()
    d.save(buf)
    return normalize_zip(buf.getvalue(), when)


def write_xlsx(doc: RenderedDoc) -> bytes:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill

    rs = _records(doc)
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = (rs.sheet or doc.title)[:31]
    ws["A1"] = doc.title
    ws["A1"].font = Font(bold=True, size=14)
    ws["A2"] = doc.effective_text
    header_row = 4
    for j, h in enumerate(rs.header, start=1):
        c = ws.cell(row=header_row, column=j, value=h)
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="EEEEEE")
    for i, rec in enumerate(rs.rows, start=header_row + 1):
        for j, value in enumerate(rec.cells, start=1):
            if value == "":
                continue
            c = ws.cell(row=i, column=j, value=value)
            if isinstance(value, int):
                c.number_format = "#,##0"
    for j in range(1, len(rs.header) + 1):
        ws.column_dimensions[chr(ord("A") + j - 1)].width = 18 if j < len(rs.header) else 60
    ws.freeze_panes = f"A{header_row + 1}"
    when = _fixed_time(doc)
    wb.properties.creator = doc.store_name
    wb.properties.lastModifiedBy = doc.store_name
    wb.properties.title = doc.title
    wb.properties.subject = doc.doc_id
    wb.properties.language = doc.lang
    wb.properties.created = when
    wb.properties.modified = when
    buf = io.BytesIO()
    wb.save(buf)
    return normalize_zip(buf.getvalue(), when)


def _records(doc: RenderedDoc) -> RRecords:
    if doc.records is None:
        raise ValueError(f"{doc.doc_id}: {doc.format} needs a record document")
    return doc.records


# ---------------------------------------------------------------- CSV / JSON / TXT


def write_csv(doc: RenderedDoc) -> bytes:
    rs = _records(doc)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(rs.header)
    for rec in rs.rows:
        w.writerow([cell_text(c, doc.lang) for c in rec.cells])
    return buf.getvalue().encode("utf-8")


def write_json(doc: RenderedDoc) -> bytes:
    obj = {
        "schema_version": "kb_faq_export/1",
        "doc_id": doc.doc_id,
        "lang": doc.lang,
        "store": doc.store_name,
        "title": doc.title,
        "effective_from": doc.effective_from.isoformat(),
        "intro": doc.intro,
        "items": [
            {"id": s.id, "question": s.heading, "answer": "\n".join(s.paragraphs)}
            for s in doc.sections
        ],
    }
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _wrap(text: str, first: str = "", rest: str = "") -> list[str]:
    return textwrap.wrap(
        text,
        width=TXT_WIDTH,
        initial_indent=first,
        subsequent_indent=rest,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _pipe_table(t: RTable, lang: LanguageCode) -> list[str]:
    lines = ["| " + " | ".join(t.header) + " |", "|" + "|".join("---" for _ in t.header) + "|"]
    lines += ["| " + " | ".join(cell_text(c, lang) for c in row) + " |" for row in t.rows]
    return lines


def write_txt(doc: RenderedDoc) -> bytes:
    lines = [doc.title, "=" * len(doc.title), doc.effective_text, ""]
    for p in doc.intro:
        lines += [*_wrap(p), ""]
    faq = doc.doc_type.value == "faq"
    for s in doc.sections:
        if faq:
            lines += _wrap(s.heading, "Q: ", "   ")
            for k, p in enumerate(s.paragraphs):
                lines += _wrap(p, "A: " if k == 0 else "   ", "   ")
            lines.append("")
            continue
        lines += [s.heading, ("-" if s.level == 1 else "~") * len(s.heading)]
        for p in s.paragraphs:
            lines += [*_wrap(p), ""]
        if s.table is not None:
            lines += [*_pipe_table(s.table, doc.lang), ""]
    return ("\n".join(lines).rstrip("\n") + "\n").encode("utf-8")
