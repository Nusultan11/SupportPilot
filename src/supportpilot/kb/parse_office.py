"""DOCX (python-docx), XLSX (openpyxl) and CSV (stdlib) -> blocks. Chosen by measurement
(ADR-027): python-docx exposes paragraph styles, openpyxl keeps integer cells as int.

DOCX: body children in document order. `Title` -> title, `Subtitle` -> meta, `Heading N` ->
heading level N, other paragraphs -> text (an "Effective from" line is meta whatever its
style), tables -> header row + rows. Headers/footers are not part of the body.
Locator: `block_range` = 1-based index among the body's paragraphs and tables.

XLSX: per sheet, the rows before the header are the title, the effective line and notes;
the header is the first row with >= 2 non-empty text cells; every later non-empty row is a
record keyed by the key column. Locator: sheet + Excel row.

CSV: the first row is the header (delimiter sniffed among , ; tab). Locator: file lines.
"""

from __future__ import annotations

import csv
import io
from typing import Any

from supportpilot.common.schemas import Locator
from supportpilot.kb.blocks import (
    Block,
    BlockKind,
    ExtractionError,
    ParsedDoc,
    clean,
    parse_effective,
)

PARSER_REVISION = 1


def _version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


# ---------------------------------------------------------------- DOCX


def parse_docx(data: bytes) -> ParsedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    d = docx.Document(io.BytesIO(data))
    blocks: list[Block] = []
    idx = 0
    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag not in ("p", "tbl"):
            continue
        idx += 1
        loc = Locator(block_range=(idx, idx))
        if tag == "tbl":
            rows = [[clean(c.text) for c in r.cells] for r in Table(child, d).rows]
            if rows:
                blocks.append(Block("table", loc, header=rows[0], rows=rows[1:]))
            continue
        p = Paragraph(child, d)
        text = clean(p.text)
        if not text:
            continue
        style = p.style.name if p.style is not None else "Normal"
        if style == "Title":
            blocks.append(Block("title", loc, text=text))
        elif style == "Subtitle" or parse_effective(text) is not None:
            blocks.append(Block("meta", loc, text=text))
        elif style.startswith("Heading ") and style[8:].isdigit():
            blocks.append(Block("heading", loc, text=text, level=int(style[8:])))
        else:
            blocks.append(Block("paragraph", loc, text=text))
    if not blocks:
        raise ExtractionError("the DOCX body has no text")
    return ParsedDoc(blocks=blocks, parser=f"python-docx/{_version(docx)}+kb{PARSER_REVISION}")


# ---------------------------------------------------------------- XLSX / CSV


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return clean(str(v))


def _table_blocks(
    rows: list[tuple[int, int, list[str]]], *, key_column: int, sheet: str | None, line_based: bool
) -> list[Block]:
    """Rows (first line, last line, cells) -> preamble blocks + one record block per row."""
    blocks: list[Block] = []
    header: list[str] | None = None

    def loc(a: int, b: int, key: str | None = None) -> Locator:
        if line_based:
            return Locator(row_range=(a, b), record_id=key)
        return Locator(sheet=sheet, row_range=(a, b), record_id=key)

    for n, last, cells in rows:
        filled = [c for c in cells if c]
        if not filled:
            continue
        if header is None:
            if len(filled) >= 2 and all(not c.replace(" ", "").isdigit() for c in filled):
                width = max(i for i, c in enumerate(cells) if c) + 1
                header = cells[:width]
                continue
            text = " ".join(filled)
            kind: BlockKind = (
                "meta" if parse_effective(text) else ("title" if not blocks else "paragraph")
            )
            blocks.append(Block(kind, loc(n, last), text=text))
            continue
        row = (cells + [""] * len(header))[: len(header)]
        if any(c for c in cells[len(header) :]):
            raise ExtractionError(f"row {n} has values outside the header columns")
        key = row[key_column]
        if not key:
            raise ExtractionError(f"row {n} has no key in column {key_column + 1}")
        blocks.append(
            Block("record", loc(n, last, key), header=header, rows=[row], key=key, sheet=sheet)
        )
    if header is None:
        raise ExtractionError("no header row found")
    return blocks


def parse_xlsx(data: bytes, key_column: int = 0) -> ParsedDoc:
    import openpyxl

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    blocks: list[Block] = []
    try:
        for ws in wb.worksheets:
            rows = [
                (n, n, [_cell(v) for v in r])
                for n, r in enumerate(ws.iter_rows(values_only=True), start=1)
            ]
            blocks += _table_blocks(rows, key_column=key_column, sheet=ws.title, line_based=False)
    finally:
        wb.close()
    return ParsedDoc(blocks=blocks, parser=f"openpyxl/{openpyxl.__version__}+kb{PARSER_REVISION}")


def parse_csv(data: bytes, key_column: int = 0) -> ParsedDoc:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ExtractionError(f"CSV is not UTF-8: {e}") from e
    try:
        dialect: Any = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(text, newline=""), dialect)
    rows: list[tuple[int, int, list[str]]] = []
    start = 1
    for cells in reader:
        rows.append((start, reader.line_num, [clean(c) for c in cells]))
        start = reader.line_num + 1
    blocks = _table_blocks(rows, key_column=key_column, sheet=None, line_based=True)
    return ParsedDoc(blocks=blocks, parser=f"csv/stdlib+kb{PARSER_REVISION}")
