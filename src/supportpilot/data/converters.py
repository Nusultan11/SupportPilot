"""Per-source converters: raw files -> `RawRecord` rows (DATA_CONTRACTS §4a).

Rules for every converter:
- Values are copied exactly (no normalisation, no placeholder filling). Labels that a source
  does not provide stay `None`; nothing is inferred.
- `record_id` = `{source_id}:{file_key}.{row}` where `row` is the 0-based data row in the file
  at the pinned revision (deterministic).
- Fields without a dedicated column go to `extra_json` (sorted keys).
- Bad rows become `Rejected(reason)` instead of failing the whole source; structural problems
  (missing columns, unreadable file) raise `ConversionError`.
- Files are read in a streaming way (csv module, line-by-line JSONL, Parquet batches).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_record_id
from supportpilot.common.schemas import LanguageCode, RawRecord, RecordType, SourceEntry

csv.field_size_limit(2**31 - 1)


class ConversionError(Exception):
    """The source cannot be converted (structure differs from what the converter expects)."""


@dataclass(frozen=True)
class Rejected:
    source_file: str
    source_row: int
    reason: str
    snippet: str = ""


Row = RawRecord | Rejected


@dataclass(frozen=True)
class Converter:
    name: str
    version: int  # bump when output for the same input changes
    convert: Callable[[Path, SourceEntry], Iterator[Row]]


def _extra(values: dict[str, Any]) -> str:
    return json.dumps(values, ensure_ascii=False, sort_keys=True, default=str)


def _record(
    entry: SourceEntry,
    *,
    rel: str,
    key: str,
    row: int,
    split: str | None,
    record_type: RecordType,
    lang: LanguageCode,
    text: str,
    response: str | None = None,
    intent: str | None = None,
    category: str | None = None,
    tags: str | None = None,
    extra: dict[str, Any] | None = None,
) -> RawRecord:
    return RawRecord(
        record_id=make_record_id(entry.source_id, f"{key}.{row}"),
        source_id=entry.source_id,
        revision=entry.revision,
        source_file=rel,
        source_row=row,
        source_split=split,
        record_type=record_type,
        lang=lang,
        text=text,
        response=response,
        source_intent=intent,
        source_category=category,
        source_tags=tags,
        extra_json=_extra(extra or {}),
        text_sha256=sha256_text(text),
    )


def _files(raw: Path, pattern: str) -> list[Path]:
    files = sorted(raw.glob(pattern), key=lambda p: p.relative_to(raw).as_posix())
    if not files:
        raise ConversionError(f"no files match {pattern!r} in {raw}")
    return files


def _blank(value: object) -> bool:
    return not isinstance(value, str) or not value.strip()


def _empty_to_none(value: str | None) -> str | None:
    return value if value else None


# ---------------------------------------------------------------- Bitext (CSV)

_BITEXT_TAG_COLUMNS = ("tags", "flags")  # retail uses `tags`, customer support uses `flags`
_BITEXT_REQUIRED = {"instruction", "intent", "category", "response"}


def convert_bitext(raw: Path, entry: SourceEntry) -> Iterator[Row]:
    for path in _files(raw, "*.csv"):
        rel = path.relative_to(raw).as_posix()
        with path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = set(reader.fieldnames or [])
            missing = _BITEXT_REQUIRED - cols
            if missing:
                raise ConversionError(f"{rel}: missing columns {sorted(missing)}")
            tag_col = next((c for c in _BITEXT_TAG_COLUMNS if c in cols), None)
            known = _BITEXT_REQUIRED | {tag_col}
            for row, rec in enumerate(reader):
                if None in rec:  # more fields than header
                    yield Rejected(rel, row, "extra_fields", str(rec.get("instruction"))[:200])
                    continue
                text = rec["instruction"]
                if _blank(text):
                    yield Rejected(rel, row, "empty_text")
                    continue
                yield _record(
                    entry,
                    rel=rel,
                    key="csv",
                    row=row,
                    split=None,
                    record_type=RecordType.SINGLE_TURN,
                    lang="en",
                    text=text,
                    response=_empty_to_none(rec["response"]),
                    intent=_empty_to_none(rec["intent"]),
                    category=_empty_to_none(rec["category"]),
                    tags=_empty_to_none(rec[tag_col]) if tag_col else None,
                    extra={k: v for k, v in rec.items() if k not in known},
                )


# ---------------------------------------------------------------- while-ai (JSONL dialogues)


def _first_user_turn(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for i, m in enumerate(messages):
        if m.get("role") == "user" and not _blank(m.get("content")):
            reply = next(
                (n.get("content") for n in messages[i + 1 :] if n.get("role") == "assistant"),
                None,
            )
            return str(m["content"]), (str(reply) if reply else None)
    return None, None


def convert_whileai(raw: Path, entry: SourceEntry) -> Iterator[Row]:
    for path in _files(raw, "*.jsonl"):
        rel = path.relative_to(raw).as_posix()
        split = path.stem  # train | eval
        with path.open(encoding="utf-8") as fh:
            for row, line in enumerate(fh):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield Rejected(rel, row, "invalid_json", line[:200])
                    continue
                messages = obj.get("messages")
                target = obj.get("target")
                if not isinstance(messages, list) or not isinstance(target, dict):
                    yield Rejected(rel, row, "missing_field:messages/target", line[:200])
                    continue
                text, reply = _first_user_turn(messages)
                if text is None:
                    yield Rejected(rel, row, "no_user_turn", line[:200])
                    continue
                core = target.get("core_type")
                yield _record(
                    entry,
                    rel=rel,
                    key=split,
                    row=row,
                    split=split,
                    record_type=RecordType.DIALOGUE,
                    lang="en",
                    text=text,
                    response=reply,
                    intent=core if isinstance(core, str) and core else None,
                    extra=dict(obj),  # messages, target, conv_id, *_meta
                )


# ---------------------------------------------------------------- Belebele (JSONL MCQ)

_BELEBELE_LANG: dict[str, LanguageCode] = {"kaz_Cyrl": "kk", "rus_Cyrl": "ru", "eng_Latn": "en"}
_BELEBELE_REQUIRED = (
    "question",
    "flores_passage",
    "correct_answer_num",
    *(f"mc_answer{i}" for i in range(1, 5)),
)


def convert_belebele(raw: Path, entry: SourceEntry) -> Iterator[Row]:
    for path in _files(raw, "data/*.jsonl"):
        rel = path.relative_to(raw).as_posix()
        lang = _BELEBELE_LANG.get(path.stem)
        if lang is None:
            raise ConversionError(f"{rel}: language {path.stem!r} is not mapped")
        with path.open(encoding="utf-8") as fh:
            for row, line in enumerate(fh):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield Rejected(rel, row, "invalid_json", line[:200])
                    continue
                absent = [k for k in _BELEBELE_REQUIRED if _blank(obj.get(k))]
                if absent:
                    yield Rejected(rel, row, f"missing_field:{absent[0]}", line[:200])
                    continue
                yield _record(
                    entry,
                    rel=rel,
                    key=path.stem,
                    row=row,
                    split="test",
                    record_type=RecordType.READING_COMPREHENSION,
                    lang=lang,
                    text=obj["question"],
                    extra={k: v for k, v in obj.items() if k != "question"},
                )


# ---------------------------------------------------------------- Support-JSON-RU (Parquet)

_SJR_REQUIRED = {"id", "split", "message", "category", "reply"}


def convert_support_json_ru(raw: Path, entry: SourceEntry) -> Iterator[Row]:
    for path in _files(raw, "data/selected/*.parquet"):
        rel = path.relative_to(raw).as_posix()
        pf = pq.ParquetFile(path)
        missing = _SJR_REQUIRED - set(pf.schema_arrow.names)
        if missing:
            raise ConversionError(f"{rel}: missing columns {sorted(missing)}")
        row = 0
        for batch in pf.iter_batches(batch_size=1000):
            for rec in batch.to_pylist():
                text = rec.get("message")
                if _blank(text):
                    yield Rejected(rel, row, "empty_text")
                else:
                    yield _record(
                        entry,
                        rel=rel,
                        key=path.stem,
                        row=row,
                        split=rec.get("split") or path.stem,
                        record_type=RecordType.POLICY_DECISION,
                        lang="ru",
                        text=str(text),
                        response=_empty_to_none(rec.get("reply")),
                        category=_empty_to_none(rec.get("category")),
                        extra={k: v for k, v in rec.items() if k not in ("message", "reply")},
                    )
                row += 1


CONVERTERS: dict[str, Converter] = {
    "bitext_retail_ecom": Converter("bitext_csv", 1, convert_bitext),
    "bitext_customer_support": Converter("bitext_csv", 1, convert_bitext),
    "whileai_ecommerce_intent": Converter("whileai_jsonl", 1, convert_whileai),
    "belebele": Converter("belebele_jsonl", 1, convert_belebele),
    "support_json_ru": Converter("support_json_ru_parquet", 1, convert_support_json_ru),
}
