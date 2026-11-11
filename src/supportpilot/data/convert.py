"""Convert an acquired raw revision into `data/interim/records/{source_id}/{revision}/`.

Output: `records.parquet` (RawRecord rows, streamed in batches), `rejected.jsonl` (rows the
converter refused, with reasons) and `manifest.json` (RecordsManifest).

- Idempotent: skipped when the manifest matches the current raw manifest, converter version
  and file hashes. `force=True` rebuilds.
- Crash-safe: built under `.partial-*`, then swapped in; stale partials are removed first.
- Quality gate: fails if rejected rows exceed `max_reject_ratio` or no rows are produced.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections import Counter
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TextIO

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from supportpilot.common.hashing import sha256_file
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import (
    FileProblem,
    read_model_json,
    scan_files,
    verify_files,
    write_model_json,
)
from supportpilot.common.schemas import RawRecord, RecordsManifest, SourceEntry
from supportpilot.data.acquire import MANIFEST, raw_dir, verify_raw
from supportpilot.data.converters import CONVERTERS, ConversionError, Rejected, Row
from supportpilot.data.policy import Purpose, check_use

log = get_logger(__name__)
RECORDS = "records.parquet"
REJECTED = "rejected.jsonl"
BATCH = 5000
NL = chr(10)  # LF on every platform

ARROW_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("revision", pa.string()),
        ("source_file", pa.string()),
        ("source_row", pa.int64()),
        ("source_split", pa.string()),
        ("record_type", pa.string()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("response", pa.string()),
        ("source_intent", pa.string()),
        ("source_category", pa.string()),
        ("source_tags", pa.string()),
        ("extra_json", pa.string()),
        ("text_sha256", pa.string()),
    ]
)


class ConvertResult(BaseModel):
    source_id: str
    revision: str
    status: Literal["converted", "skipped"]
    path: str
    rows: int
    rejected: int
    seconds: float


def records_dir(data_dir: Path, entry: SourceEntry) -> Path:
    return data_dir / "interim" / "records" / entry.source_id / entry.revision


def verify_records(path: Path) -> list[FileProblem]:
    if not (path / MANIFEST).is_file():
        return [FileProblem(path=MANIFEST, problem="missing")]
    m = read_model_json(path / MANIFEST, RecordsManifest)
    return verify_files(path, m.files, exclude=frozenset({MANIFEST}))


def _is_current(out: Path, raw_manifest_sha: str, version: int) -> bool:
    if not (out / MANIFEST).is_file():
        return False
    m = read_model_json(out / MANIFEST, RecordsManifest)
    return (
        m.raw_manifest_sha256 == raw_manifest_sha
        and m.converter_version == version
        and not verify_records(out)
    )


def convert_source(
    entry: SourceEntry,
    data_dir: Path,
    *,
    max_reject_ratio: float,
    code_commit: str | None = None,
    force: bool = False,
) -> ConvertResult:
    check_use(entry, Purpose.CONVERT)
    converter = CONVERTERS.get(entry.source_id)
    if converter is None:
        raise ConversionError(f"{entry.source_id}: no converter registered")
    raw = raw_dir(data_dir, entry)
    if not (raw / MANIFEST).is_file():
        raise ConversionError(f"{entry.source_id}: not acquired ({raw})")
    problems = verify_raw(raw)
    if problems:
        raise ConversionError(f"{entry.source_id}: raw data fails verification: {problems[:3]}")

    start = time.perf_counter()
    raw_manifest_sha = sha256_file(raw / MANIFEST)
    out = records_dir(data_dir, entry)
    if not force and _is_current(out, raw_manifest_sha, converter.version):
        m = read_model_json(out / MANIFEST, RecordsManifest)
        return ConvertResult(
            source_id=entry.source_id,
            revision=entry.revision,
            status="skipped",
            path=str(out),
            rows=m.row_count,
            rejected=m.rejected_count,
            seconds=round(time.perf_counter() - start, 2),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob(".partial-*"):
        shutil.rmtree(stale, ignore_errors=True)
    partial = out.parent / f".partial-{entry.revision}-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        rows, rejected, counts, reasons, content = _write(partial, converter.convert(raw, entry))
        total = rows + rejected
        if rows == 0:
            raise ConversionError(f"{entry.source_id}: converter produced no records")
        if rejected / total > max_reject_ratio:
            raise ConversionError(
                f"{entry.source_id}: {rejected}/{total} rows rejected "
                f"(> {max_reject_ratio:.2%}); reasons {dict(reasons)}"
            )
        manifest = RecordsManifest(
            source_id=entry.source_id,
            revision=entry.revision,
            allowed_use=entry.allowed_use,
            converter=converter.name,
            converter_version=converter.version,
            raw_manifest_sha256=raw_manifest_sha,
            code_commit=code_commit,
            created_at=datetime.now(UTC),
            row_count=rows,
            rejected_count=rejected,
            rejected_reasons=dict(sorted(reasons.items())),
            counts=dict(sorted(counts.items())),
            content_sha256=content,
            files=scan_files(partial),
        )
        write_model_json(partial / MANIFEST, manifest)
        old = out.parent / f".old-{uuid.uuid4().hex[:8]}"
        if out.exists():
            out.rename(old)
        partial.rename(out)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    log.info("source converted", extra={"source_id": entry.source_id, "rows": rows})
    return ConvertResult(
        source_id=entry.source_id,
        revision=entry.revision,
        status="converted",
        path=str(out),
        rows=rows,
        rejected=rejected,
        seconds=round(time.perf_counter() - start, 2),
    )


def _write(
    partial: Path, rows_iter: Iterable[Row]
) -> tuple[int, int, Counter[str], Counter[str], str]:
    """Stream rows to Parquet / rejected.jsonl; returns counts and the content hash."""
    content = hashlib.sha256()
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    seen: set[str] = set()
    batch: list[dict[str, object]] = []
    n_ok = n_bad = 0
    with ExitStack() as stack:
        writer = stack.enter_context(
            pq.ParquetWriter(partial / RECORDS, ARROW_SCHEMA, compression="zstd")
        )
        rejected_fh: TextIO | None = None
        for item in rows_iter:
            if isinstance(item, RawRecord) and item.record_id in seen:
                item = Rejected(item.source_file, item.source_row, "duplicate_record_id")
            if isinstance(item, Rejected):
                if rejected_fh is None:
                    rejected_fh = stack.enter_context(
                        (partial / REJECTED).open("w", encoding="utf-8", newline=NL)
                    )
                rejected_fh.write(json.dumps(asdict(item), ensure_ascii=False) + NL)
                reasons[item.reason.split(":")[0]] += 1
                n_bad += 1
                continue
            seen.add(item.record_id)
            row = item.model_dump(mode="json")
            content.update((json.dumps(row, ensure_ascii=False, sort_keys=True) + NL).encode())
            counts[f"lang={item.lang}"] += 1
            counts[f"type={item.record_type.value}"] += 1
            counts[f"split={item.source_split or 'none'}"] += 1
            batch.append(row)
            n_ok += 1
            if len(batch) >= BATCH:
                writer.write_table(pa.Table.from_pylist(batch, schema=ARROW_SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=ARROW_SCHEMA))
    return n_ok, n_bad, counts, reasons, content.hexdigest()
