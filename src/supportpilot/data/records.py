"""Read converted raw records for a stated purpose (the only sanctioned way to load them).

The registry's `allowed_use` is checked for every requested source, so an `eval_only` or
`reference_only` source can never reach a training dataset by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import polars as pl

from supportpilot.common.schemas import SourceRegistry
from supportpilot.data.convert import RECORDS, records_dir, verify_records
from supportpilot.data.converters import ConversionError
from supportpilot.data.policy import Purpose, check_use


def scan_records(
    registry: SourceRegistry,
    data_dir: Path,
    source_ids: Sequence[str],
    purpose: Purpose,
    *,
    verify: bool = False,
) -> pl.LazyFrame:
    """Lazy frame over the records of `source_ids`; raises PolicyError if any is not allowed."""
    if not source_ids:
        raise ValueError("no sources requested")
    frames: list[pl.LazyFrame] = []
    for sid in source_ids:
        entry = registry.get(sid)
        check_use(entry, purpose)
        out = records_dir(data_dir, entry)
        if not (out / RECORDS).is_file():
            raise ConversionError(f"{sid}: no converted records at {out}")
        if verify and (problems := verify_records(out)):
            raise ConversionError(f"{sid}: records fail verification: {problems[:3]}")
        frames.append(pl.scan_parquet(out / RECORDS))
    return pl.concat(frames, how="vertical")
