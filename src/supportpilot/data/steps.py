"""Checkpointed pipeline steps for Stage 4 (restartable, atomic, skip when current).

A step writes into `<out>.partial-<id>/`, writes `manifest.json` (dataset_manifest/1) last,
and is swapped in with a rename, so a killed run never leaves a half-written output in
place; leftovers are removed by the next run. A step is skipped when its manifest records
the same inputs and parameters and every file still matches its sha256.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from supportpilot.common.ids import make_dataset_version
from supportpilot.common.manifest import read_model_json, scan_files, verify_files, write_model_json
from supportpilot.common.schemas import DatasetManifest

MANIFEST = "manifest.json"


def jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def params_sha(params: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(jsonable(params), sort_keys=True).encode()).hexdigest()


def content_hash(rows: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        h.update((json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n").encode())
    return h.hexdigest()


def is_current(out: Path, inputs: list[str], params: Mapping[str, Any]) -> bool:
    if not (out / MANIFEST).is_file():
        return False
    m = read_model_json(out / MANIFEST, DatasetManifest)
    return (
        m.inputs == inputs
        and m.params == jsonable(params)
        and not verify_files(out, m.files, exclude=frozenset({MANIFEST}))
    )


def read_manifest(out: Path) -> DatasetManifest:
    return read_model_json(out / MANIFEST, DatasetManifest)


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema) if rows else schema.empty_table()
    pq.write_table(table, path, compression="zstd")


def counts(rows: list[dict[str, Any]], *keys: str, prefix: str = "") -> dict[str, int]:
    c: Counter[str] = Counter()
    for r in rows:
        c["|".join(f"{k}={r.get(k)}" for k in keys)] += 1
    return {f"{prefix}{k}": v for k, v in sorted(c.items())}


@contextmanager
def partial_dir(out: Path) -> Iterator[Path]:
    """Yield a fresh partial directory next to `out`; on success the caller calls `commit`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    for stale in out.parent.glob(f".partial-{out.name}-*"):
        shutil.rmtree(stale, ignore_errors=True)
    partial = out.parent / f".partial-{out.name}-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        yield partial
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def commit(
    partial: Path,
    out: Path,
    *,
    name: str,
    data_schema_version: str,
    code_commit: str | None,
    inputs: list[str],
    params: Mapping[str, Any],
    row_counts: Mapping[str, int],
    content_sha256: str,
    quality_report: str | None = None,
    gate: Callable[[Path], bool] | None = None,
) -> DatasetManifest:
    """Write the manifest into `partial` and swap it in for `out` (atomic rename)."""
    manifest = DatasetManifest(
        dataset_version=make_dataset_version(name, datetime.now(UTC).date(), content_sha256),
        data_schema_version=data_schema_version,
        code_commit=code_commit or "unknown",
        inputs=inputs,
        params=jsonable(params),
        row_counts=dict(sorted(row_counts.items())),
        files=scan_files(partial),
        created_at=datetime.now(UTC),
        quality_report=quality_report,
    )
    write_model_json(partial / MANIFEST, manifest)
    if gate is not None and not gate(partial):
        failed = out.parent / f"{out.name}.failed"
        shutil.rmtree(failed, ignore_errors=True)
        partial.rename(failed)
        raise StepError(f"{name}: quality gates failed; output kept at {failed}")
    old = out.parent / f".old-{out.name}-{uuid.uuid4().hex[:8]}"
    if out.exists():
        out.rename(old)
    partial.rename(out)
    shutil.rmtree(old, ignore_errors=True)
    shutil.rmtree(out.parent / f"{out.name}.failed", ignore_errors=True)
    return manifest


class StepError(Exception):
    """A Stage 4 step cannot run or its output failed its gates."""
