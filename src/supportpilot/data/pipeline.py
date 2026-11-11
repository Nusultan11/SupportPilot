"""Stage 3 orchestration used by the CLI: clean every source, build, run quality gates."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import psutil
from pydantic import BaseModel

from supportpilot.common.config import Settings
from supportpilot.common.logging import get_logger
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.build import (
    QUALITY,
    BuildResult,
    build_canonical,
    build_params,
    canonical_dir,
)
from supportpilot.data.clean import CleanResult, clean_params, clean_source
from supportpilot.data.gates import GateReport, run_gates
from supportpilot.data.labels import load_intent_map
from supportpilot.data.placeholders import load_catalog

log = get_logger(__name__)
# Peak RSS measured for the full build (~100k records, 96k distinct texts) was ~2.5 GB.
MIN_FREE_RAM_GB = 3.0


class PrepareResult(BaseModel):
    cleaned: list[CleanResult]
    build: BuildResult | None
    gates: GateReport | None


def _check_memory() -> None:
    free = psutil.virtual_memory().available / 2**30
    if free < MIN_FREE_RAM_GB:
        log.warning(
            "low free RAM; the build may swap (close Docker/IDE or raise the page file)",
            extra={"free_gib": round(free, 2), "recommended_gib": MIN_FREE_RAM_GB},
        )


def prepare(
    settings: Settings,
    entries: Sequence[SourceEntry],
    *,
    build: bool,
    force: bool,
    code_commit: str | None,
) -> PrepareResult:
    data_dir = settings.resolve(settings.paths.data_dir)
    cfg = settings.clean
    catalog_path = settings.resolve(cfg.placeholders)
    catalog = load_catalog(catalog_path)
    imap = load_intent_map(settings.resolve(cfg.intent_map))
    params = clean_params(cfg, catalog_path)
    cleaned = [
        clean_source(e, data_dir, cfg, catalog, params, code_commit=code_commit, force=force)
        for e in entries
    ]
    if not build:
        return PrepareResult(cleaned=cleaned, build=None, gates=None)
    _check_memory()
    report: dict[str, GateReport] = {}

    def gates(partial: Path) -> tuple[bool, dict[str, object]]:
        rep = run_gates(partial, entries, data_dir, cfg, catalog)
        report["r"] = rep
        return rep.passed, rep.model_dump(mode="json")

    result = build_canonical(
        entries,
        data_dir,
        cfg,
        imap,
        build_params(cfg, settings.resolve(cfg.intent_map)),
        code_commit=code_commit,
        force=force,
        run_gates=gates,
    )
    rep = report.get("r") or GateReport.model_validate(
        json.loads((canonical_dir(data_dir) / QUALITY).read_text(encoding="utf-8"))
    )
    return PrepareResult(cleaned=cleaned, build=result, gates=rep)


def regate(settings: Settings, entries: Sequence[SourceEntry]) -> GateReport:
    """Re-run the gates on the current build (no rebuild)."""
    data_dir = settings.resolve(settings.paths.data_dir)
    catalog = load_catalog(settings.resolve(settings.clean.placeholders))
    out = canonical_dir(data_dir)
    if not (out / QUALITY).is_file():
        raise FileNotFoundError(f"no build at {out}; run `data clean --all` first")
    return run_gates(out, entries, data_dir, settings.clean, catalog)
