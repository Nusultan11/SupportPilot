"""Stage 4 entry points used by the CLI (`supportpilot data select|native|kb-train|mt-compare|
translate|qe-score|sft|gold|taxonomy`). Each step is checkpointed and skipped when current."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import polars as pl

from supportpilot.common.config import Settings
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.kbtrain import build_kb_train
from supportpilot.data.locale import load_locale
from supportpilot.data.native import build_native, native_params
from supportpilot.data.select import BACKBONE, build_backbone, select_params, selected_dir
from supportpilot.data.taxonomy import load_taxonomy


def run_select(
    settings: Settings, entries: Sequence[SourceEntry], *, code_commit: str | None, force: bool
) -> dict[str, Any]:
    cfg = settings.stage4
    tax_path = settings.resolve(cfg.taxonomy)
    return build_backbone(
        entries,
        settings.resolve(settings.paths.data_dir),
        cfg,
        load_taxonomy(tax_path),
        select_params(cfg, tax_path),
        code_commit=code_commit,
        force=force,
    )


def run_native(
    settings: Settings, entries: Sequence[SourceEntry], *, code_commit: str | None, force: bool
) -> dict[str, Any]:
    cfg = settings.stage4
    tax_path = settings.resolve(cfg.taxonomy)
    return build_native(
        entries,
        settings.resolve(settings.paths.data_dir),
        cfg,
        load_taxonomy(tax_path),
        native_params(cfg, tax_path),
        code_commit=code_commit,
        force=force,
    )


def run_kb(settings: Settings, *, code_commit: str | None, force: bool) -> dict[str, Any]:
    cfg = settings.stage4
    return build_kb_train(
        settings.resolve(cfg.authored_dir),
        settings.resolve(cfg.demo_facts),
        load_locale(settings.resolve(cfg.locale)),
        settings.resolve(cfg.locale),
        settings.resolve(settings.paths.data_dir),
        code_commit=code_commit,
        force=force,
    )


def taxonomy_report(settings: Settings) -> dict[str, Any]:
    """Validate the v1 taxonomy and summarise how the backbone was relabelled."""
    tax = load_taxonomy(settings.resolve(settings.stage4.taxonomy))
    out: dict[str, Any] = {
        "version": tax.version,
        "intents": len(tax.intents),
        "by_group": dict(sorted(Counter(i.group for i in tax.intents.values()).items())),
        "by_action": dict(sorted(Counter(i.action for i in tax.intents.values()).items())),
        "escalation_reasons": sorted(tax.escalation_reasons),
        "source_labels": {
            sid: {
                "mapped": sum(1 for m in labels.values() if m.intent),
                "rules": sum(1 for m in labels.values() if m.rules),
                "excluded": sum(1 for m in labels.values() if m.exclude),
            }
            for sid, labels in tax.sources.items()
        },
    }
    path = selected_dir(settings.resolve(settings.paths.data_dir)) / BACKBONE
    if path.is_file():
        bb = pl.read_parquet(path)
        out["backbone_rows"] = bb.height
        out["backbone_rules_used"] = dict(
            Counter(r for r in bb["rule"].to_list() if r is not None).most_common()
        )
    return out
