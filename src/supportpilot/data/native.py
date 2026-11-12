"""Stage 4 step: native-language rows of licensed sources (ADR-028).

Some train sources are written (or human-localised) in the target languages themselves:
`mangi_kk_support_qa` (Kazakh) and `massive_en_ru` (English, Russian). They are not
machine-translated and do not join the English backbone or the MT queue (select.py); this step
relabels their Stage 3 rows with taxonomy v1 (`configs/taxonomy.yaml` `sources:`) and keeps:

- only rows whose label maps (excluded / unlabelled rows are written to `excluded.parquet`
  with the reason, never guessed);
- one row per Stage 3 paraphrase family;
- only rows whose Stage 3 language is one the registry lists for the source;
- at most `native_max_per_label[source]` (default 8) rows per (source, source label, language),
  by sha256 rank.

Output `data/interim/stage4/native/native.parquet` (`stage4_native/1`) + manifest. Every row
keeps its Stage 3 lineage (`example_id`, `record_id`, `family_id`) and goes to train
(source-derived, ADR-026).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from supportpilot.common.config import Stage4Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.build import PENDING, canonical_dir
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.steps import (
    StepError,
    commit,
    content_hash,
    counts,
    is_current,
    partial_dir,
    read_manifest,
    write_parquet,
)
from supportpilot.data.taxonomy import Taxonomy, relabel

NATIVE_VERSION = 2
NATIVE = "native.parquet"
EXCLUDED = "excluded.parquet"
# Stage 3 leaves rows of labels without a v0 intent pending with one of these reasons.
_PENDING_REASONS = ("intent_unmapped", "intent_not_in_map")
NATIVE_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("lang", pa.string()),
        ("code_switch", pa.bool_()),
        ("text", pa.string()),
        ("source_intent", pa.string()),
        ("intent", pa.string()),
        ("action", pa.string()),
        ("escalation_reason", pa.string()),
        ("rule", pa.string()),
        ("lid_conf", pa.float64()),
        ("stage3_flags", pa.list_(pa.string())),
    ]
)
EXCLUDED_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_intent", pa.string()),
        ("lang", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)


def native_dir(data_dir: Path) -> Path:
    return data_dir / "interim" / "stage4" / "native"


def native_params(cfg: Stage4Settings, taxonomy_path: Path) -> dict[str, Any]:
    return {
        "native_version": NATIVE_VERSION,
        "taxonomy_sha256": sha256_text(taxonomy_path.read_text(encoding="utf-8")),
        "native_sources": sorted(cfg.native_sources),
        "native_max_per_label": dict(sorted(cfg.native_max_per_label.items())),
        "seed": cfg.seed,
    }


def _rank(*parts: object) -> str:
    return sha256_text("|".join(map(str, parts)))


DEFAULT_CAP = 8


def _load(
    data_dir: Path, entries: Sequence[SourceEntry], wanted: Sequence[str]
) -> tuple[pl.DataFrame, str, dict[str, set[str]]]:
    can_dir = canonical_dir(data_dir)
    if not (can_dir / "manifest.json").is_file():
        raise StepError(f"no Stage 3 build at {can_dir}; run `supportpilot data clean --all`")
    ids = []
    langs: dict[str, set[str]] = {}
    for e in entries:
        if e.source_id in wanted:
            check_use(e, Purpose.TRAINING)  # a native source must be train-allowed
            ids.append(e.source_id)
            langs[e.source_id] = set(e.languages)
    if missing := sorted(set(wanted) - set(ids)):
        raise StepError(f"native sources not selected in the registry: {missing}")
    df = (
        pl.scan_parquet(can_dir / PENDING)
        .filter(pl.col("source_id").is_in(ids) & pl.col("reason").is_in(_PENDING_REASONS))
        .select(
            "example_id",
            "record_id",
            "source_id",
            "scenario_id",
            "family_id",
            "lang",
            "code_switch",
            "text",
            "source_intent",
            pl.col("quality").struct.field("lid_conf").alias("lid_conf"),
            pl.col("quality").struct.field("flags").alias("stage3_flags"),
        )
        .collect()
        .sort("source_id", "record_id")
    )
    return df, read_manifest(can_dir).dataset_version, langs


def build_native(
    entries: Sequence[SourceEntry],
    data_dir: Path,
    cfg: Stage4Settings,
    tax: Taxonomy,
    params: dict[str, Any],
    *,
    code_commit: str | None,
    force: bool = False,
) -> dict[str, Any]:
    df, can_version, langs = _load(data_dir, entries, cfg.native_sources)
    inputs = [
        f"canonical@{can_version}",
        "registry@"
        + sha256_text(
            "".join(
                e.model_dump_json()
                for e in sorted(entries, key=lambda e: e.source_id)
                if e.source_id in cfg.native_sources
            )
        ),
    ]
    out = native_dir(data_dir)
    if not force and is_current(out, inputs, params):
        m = read_manifest(out)
        return {"status": "skipped", "dataset_version": m.dataset_version, "rows": m.row_counts}

    mapped: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []

    def drop(r: dict[str, Any], reason: str, detail: str | None) -> None:
        excluded.append(
            {k: r[k] for k in ("example_id", "record_id", "source_id", "source_intent", "lang")}
            | {"reason": reason, "detail": detail}
        )

    for r in df.iter_rows(named=True):
        if r["lang"] not in langs[r["source_id"]]:
            drop(r, "language", f"Stage 3 language {r['lang']} is not a source language")
            continue
        lab = relabel(tax, r["source_id"], r["source_intent"], r["text"], {})
        if lab.status != "mapped":
            drop(r, lab.status, lab.detail)
            continue
        mapped.append(
            {
                **r,
                "stage3_scenario_id": r["scenario_id"],
                "intent": lab.intent,
                "action": lab.action,
                "escalation_reason": lab.escalation_reason,
                "rule": lab.rule,
                "stage3_flags": list(r["stage3_flags"] or []),
            }
        )

    # v1 scenario: the Stage 3 scenario when it maps to one (intent, action), else split.
    scn: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for r in mapped:
        scn[r["stage3_scenario_id"]].add((r["intent"], r["action"]))
    for r in mapped:
        old = r["stage3_scenario_id"]
        r["scenario_id"] = (
            old
            if len(scn[old]) == 1
            else "scn_" + sha256_text(f"{old}|{r['intent']}|{r['action']}")[:16]
        )

    # one row per paraphrase family, then a cap per source label and language
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in mapped:
        by_family[r["family_id"]].append(r)
    reps = []
    for _fam, members in sorted(by_family.items()):
        members.sort(key=lambda r: _rank(cfg.seed, "native-family", r["example_id"]))
        reps.append(members[0])
        for r in members[1:]:
            drop(r, "family_cap", "one row per paraphrase family")
    by_label: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in reps:
        by_label[(r["source_id"], r["source_intent"], r["lang"])].append(r)
    kept: list[dict[str, Any]] = []
    for _key, members in sorted(by_label.items()):
        cap = cfg.native_max_per_label.get(_key[0], DEFAULT_CAP)
        members.sort(key=lambda r: _rank(cfg.seed, "native", r["example_id"]))
        kept.extend(members[:cap])
        for r in members[cap:]:
            drop(r, "label_cap", f"more than {cap} per label and language")

    kept.sort(key=lambda r: (r["source_id"], r["record_id"]))
    excluded.sort(key=lambda r: (r["source_id"], r["record_id"]))
    rows = [{k: r[k] for k in NATIVE_SCHEMA.names} for r in kept]
    row_counts = {
        "kept": len(rows),
        "excluded": len(excluded),
        **counts(rows, "source_id", "lang", "intent", prefix="kept|"),
        **counts(excluded, "source_id", "reason", prefix="excluded|"),
    }
    with partial_dir(out) as partial:
        write_parquet(partial / NATIVE, rows, NATIVE_SCHEMA)
        write_parquet(partial / EXCLUDED, excluded, EXCLUDED_SCHEMA)
        m = commit(
            partial,
            out,
            name="native",
            data_schema_version="stage4_native/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts=row_counts,
            content_sha256=content_hash(rows),
        )
    return {"status": "built", "dataset_version": m.dataset_version, "rows": m.row_counts}
