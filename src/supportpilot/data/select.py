"""Stage 4 step 1: v1 relabelling and selection of the English backbone (ADR-023, ADR-026).

Input: the Stage 3 build (`data/interim/canonical/`), read-only. Rows come from
`examples/` (canonical) and from `pending.parquet` with reason `intent_unmapped` (labels that
had no v0 intent; v1 maps most of them). Only `train` sources are read (usage policy).
Rows pending for `label_conflict` or `label_suspect` stay out: their labels are not trusted.

Selection fights the Stage 3 findings:
- one row per paraphrase family (`max_per_family`): families are near-copies (J >= 0.8);
- at most `max_per_label` rows per source label and action;
- inserted profanity (49% of retail rows) capped at `max_profane_share` of each label;
- a family whose members get different v1 labels (rule split) is dropped, never guessed.
`mt_ru` / `mt_kk` mark `mt_per_label` rows per label for translation; RU and KK use
independent draws, so the two languages cover different families.

Output `data/interim/stage4/selected/backbone.parquet` (+ manifest).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa

from supportpilot.common.config import Stage4Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.build import EXAMPLES, PENDING, canonical_dir
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

SELECT_VERSION = 1
BACKBONE = "backbone.parquet"
EXCLUDED = "excluded.parquet"
BACKBONE_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("scenario_id", pa.string()),
        ("family_id", pa.string()),
        ("stage3_scenario_id", pa.string()),
        ("text", pa.string()),
        ("slots_json", pa.string()),
        ("source_intent", pa.string()),
        ("intent", pa.string()),
        ("action", pa.string()),
        ("escalate", pa.bool_()),
        ("escalation_reason", pa.string()),
        ("rule", pa.string()),
        ("offensive", pa.bool_()),
        ("lid_conf", pa.float64()),
        ("stage3_flags", pa.list_(pa.string())),
        ("selected_en", pa.bool_()),
        ("mt_ru", pa.bool_()),
        ("mt_kk", pa.bool_()),
    ]
)
EXCLUDED_SCHEMA = pa.schema(
    [
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_intent", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)


def selected_dir(data_dir: Path) -> Path:
    return data_dir / "interim" / "stage4" / "selected"


def select_params(cfg: Stage4Settings, taxonomy_path: Path) -> dict[str, Any]:
    return {
        "select_version": SELECT_VERSION,
        "taxonomy_sha256": sha256_text(taxonomy_path.read_text(encoding="utf-8")),
        "max_per_family": cfg.max_per_family,
        "max_per_label": cfg.max_per_label,
        "max_profane_share": cfg.max_profane_share,
        "mt_per_label": cfg.mt_per_label,
        "seed": cfg.seed,
    }


def _load(data_dir: Path, entries: Sequence[SourceEntry]) -> tuple[pl.DataFrame, str]:
    can_dir = canonical_dir(data_dir)
    if not (can_dir / "manifest.json").is_file():
        raise StepError(f"no Stage 3 build at {can_dir}; run `supportpilot data clean --all`")
    version = read_manifest(can_dir).dataset_version
    train_ids: list[str] = []
    for e in entries:
        try:
            check_use(e, Purpose.TRAINING)
        except Exception:  # noqa: S112 - non-train sources are simply not read
            continue
        train_ids.append(e.source_id)
    flag = pl.col("quality").struct.field("flags")
    lid = pl.col("quality").struct.field("lid_conf")
    cols = [
        pl.col("example_id"),
        pl.col("record_id"),
        pl.col("source_id"),
        pl.col("scenario_id"),
        pl.col("family_id"),
        pl.col("text"),
        pl.col("slots"),
        pl.col("source_intent"),
        flag.alias("stage3_flags"),
        lid.alias("lid_conf"),
    ]
    can = pl.scan_parquet(can_dir / EXAMPLES / "lang=en" / "*.parquet").select(cols)
    pend = (
        pl.scan_parquet(can_dir / PENDING)
        .filter((pl.col("reason") == "intent_unmapped") & (pl.col("lang") == "en"))
        .select(cols)
    )
    df = pl.concat([can, pend]).filter(pl.col("source_id").is_in(train_ids)).collect()
    return df.sort("source_id", "record_id"), version


def _slots(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    return {str(d["key"]): str(d["value"]) for d in value}


def _rank(*parts: object) -> str:
    return sha256_text("|".join(map(str, parts)))


def build_backbone(
    entries: Sequence[SourceEntry],
    data_dir: Path,
    cfg: Stage4Settings,
    tax: Taxonomy,
    params: dict[str, Any],
    *,
    code_commit: str | None,
    force: bool = False,
) -> dict[str, Any]:
    df, can_version = _load(data_dir, entries)
    inputs = [f"canonical@{can_version}"]
    out = selected_dir(data_dir)
    if not force and is_current(out, inputs, params):
        m = read_manifest(out)
        return {"status": "skipped", "dataset_version": m.dataset_version, "rows": m.row_counts}

    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for r in df.iter_rows(named=True):
        slots = _slots(r["slots"])
        lab = relabel(tax, r["source_id"], r["source_intent"], r["text"], slots)
        if lab.status != "mapped":
            excluded.append(
                {
                    "example_id": r["example_id"],
                    "record_id": r["record_id"],
                    "source_id": r["source_id"],
                    "source_intent": r["source_intent"],
                    "reason": lab.status,
                    "detail": lab.detail,
                }
            )
            continue
        flags = list(r["stage3_flags"] or [])
        rows.append(
            {
                "example_id": r["example_id"],
                "record_id": r["record_id"],
                "source_id": r["source_id"],
                "stage3_scenario_id": r["scenario_id"],
                "family_id": r["family_id"],
                "text": r["text"],
                "slots_json": json.dumps(slots, sort_keys=True, ensure_ascii=False),
                "source_intent": r["source_intent"],
                "intent": lab.intent,
                "action": lab.action,
                "escalate": lab.escalate,
                "escalation_reason": lab.escalation_reason,
                "rule": lab.rule,
                "offensive": "offensive" in flags,
                "lid_conf": r["lid_conf"],
                "stage3_flags": flags,
            }
        )

    # A family split by a rule into different v1 labels is not trusted.
    fam_labels: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        fam_labels[row["family_id"]].add((row["intent"], row["action"]))
    bad = {f for f, labs in fam_labels.items() if len(labs) > 1}
    kept: list[dict[str, Any]] = []
    for row in rows:
        if row["family_id"] in bad:
            excluded.append(
                {
                    "example_id": row["example_id"],
                    "record_id": row["record_id"],
                    "source_id": row["source_id"],
                    "source_intent": row["source_intent"],
                    "reason": "family_label_conflict",
                    "detail": "members of one paraphrase family received different v1 labels",
                }
            )
        else:
            kept.append(row)

    # v1 scenario: keep the Stage 3 id when that scenario maps to one (intent, action).
    scn_labels: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in kept:
        scn_labels[row["stage3_scenario_id"]].add((row["intent"], row["action"]))
    for row in kept:
        old = row["stage3_scenario_id"]
        if len(scn_labels[old]) == 1:
            row["scenario_id"] = old
        else:
            row["scenario_id"] = "scn_" + sha256_text(f"{old}|{row['intent']}|{row['action']}")[:16]

    _select(kept, cfg)
    kept.sort(key=lambda r: (r["source_id"], r["record_id"]))
    excluded.sort(key=lambda r: (r["source_id"], r["record_id"]))
    selected = [r for r in kept if r["selected_en"]]
    row_counts = {
        "relabelled": len(kept),
        "excluded": len(excluded),
        "selected_en": len(selected),
        "mt_ru": sum(r["mt_ru"] for r in kept),
        "mt_kk": sum(r["mt_kk"] for r in kept),
        **counts(excluded, "reason", prefix="excluded|"),
        **counts(selected, "intent", prefix="selected_en|"),
    }
    with partial_dir(out) as partial:
        write_parquet(partial / BACKBONE, kept, BACKBONE_SCHEMA)
        write_parquet(partial / EXCLUDED, excluded, EXCLUDED_SCHEMA)
        m = commit(
            partial,
            out,
            name="backbone",
            data_schema_version="stage4_backbone/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts=row_counts,
            content_sha256=content_hash(kept),
        )
    return {"status": "built", "dataset_version": m.dataset_version, "rows": m.row_counts}


def _select(rows: list[dict[str, Any]], cfg: Stage4Settings) -> None:
    """Set selected_en / mt_ru / mt_kk in place (deterministic: sha256 ranks, no RNG state)."""
    by_label: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        r["selected_en"] = r["mt_ru"] = r["mt_kk"] = False
        by_label[(r["source_id"], r["source_intent"], r["action"])].append(r)
    for key, members in sorted(by_label.items()):
        # one representative per family; prefer the non-profane register
        per_fam: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in members:
            per_fam[r["family_id"]].append(r)
        reps: list[dict[str, Any]] = []
        for _fam, items in sorted(per_fam.items()):
            items.sort(key=lambda r: (r["offensive"], _rank(cfg.seed, "fam", r["example_id"])))
            reps.extend(items[: cfg.max_per_family])
        reps.sort(key=lambda r: _rank(cfg.seed, "label", *key, r["example_id"]))
        clean = [r for r in reps if not r["offensive"]]
        profane = [r for r in reps if r["offensive"]]
        n_clean = min(len(clean), cfg.max_per_label)
        n_prof = min(
            len(profane), int(cfg.max_profane_share * n_clean / (1 - cfg.max_profane_share))
        )
        chosen = clean[:n_clean] + profane[:n_prof]
        for r in chosen:
            r["selected_en"] = True
        for lang in ("ru", "kk"):
            order = sorted(reps, key=lambda r: _rank(cfg.seed, "mt", lang, r["example_id"]))
            prof_budget = int(cfg.max_profane_share * cfg.mt_per_label)
            taken = 0
            for r in order:
                if taken >= cfg.mt_per_label:
                    break
                if r["offensive"]:
                    if prof_budget <= 0:
                        continue
                    prof_budget -= 1
                r[f"mt_{lang}"] = True
                taken += 1


def profile(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c: Counter[str] = Counter(r["intent"] for r in rows if r["selected_en"])
    return {"intents": len(c), "min": min(c.values(), default=0), "max": max(c.values(), default=0)}
