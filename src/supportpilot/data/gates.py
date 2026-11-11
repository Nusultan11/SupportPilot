"""Quality gates over a built `data/interim/canonical/` directory (DATA_CONTRACTS §10).

Gates re-read the written files and re-check them independently of the build code (the
language gate re-runs LID, the id gate recomputes example ids, the PII gate re-scans text).
Each gate reports `pass`, `fail` or `not_applicable` with its metric and threshold.
Split/leakage gates are `not_applicable` until Stage 4 creates splits.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ValidationError

from supportpilot.common.config import CleanSettings
from supportpilot.common.ids import make_example_id
from supportpilot.common.manifest import read_model_json
from supportpilot.common.schemas import (
    INTENTS,
    CanonicalExample,
    PendingExample,
    RecordsManifest,
    RejectedRecord,
    SourceEntry,
)
from supportpilot.data.acquire import MANIFEST
from supportpilot.data.convert import records_dir
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.placeholders import PLACEHOLDER_RE, PlaceholderCatalog
from supportpilot.data.policy import PolicyError, Purpose, check_use
from supportpilot.data.textnorm import CARD_RE, EMAIL_RE, PHONE_RE, luhn_ok

Status = Literal["pass", "fail", "not_applicable"]
_SYNTH_EMAIL = re.compile(r"^customer\d{4}@example\.com$")
_SYNTH_PHONE = re.compile(r"^\+1 202 555 01\d\d$")
_SYNTH_CARD = "4000 0000 0000 0002"


class Gate(BaseModel):
    name: str
    status: Status
    metric: float | int | None = None
    threshold: float | int | None = None
    details: dict[str, Any] = {}


class GateReport(BaseModel):
    passed: bool
    gates: list[Gate]


def _read(build: Path, name: str) -> pl.DataFrame:
    return pl.read_parquet(build / name)


def _canonical(build: Path) -> pl.DataFrame:
    files = sorted((build / "examples").glob("lang=*/*.parquet"))
    if not files:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def slots_dict(value: Any) -> dict[str, str]:
    """Polars reads a Parquet map as a list of {key, value} structs."""
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {kv["key"]: kv["value"] for kv in value}


def _validate(df: pl.DataFrame, model: type[BaseModel]) -> tuple[int, Counter[str]]:
    errors: Counter[str] = Counter()
    ok = 0
    for row in df.iter_rows(named=True):
        if "slots" in row:
            row["slots"] = slots_dict(row["slots"])
        if row.get("created_at") is None:
            row.pop("created_at", None)
        try:
            model.model_validate(row)
            ok += 1
        except ValidationError as exc:
            err = exc.errors()[0]
            errors[f"{'.'.join(str(p) for p in err['loc'])}: {err['type']}"] += 1
    return ok, errors


def g_accounting(build: Path, entries: Sequence[SourceEntry], data_dir: Path) -> Gate:
    expected = 0
    for e in entries:
        expected += read_model_json(records_dir(data_dir, e) / MANIFEST, RecordsManifest).row_count
    ids: list[str] = []
    for part in (
        _canonical(build),
        _read(build, "pending.parquet"),
        _read(build, "rejected.parquet"),
    ):
        if part.height:
            ids.extend(part["record_id"].to_list())
    dupes = len(ids) - len(set(ids))
    ok = len(ids) == expected and dupes == 0
    return Gate(
        name="accounting",
        status="pass" if ok else "fail",
        metric=len(ids),
        threshold=expected,
        details={
            "input_records": expected,
            "output_rows": len(ids),
            "record_in_two_outputs": dupes,
        },
    )


def g_schema(build: Path, min_validity: float) -> Gate:
    can = _canonical(build)
    pen = _read(build, "pending.parquet")
    rej = _read(build, "rejected.parquet")
    c_ok, c_err = _validate(can, CanonicalExample)
    p_ok, p_err = _validate(pen, PendingExample)
    r_ok, r_err = _validate(rej, RejectedRecord)
    validate_rejected = rej.filter(pl.col("stage") == "validate").height if rej.height else 0
    accepted = can.height + pen.height
    valid = c_ok + p_ok
    validity = valid / (accepted + validate_rejected) if accepted + validate_rejected else 0.0
    ok = validity >= min_validity and not c_err and not p_err and not r_err
    return Gate(
        name="schema",
        status="pass" if ok else "fail",
        metric=round(validity, 6),
        threshold=min_validity,
        details={
            "canonical_valid": f"{c_ok}/{can.height}",
            "pending_valid": f"{p_ok}/{pen.height}",
            "rejected_valid": f"{r_ok}/{rej.height}",
            "rejected_at_validation": validate_rejected,
            "errors": dict((c_err + p_err + r_err).most_common(10)),
        },
    )


def g_ids(build: Path) -> Gate:
    can = _canonical(build)
    pen = _read(build, "pending.parquet")
    rej = _read(build, "rejected.parquet")
    groups = _read(build, "groups.parquet")
    kept = pl.concat(
        [can.select("example_id", "record_id"), pen.select("example_id", "record_id")]
        if can.height
        else [pen.select("example_id", "record_id")]
    )
    dup_ids = kept.height - kept["example_id"].n_unique()
    bad_recompute = 0
    recompute = pl.concat(
        [
            p.select("source_id", "record_id", "lang", "text", "example_id")
            for p in (can, pen)
            if p.height
        ]
    )
    for r in recompute.iter_rows(named=True):
        expected = make_example_id(r["source_id"], r["record_id"], r["lang"], "original", r["text"])
        bad_recompute += expected != r["example_id"]
    fam_scn = groups.group_by("family_id").agg(pl.col("scenario_id").n_unique().alias("n"))
    families_in_two_scenarios = fam_scn.filter(pl.col("n") > 1).height
    all_ids = set(groups["example_id"].to_list())
    dangling = 0
    if rej.height:
        dup_of = rej.filter(pl.col("duplicate_of").is_not_null())["duplicate_of"].to_list()
        kept_ids = set(kept["example_id"].to_list())
        dangling = sum(1 for d in dup_of if d not in kept_ids)
    missing_group = sum(1 for e in kept["example_id"].to_list() if e not in all_ids)
    problems = dup_ids + bad_recompute + families_in_two_scenarios + dangling + missing_group
    return Gate(
        name="ids",
        status="pass" if problems == 0 else "fail",
        metric=problems,
        threshold=0,
        details={
            "duplicate_example_ids": dup_ids,
            "example_id_not_recomputable": bad_recompute,
            "families_in_two_scenarios": families_in_two_scenarios,
            "duplicate_of_not_kept": dangling,
            "kept_without_group": missing_group,
        },
    )


def g_language(build: Path, min_agreement: float, lid: LanguageIdentifier, low_conf: float) -> Gate:
    can = _canonical(build)
    if not can.height:
        return Gate(name="language", status="fail", details={"error": "no canonical rows"})
    texts = can["text"].to_list()
    res = lid.detect_many(texts)
    langs = can["lang"].to_list()
    agree = sum(1 for r, lang in zip(res, langs, strict=True) if r.lang == lang)
    per_source: dict[str, str] = {}
    for sid in sorted(set(can["source_id"].to_list())):
        idx = [i for i, s in enumerate(can["source_id"].to_list()) if s == sid]
        a = sum(1 for i in idx if res[i].lang == langs[i])
        per_source[sid] = f"{a}/{len(idx)}"
    rate = agree / len(texts)
    low = sum(1 for r in res if r.conf < low_conf)
    by_lang = Counter(langs)
    return Gate(
        name="language",
        status="pass" if rate >= min_agreement else "fail",
        metric=round(rate, 6),
        threshold=min_agreement,
        details={
            "agreement_by_source": per_source,
            "canonical_langs": dict(by_lang),
            "low_confidence_rows": low,
            "code_switch_rows": int(can["code_switch"].sum()),
            "note": "LID re-run on canonical text; see ADR-020 for the decision rule",
        },
    )


def g_duplicates(build: Path, threshold: float) -> Gate:
    groups = _read(build, "groups.parquet")
    pairs = _read(build, "pairs.parquet")
    can_groups = groups.filter(pl.col("destination") == "canonical")
    exact_in_canonical = can_groups.height - can_groups["exact_group"].n_unique()
    can = _canonical(build)
    text_dupes = can.height - can["text"].n_unique() if can.height else 0
    fam = dict(zip(groups["example_id"].to_list(), groups["family_id"].to_list(), strict=True))
    strong = pairs.filter(pl.col("compatible") & (pl.col("jaccard") >= threshold))
    split_pairs = sum(
        1
        for a, b in zip(strong["example_a"].to_list(), strong["example_b"].to_list(), strict=True)
        if fam.get(a) != fam.get(b)
    )
    blocked = pairs.filter(~pl.col("compatible") & (pl.col("jaccard") >= threshold)).height
    problems = exact_in_canonical + text_dupes + split_pairs
    return Gate(
        name="duplicates",
        status="pass" if problems == 0 else "fail",
        metric=problems,
        threshold=0,
        details={
            "exact_groups_with_2plus_canonical_rows": exact_in_canonical,
            "identical_canonical_texts": text_dupes,
            "compatible_near_dup_pairs_across_families": split_pairs,
            "near_dup_pairs_blocked_by_label_guard": blocked,
            "family_threshold": threshold,
        },
    )


def _slot_ok(name: str, value: str, catalog: PlaceholderCatalog) -> bool:
    for spec in catalog.slots.values():
        if spec.slot != name:
            continue
        match spec.kind:
            case "choice":
                return value in spec.values
            case "fixed":
                return value == spec.value
            case "digits":
                return value.isdigit() and len(value) == (spec.length or 0)
            case "template":
                prefix = (spec.template or "").split("{digits6}")[0]
                return value.startswith(prefix) and value[len(prefix) :].isdigit()
            case "integer":
                return value.isdigit() and (spec.min or 0) <= int(value) <= (spec.max or 0)
    return False


def g_placeholders(build: Path, catalog: PlaceholderCatalog) -> Gate:
    can = _canonical(build)
    pen = _read(build, "pending.parquet")
    unresolved = sum(
        1 for p in (can, pen) if p.height for t in p["text"].to_list() if PLACEHOLDER_RE.search(t)
    )
    bad_slots = 0
    for p in (can, pen):
        if not p.height:
            continue
        for slots in p["slots"].to_list():
            for k, v in slots_dict(slots).items():
                if not _slot_ok(k, v, catalog):
                    bad_slots += 1
    return Gate(
        name="placeholders",
        status="pass" if unresolved == 0 and bad_slots == 0 else "fail",
        metric=unresolved + bad_slots,
        threshold=0,
        details={"unresolved_in_text": unresolved, "slot_values_not_from_catalog": bad_slots},
    )


def g_pii(build: Path) -> Gate:
    found: Counter[str] = Counter()
    for p in (_canonical(build), _read(build, "pending.parquet")):
        if not p.height:
            continue
        for t in p["text"].to_list():
            found["email"] += sum(
                1 for m in EMAIL_RE.finditer(t) if not _SYNTH_EMAIL.match(m.group(0))
            )
            for m in CARD_RE.finditer(t):
                digits = re.sub(r"\D", "", m.group(0))
                if m.group(0) != _SYNTH_CARD and 13 <= len(digits) <= 19 and luhn_ok(digits):
                    found["card"] += 1
            for m in PHONE_RE.finditer(t):
                digits = re.sub(r"\D", "", m.group(0))
                if 10 <= len(digits) <= 13 and not _SYNTH_PHONE.match(m.group(0)):
                    found["phone"] += 1
    total = sum(found.values())
    return Gate(
        name="pii",
        status="pass" if total == 0 else "fail",
        metric=total,
        threshold=0,
        details={"unmasked": dict(found), "scope": "customer text of canonical + pending"},
    )


def g_licence(build: Path, entries: Sequence[SourceEntry]) -> Gate:
    can = _canonical(build)
    by_id = {e.source_id: e for e in entries}
    not_trainable: list[str] = []
    for sid in sorted(set(can["source_id"].to_list())) if can.height else []:
        try:
            check_use(by_id[sid], Purpose.TRAINING)
        except (PolicyError, KeyError):
            not_trainable.append(sid)
    pen = _read(build, "pending.parquet")
    mismatch = 0
    for p in (pen, _read(build, "rejected.parquet")):
        for sid, use in zip(p["source_id"].to_list(), p["allowed_use"].to_list(), strict=True):
            if sid not in by_id or by_id[sid].allowed_use.value != use:
                mismatch += 1
    return Gate(
        name="licence",
        status="pass" if not not_trainable and mismatch == 0 else "fail",
        metric=len(not_trainable) + mismatch,
        threshold=0,
        details={
            "canonical_sources": sorted(set(can["source_id"].to_list())) if can.height else [],
            "canonical_sources_not_trainable": not_trainable,
            "rows_with_wrong_allowed_use": mismatch,
        },
    )


def g_labels(build: Path) -> Gate:
    can = _canonical(build)
    unknown = can.filter(~pl.col("intent").is_in(sorted(INTENTS))).height if can.height else 0
    esc_bad = (
        can.filter(pl.col("escalate") & pl.col("escalation_reason").is_null()).height
        if can.height
        else 0
    )
    actions = dict(Counter(can["action"].to_list())) if can.height else {}
    return Gate(
        name="labels",
        status="pass" if unknown == 0 and esc_bad == 0 else "fail",
        metric=unknown + esc_bad,
        threshold=0,
        details={
            "unknown_intents": unknown,
            "escalate_without_reason": esc_bad,
            "actions": actions,
            "intents": can["intent"].n_unique() if can.height else 0,
            "note": "labels come only from configs/intent_map.yaml (draft v0, ADR-021)",
        },
    )


def run_gates(
    build: Path,
    entries: Sequence[SourceEntry],
    data_dir: Path,
    cfg: CleanSettings,
    catalog: PlaceholderCatalog,
) -> GateReport:
    lid = LanguageIdentifier(other_min_conf=cfg.lid_other_min_conf, min_letters=cfg.lid_min_letters)
    gates = [
        g_accounting(build, entries, data_dir),
        g_schema(build, cfg.min_validity),
        g_ids(build),
        g_language(build, cfg.min_lid_agreement, lid, cfg.lid_low_conf),
        g_duplicates(build, cfg.family_threshold),
        g_placeholders(build, catalog),
        g_pii(build),
        g_licence(build, entries),
        g_labels(build),
        Gate(
            name="splits_and_leakage",
            status="not_applicable",
            details={"reason": "train/validation/test/gold splits are created in Stage 4"},
        ),
    ]
    return GateReport(passed=all(g.status != "fail" for g in gates), gates=gates)
