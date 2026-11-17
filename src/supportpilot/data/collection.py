"""Commissioned native-speaker collection (ADR-049, DATA_CONTRACTS §10).

plan (`configs/collection.yaml`) -> scenario briefs -> writers deliver `rows.csv` ->
`check_delivery` (contract, provenance, rights, language, PII, duplicates, coverage) ->
`intake_delivery` copies the checked delivery into immutable raw storage ->
`read_commissioned` feeds it to the corpus build like any other source.

Briefs are structured facets, never example sentences, so writers phrase each request
themselves. Authorship cannot be proven by code: the contract attestation, contributor
qualification, timing/duplicate flags and the independent review (`review.py`) together are
the evidence, and each is reported separately.
"""

from __future__ import annotations

import csv
import io
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from supportpilot.common.hashing import sha256_text
from supportpilot.common.manifest import read_model_json, scan_files, write_model_json
from supportpilot.common.schemas import (
    INTENTS,
    LanguageCode,
    NonEmptyStr,
    RawManifest,
    Slug,
    SourceEntry,
)
from supportpilot.data.acquire import MANIFEST, raw_dir, verify_raw
from supportpilot.data.dedup import near_duplicate_pairs
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.readers import RawItem
from supportpilot.data.taxonomy import NEEDS_REFERENCE, Taxonomy
from supportpilot.data.text import dedup_key, detect_language, normalize_text, residual_pii

DELIVERY_MANIFEST = "delivery.json"
ROWS_FILE = "rows.csv"
BRIEFS_FILE = "briefs.csv"
ROW_COLUMNS = ("row_id", "writer_id", "brief_id", "slice", "lang", "text", "duration_sec")
BRIEF_COLUMNS = ("brief_id", "intent", "description", "facets")
LICENCE = "commissioned-assignment"


class CollectionError(Exception):
    """A delivery or plan is unusable; nothing was taken in."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- plan


class SliceSpec(_Model):
    lang: LanguageCode | None = None  # None: dominant language declared per row
    code_switch: bool
    native: list[LanguageCode] = Field(min_length=1)


class ReviewStratum(_Model):
    name: Slug
    size: int = Field(ge=10)
    lang: LanguageCode | None = None
    source: str | None = None
    exclude_source: str | None = None
    slice: str | None = None


class ReviewPlan(_Model):
    double_label_fraction: float = Field(ge=0.1, le=1)
    min_kappa: float = Field(gt=0, le=1)
    strata: list[ReviewStratum] = Field(min_length=1)


class CollectionPlan(_Model):
    schema_version: Literal["collection_plan/1"] = "collection_plan/1"
    plan_id: Slug
    seed: int
    source_id: Slug
    scenarios_per_intent: int = Field(ge=1)
    writers_per_scenario: int = Field(ge=1)
    min_writers_per_slice: int = Field(ge=1)
    min_duration_sec: int = Field(ge=0)
    max_flagged_fraction: float = Field(ge=0, le=1)
    near_duplicate_jaccard: float = Field(gt=0, le=1)
    slices: dict[str, SliceSpec] = Field(min_length=1)
    facets: dict[str, list[NonEmptyStr]]
    review: ReviewPlan

    @model_validator(mode="after")
    def _check(self) -> Self:
        needed = {"product", "customer", "channel", "city", "mood", "length", "reference", "topic"}
        missing = needed - set(self.facets)
        if missing:
            raise ValueError(f"facets missing: {sorted(missing)}")
        if self.min_writers_per_slice < self.writers_per_scenario:
            raise ValueError("min_writers_per_slice must be >= writers_per_scenario")
        for s in self.review.strata:
            if s.slice is not None and s.slice not in self.slices:
                raise ValueError(f"review stratum {s.name}: unknown slice {s.slice!r}")
        return self


def load_plan(path: Path) -> CollectionPlan:
    return CollectionPlan.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- briefs


class ScenarioBrief(_Model):
    brief_id: str = Field(pattern=r"^sc-[a-z_]+-\d{2}$")
    intent: str
    description: str
    facets: dict[str, str]


_PRODUCT_GROUPS = frozenset({"order", "delivery", "returns", "payment", "product"})
_CITY_GROUPS = frozenset({"order", "delivery", "returns", "store", "contact"})


def _pick(plan: CollectionPlan, intent: str, facet: str, index: int) -> str:
    """Seeded rotation: consecutive scenarios of one intent get distinct values."""
    values = plan.facets[facet]
    offset = int(sha256_text(f"{plan.seed}:{intent}:{facet}")[:8], 16)
    return values[(offset + index) % len(values)]


def make_briefs(plan: CollectionPlan, taxonomy: Taxonomy) -> list[ScenarioBrief]:
    """`scenarios_per_intent` independent scenarios for every v1 intent (deterministic)."""
    briefs: list[ScenarioBrief] = []
    for intent in sorted(taxonomy.intents):
        spec = taxonomy.intents[intent]
        for i in range(plan.scenarios_per_intent):
            names = ["customer", "channel", "mood", "length"]
            if intent == "out_of_scope":
                names.append("topic")
            if spec.group in _PRODUCT_GROUPS:
                names.append("product")
            if spec.group in _CITY_GROUPS:
                names.append("city")
            if spec.action == NEEDS_REFERENCE:
                names.append("reference")
            facets = {n: _pick(plan, intent, n, i) for n in sorted(names)}
            briefs.append(
                ScenarioBrief(
                    brief_id=f"sc-{intent}-{i + 1:02d}",
                    intent=intent,
                    description=spec.description,
                    facets=facets,
                )
            )
    return briefs


def briefs_csv(briefs: list[ScenarioBrief]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(BRIEF_COLUMNS)
    for b in briefs:
        facets = "; ".join(f"{k}: {v}" for k, v in b.facets.items())
        w.writerow([b.brief_id, b.intent, b.description, facets])
    return buf.getvalue()


# ---------------------------------------------------------------- delivery contract


class Qualification(_Model):
    test_id: NonEmptyStr  # language/task screening test the contributor passed
    score: float = Field(ge=0, le=1)
    passed: bool


class Attestation(_Model):
    """Signed by each contributor in the contract; all must be true."""

    original_work: bool  # written personally, not copied
    no_llm: bool  # no LLM or text generator used
    no_machine_translation: bool
    no_real_personal_data: bool
    rights_assigned: bool  # all rights incl. ML training/evaluation and publication


class Contributor(_Model):
    contributor_id: str = Field(pattern=r"^[a-z]{1,4}\d{2,4}$")  # pseudonymous, e.g. w01, r02
    role: Literal["writer", "reviewer"]
    native: list[LanguageCode] = Field(min_length=1)
    fluent: list[LanguageCode] = Field(default_factory=list)
    country: str = Field(pattern=r"^[A-Z]{2}$")
    contract_ref: NonEmptyStr  # owner's contract record id; real identities stay off-repo
    qualification: Qualification
    attestation: Attestation


class DeliveryManifest(_Model):
    schema_version: Literal["commissioned_delivery/1"] = "commissioned_delivery/1"
    plan_id: Slug
    batch_id: Slug
    vendor: NonEmptyStr
    licence: Literal["commissioned-assignment"]
    licence_evidence: NonEmptyStr  # clause / contract section granting the rights
    delivered_at: date
    contributors: list[Contributor] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique(self) -> Self:
        ids = [c.contributor_id for c in self.contributors]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate contributor_id")
        return self


class DeliveredRow(_Model):
    row_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    writer_id: str
    brief_id: str
    slice: str
    lang: LanguageCode
    text: str = Field(min_length=1, max_length=1000)
    duration_sec: NonNegativeInt


class SliceCounts(_Model):
    rows: int
    scenarios: int  # distinct briefs with at least one row
    writers: int
    intents: int
    scenarios_full: int  # briefs with >= writers_per_scenario distinct writers
    flagged: dict[str, int]
    flagged_fraction: float


class DeliveryReport(_Model):
    plan_id: str
    batch_id: str
    content_sha256: str
    rows: int
    slices: dict[str, SliceCounts]
    problems: list[str]  # blocking
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems


def delivery_content_sha256(path: Path) -> str:
    """sha256 over the sorted `path\\tsha256` lines of the delivery files (= registry revision)."""
    lines = sorted(f"{e.path}\t{e.sha256}\n" for e in scan_files(path))
    return sha256_text("".join(lines))


def _read_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != columns:
            raise CollectionError(f"{path.name}: columns must be {list(columns)}")
        return list(reader)


def load_rows(path: Path) -> list[DeliveredRow]:
    rows = []
    for i, raw in enumerate(_read_csv(path / ROWS_FILE, ROW_COLUMNS), start=2):
        try:
            rows.append(DeliveredRow.model_validate(raw))
        except ValueError as exc:
            raise CollectionError(f"{ROWS_FILE} line {i}: {exc}") from exc
    return rows


def _contributor_problems(m: DeliveryManifest, plan: CollectionPlan) -> list[str]:
    out = []
    for c in m.contributors:
        if not all(c.attestation.model_dump().values()):
            out.append(f"{c.contributor_id}: attestation incomplete")
        if not c.qualification.passed:
            out.append(f"{c.contributor_id}: did not pass {c.qualification.test_id}")
    if m.plan_id != plan.plan_id:
        out.append(f"plan_id {m.plan_id!r} != {plan.plan_id!r}")
    return out


def check_delivery(path: Path, plan: CollectionPlan, taxonomy: Taxonomy) -> DeliveryReport:
    """All intake checks; `problems` block intake, `warnings` are reported."""
    for name in (DELIVERY_MANIFEST, ROWS_FILE, BRIEFS_FILE):
        if not (path / name).is_file():
            raise CollectionError(f"{path}: missing {name}")
    manifest = read_model_json(path / DELIVERY_MANIFEST, DeliveryManifest)
    rows = load_rows(path)
    problems = _contributor_problems(manifest, plan)
    warnings: list[str] = []

    expected = make_briefs(plan, taxonomy)
    if (path / BRIEFS_FILE).read_text(encoding="utf-8") != briefs_csv(expected):
        problems.append(f"{BRIEFS_FILE} differs from the plan's briefs")
    brief_intent = {b.brief_id: b.intent for b in expected}
    people = {c.contributor_id: c for c in manifest.contributors}

    seen_ids: set[str] = set()
    seen_text: dict[str, str] = {}
    by_slice: dict[str, list[DeliveredRow]] = defaultdict(list)
    flags: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        where = f"row {r.row_id}"
        if r.row_id in seen_ids:
            problems.append(f"{where}: duplicate row_id")
        seen_ids.add(r.row_id)
        spec = plan.slices.get(r.slice)
        if spec is None:
            problems.append(f"{where}: unknown slice {r.slice!r}")
            continue
        by_slice[r.slice].append(r)
        writer = people.get(r.writer_id)
        if writer is None or writer.role != "writer":
            problems.append(f"{where}: {r.writer_id!r} is not a writer in {DELIVERY_MANIFEST}")
        elif not set(spec.native) <= set(writer.native) | set(writer.fluent):
            problems.append(f"{where}: writer {r.writer_id} lacks {spec.native} for {r.slice}")
        elif not set(spec.native) & set(writer.native):
            problems.append(f"{where}: writer {r.writer_id} is not native in {spec.native}")
        if r.brief_id not in brief_intent:
            problems.append(f"{where}: unknown brief {r.brief_id!r}")
        if spec.lang is not None and r.lang != spec.lang:
            problems.append(f"{where}: lang {r.lang} != slice {r.slice}")
        text = normalize_text(r.text)
        if residual_pii(text):
            problems.append(f"{where}: personal data pattern; use placeholders like <order_id>")
        key = f"{r.slice}|{dedup_key(text)}"
        if key in seen_text:
            problems.append(f"{where}: exact duplicate of row {seen_text[key]}")
        seen_text.setdefault(key, r.row_id)
        sig = detect_language(text)
        if sig.guess not in (r.lang, "unknown"):
            flags[r.slice][f"lang_guess_{sig.guess}"] += 1
        if sig.code_switch != spec.code_switch:
            flags[r.slice]["code_switch_mismatch"] += 1
        if r.duration_sec < plan.min_duration_sec:
            flags[r.slice]["too_fast"] += 1

    per_brief_writer = Counter((r.slice, r.brief_id, r.writer_id) for r in rows)
    for (sl, brief, writer_id), n in sorted(per_brief_writer.items()):
        if n > 1:
            problems.append(f"{sl}/{brief}: writer {writer_id} wrote {n} rows (one per scenario)")

    slices: dict[str, SliceCounts] = {}
    for name in sorted(plan.slices):
        part = by_slice.get(name, [])
        for i, j, _ in near_duplicate_pairs(
            [dedup_key(normalize_text(r.text)) for r in part],
            shingle=4,
            threshold=plan.near_duplicate_jaccard,
            num_perm=64,
            bands=16,
            seed=plan.seed,
        ):
            if part[i].writer_id != part[j].writer_id:
                flags[name]["near_duplicate_across_writers"] += 1
        writers_by_brief: dict[str, set[str]] = defaultdict(set)
        for r in part:
            writers_by_brief[r.brief_id].add(r.writer_id)
        flagged_rows = sum(flags[name].values())
        counts = SliceCounts(
            rows=len(part),
            scenarios=len(writers_by_brief),
            writers=len({r.writer_id for r in part}),
            intents=len({brief_intent[b] for b in writers_by_brief if b in brief_intent}),
            scenarios_full=sum(
                1 for w in writers_by_brief.values() if len(w) >= plan.writers_per_scenario
            ),
            flagged=dict(sorted(flags[name].items())),
            flagged_fraction=round(flagged_rows / len(part), 4) if part else 0.0,
        )
        slices[name] = counts
        if part and counts.flagged_fraction > plan.max_flagged_fraction:
            problems.append(
                f"{name}: {counts.flagged_fraction:.1%} rows flagged "
                f"(max {plan.max_flagged_fraction:.0%}): {counts.flagged}"
            )
        target = len(INTENTS) * plan.scenarios_per_intent
        if counts.scenarios_full < target or counts.writers < plan.min_writers_per_slice:
            warnings.append(
                f"{name}: {counts.scenarios_full}/{target} scenarios with "
                f">= {plan.writers_per_scenario} writers, {counts.writers} writers "
                f"(target >= {plan.min_writers_per_slice}); batch incomplete for acceptance"
            )
    return DeliveryReport(
        plan_id=manifest.plan_id,
        batch_id=manifest.batch_id,
        content_sha256=delivery_content_sha256(path),
        rows=len(rows),
        slices=slices,
        problems=problems,
        warnings=warnings,
    )


def intake_delivery(
    path: Path,
    data_dir: Path,
    entry: SourceEntry,
    plan: CollectionPlan,
    taxonomy: Taxonomy,
) -> DeliveryReport:
    """Check the delivery and copy it, unchanged, into `data/raw/{source_id}/{sha256}/`."""
    check_use(entry, Purpose.ACQUIRE)
    if entry.fetch is None or entry.fetch.type != "delivery" or entry.licence != LICENCE:
        raise CollectionError(f"{entry.source_id}: registry entry is not a {LICENCE} delivery")
    report = check_delivery(path, plan, taxonomy)
    if not report.ok:
        raise CollectionError(
            f"delivery failed {len(report.problems)} checks: {report.problems[:5]}"
        )
    if report.content_sha256 != entry.revision:
        raise CollectionError(
            f"delivery sha256 {report.content_sha256} != registry revision {entry.revision}"
        )
    final = raw_dir(data_dir, entry)
    if final.exists():
        if verify_raw(final):
            raise CollectionError(f"{final} exists but fails verification; restore from backup")
        return report
    staging = final.parent / f".partial-{entry.revision}"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(path, staging)
    write_model_json(
        staging / MANIFEST,
        RawManifest(
            source_id=entry.source_id,
            revision=entry.revision,
            url=entry.url,
            retrieved_at=datetime.now(UTC),
            licence=entry.licence,
            files=scan_files(staging, exclude=frozenset({MANIFEST})),
        ),
    )
    staging.rename(final)
    return report


def read_commissioned(raw: Path) -> Iterator[RawItem]:
    """Corpus reader: one item per delivered row. The scenario (brief) is the split unit for
    every slice and writer, so paraphrases of one scenario never cross splits; the slice is
    kept as `source_split`."""
    intents = {r["brief_id"]: r["intent"] for r in _read_csv(raw / BRIEFS_FILE, BRIEF_COLUMNS)}
    for r in sorted(load_rows(raw), key=lambda r: r.row_id):
        yield RawItem(
            source_item_id=f"{r.slice}-{r.row_id}",
            lang=r.lang,
            text=r.text,
            source_label=intents[r.brief_id],
            source_split=r.slice,
            group_key=r.brief_id,
        )


def writers_by_example(raw: Path, source_id: str) -> dict[str, str]:
    """example_id -> writer id (the corpus derives example ids from source_item_id)."""
    return {
        "ex-" + sha256_text(f"{source_id}|{r.slice}-{r.row_id}")[:16]: r.writer_id
        for r in load_rows(raw)
    }
