"""Independent human review of corpus labels (ADR-049, EVALUATION §1).

`make_package`: blind sheets (no label shown) per review stratum; a seeded
`double_label_fraction` of each stratum also goes to a second reviewer.
`score_review`: label agreement with the corpus (Wilson 95% CI), Cohen's kappa between the
two reviewers, language/naturalness rates, every disagreement, and an independence check
(a reviewer never reviews rows they wrote). Missing or too-small evidence stays NOT_EVALUATED.
"""

from __future__ import annotations

import csv
import io
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from supportpilot.common.hashing import sha256_text
from supportpilot.data.collection import CollectionPlan, ReviewStratum
from supportpilot.data.records import NOT_EVALUATED, IntentExample, Pool

SHEET_COLUMNS = (
    "example_id",
    "lang",
    "text",
    "intent",
    "language_ok(y/n)",
    "natural_request(y/n)",
    "notes",
)
KEY_COLUMNS = ("example_id", "stratum", "intent", "source_id", "source_item_id", "double")


class ReviewError(Exception):
    """Review inputs are inconsistent; nothing was scored."""


def _unit(seed: int, key: str) -> float:
    return int(sha256_text(f"{seed}:{key}")[:15], 16) / 16**15


def _in_stratum(ex: IntentExample, s: ReviewStratum) -> bool:
    return (
        ex.pool is Pool.CORE
        and (s.lang is None or ex.lang == s.lang)
        and (s.source is None or ex.source_id == s.source)
        and (s.exclude_source is None or ex.source_id != s.exclude_source)
        and (s.slice is None or ex.source_split == s.slice)
    )


def select_stratum(
    examples: list[IntentExample], s: ReviewStratum, seed: int
) -> list[IntentExample]:
    """Round-robin over intents in seeded order, at most one row per family per pass."""
    by_intent: dict[str, list[IntentExample]] = defaultdict(list)
    for ex in sorted(examples, key=lambda e: (_unit(seed, e.example_id), e.example_id)):
        if _in_stratum(ex, s):
            by_intent[ex.intent].append(ex)
    if not by_intent:
        raise ReviewError(f"stratum {s.name}: no core rows")
    picked: list[IntentExample] = []
    families: set[str] = set()
    while len(picked) < s.size and any(by_intent.values()):
        for intent in sorted(by_intent):
            queue = by_intent[intent]
            while queue and queue[0].family_id in families:
                queue.pop(0)
            if queue and len(picked) < s.size:
                ex = queue.pop(0)
                picked.append(ex)
                families.add(ex.family_id)
    return picked


def _csv(header: tuple[str, ...], rows: list[list[str]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def make_package(
    examples: list[IntentExample], plan: CollectionPlan, only: frozenset[str] | None = None
) -> dict[str, str]:
    """{file name: CSV}: `sheet_a.csv` (all), `sheet_b.csv` (overlap), `key.csv` (owner only).
    `only` limits the package to the named strata (e.g. before commissioned data exists)."""
    if only is not None and only - {s.name for s in plan.review.strata}:
        raise ReviewError(f"unknown strata: {sorted(only - {s.name for s in plan.review.strata})}")
    a: list[list[str]] = []
    b: list[list[str]] = []
    key: list[list[str]] = []
    for s in plan.review.strata:
        if only is not None and s.name not in only:
            continue
        rows = select_stratum(examples, s, plan.seed)
        n_double = math.ceil(len(rows) * plan.review.double_label_fraction)
        double = {
            e.example_id
            for e in sorted(rows, key=lambda e: _unit(plan.seed + 1, e.example_id))[:n_double]
        }
        for ex in rows:
            if any(k[0] == ex.example_id for k in key):
                raise ReviewError(f"{ex.example_id} falls in two strata; make them disjoint")
            line = [ex.example_id, ex.lang, ex.text, "", "", "", ""]
            a.append(line)
            if ex.example_id in double:
                b.append(line)
            key.append(
                [
                    ex.example_id,
                    s.name,
                    ex.intent,
                    ex.source_id,
                    ex.source_item_id,
                    "y" if ex.example_id in double else "n",
                ]
            )
    order = {r[0]: _unit(plan.seed + 2, r[0]) for r in a}  # strata interleaved, not grouped
    a.sort(key=lambda r: order[r[0]])
    b.sort(key=lambda r: order[r[0]])
    return {
        "sheet_a.csv": _csv(SHEET_COLUMNS, a),
        "sheet_b.csv": _csv(SHEET_COLUMNS, b),
        "key.csv": _csv(KEY_COLUMNS, key),
    }


# ---------------------------------------------------------------- scoring


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StratumScore(_Model):
    stratum: str
    reviewed: int
    label_agreement: float | str  # reviewer A vs corpus label
    label_agreement_ci95: tuple[float, float] | str
    double_labelled: int
    double_fraction: float
    kappa: float | str  # reviewer A vs reviewer B, blind
    kappa_ok: bool | str
    language_ok_rate: float | str
    natural_rate: float | str


class ReviewReport(_Model):
    reviewers: dict[str, str]  # sheet -> reviewer id
    strata: list[StratumScore]
    disagreements: list[dict[str, str]]
    problems: list[str]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        raise ValueError("n must be > 0")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("kappa needs two equal, non-empty label lists")
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca) / (n * n)
    return 1.0 if pe == 1 else round((po - pe) / (1 - pe), 4)


def _read(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = set(columns) - set(reader.fieldnames or ())
        if missing:
            raise ReviewError(f"{path.name}: missing columns {sorted(missing)}")
        return list(reader)


def _yes_rate(rows: list[dict[str, str]], col: str) -> float | str:
    vals = [r[col].strip().lower() for r in rows if r[col].strip()]
    return round(sum(v == "y" for v in vals) / len(vals), 4) if vals else NOT_EVALUATED


def score_review(
    package: Path,
    reviewer_a: str,
    reviewer_b: str,
    plan: CollectionPlan,
    intents: frozenset[str],
    writers: dict[str, str] | None = None,
) -> ReviewReport:
    """Score filled `sheet_a.csv` / `sheet_b.csv` against `key.csv`.

    `writers` maps example_id -> writer id for commissioned rows (independence check)."""
    if reviewer_a == reviewer_b:
        raise ReviewError("the two reviewers must be different people")
    key = {r["example_id"]: r for r in _read(package / "key.csv", KEY_COLUMNS)}
    sheet_a = {r["example_id"]: r for r in _read(package / "sheet_a.csv", SHEET_COLUMNS)}
    sheet_b = {r["example_id"]: r for r in _read(package / "sheet_b.csv", SHEET_COLUMNS)}
    problems: list[str] = []
    if set(sheet_a) != set(key):
        problems.append("sheet_a does not cover exactly the key's rows")
    expected_b = {k for k, r in key.items() if r["double"] == "y"}
    if set(sheet_b) != expected_b:
        problems.append("sheet_b does not cover exactly the double-labelled rows")
    for name, sheet in (("sheet_a", sheet_a), ("sheet_b", sheet_b)):
        for ex_id, row in sheet.items():
            label = row["intent"].strip()
            if label and label not in intents:
                problems.append(f"{name} {ex_id}: unknown intent {label!r}")
    for ex_id, writer in (writers or {}).items():
        if writer in (reviewer_a, reviewer_b) and ex_id in key:
            problems.append(f"{ex_id}: reviewed by its own writer {writer}")

    disagreements: list[dict[str, str]] = []
    strata: list[StratumScore] = []
    for s in plan.review.strata:
        ids = sorted(k for k, r in key.items() if r["stratum"] == s.name)
        if not ids:
            continue  # stratum not in this package
        done = [i for i in ids if sheet_a.get(i, {}).get("intent", "").strip()]
        agree = [i for i in done if sheet_a[i]["intent"].strip() == key[i]["intent"]]
        for i in done:
            if i not in agree:
                disagreements.append(
                    {
                        "example_id": i,
                        "stratum": s.name,
                        "corpus": key[i]["intent"],
                        "reviewer_a": sheet_a[i]["intent"].strip(),
                        "reviewer_b": sheet_b.get(i, {}).get("intent", "").strip(),
                        "notes": sheet_a[i]["notes"],
                    }
                )
        both = [i for i in ids if i in done and sheet_b.get(i, {}).get("intent", "").strip()]
        frac = round(len(both) / len(done), 4) if done else 0.0
        kappa: float | str = NOT_EVALUATED
        kappa_ok: bool | str = NOT_EVALUATED
        if both and frac >= 0.1:
            kappa = cohen_kappa(
                [sheet_a[i]["intent"].strip() for i in both],
                [sheet_b[i]["intent"].strip() for i in both],
            )
            kappa_ok = kappa >= plan.review.min_kappa
        rows_a = [sheet_a[i] for i in done]
        strata.append(
            StratumScore(
                stratum=s.name,
                reviewed=len(done),
                label_agreement=round(len(agree) / len(done), 4) if done else NOT_EVALUATED,
                label_agreement_ci95=wilson(len(agree), len(done)) if done else NOT_EVALUATED,
                double_labelled=len(both),
                double_fraction=frac,
                kappa=kappa,
                kappa_ok=kappa_ok,
                language_ok_rate=_yes_rate(rows_a, "language_ok(y/n)"),
                natural_rate=_yes_rate(rows_a, "natural_request(y/n)"),
            )
        )
    return ReviewReport(
        reviewers={"sheet_a": reviewer_a, "sheet_b": reviewer_b},
        strata=strata,
        disagreements=disagreements,
        problems=problems,
    )


def report_markdown(report: ReviewReport) -> str:
    lines = ["# Independent label review", ""]
    lines += [
        f"Reviewers: A = `{report.reviewers['sheet_a']}`, B = `{report.reviewers['sheet_b']}`."
    ]
    lines += [
        "",
        "| Stratum | reviewed | agreement (95% CI) | double | kappa | lang ok | natural |",
    ]
    lines += ["|---|---|---|---|---|---|---|"]
    for s in report.strata:
        ci: Any = s.label_agreement_ci95
        ci_txt = f"{ci[0]:.3f}-{ci[1]:.3f}" if isinstance(ci, tuple) else ci
        lines.append(
            f"| {s.stratum} | {s.reviewed} | {s.label_agreement} ({ci_txt}) | "
            f"{s.double_labelled} ({s.double_fraction:.0%}) | {s.kappa} | "
            f"{s.language_ok_rate} | {s.natural_rate} |"
        )
    lines += ["", f"Disagreements: {len(report.disagreements)}. Problems: {len(report.problems)}."]
    lines += [f"- {p}" for p in report.problems]
    return "\n".join(lines) + "\n"
