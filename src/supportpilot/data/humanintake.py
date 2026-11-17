"""Intake of commissioned human data: requests, label reviews and store-grounded answers.

DATA_CONTRACTS §20.2 (`human_intake/1`). A delivery from an approved supplier, volunteer
group or owner-run collection is placed under `data/raw/<source_id>/<revision>/` and checked
here before any row can reach the corpus. Software cannot certify who a person is or which
languages they speak: identities and competence come from the delivery's contributor records
and the supplier's evidence, which the owner verifies before registering the source. What is
enforced mechanically:

- every contributor attests independence and no LLM/MT substitution, and the delivery states
  rights, consent and a collection method; unattested people's work is not admitted;
- a request is admitted only with >= 2 valid label reviews by people other than its author who
  are native/fluent in its language variant (both RU and KK for `mixed`); unanimous labels are
  `human_agreed`, otherwise a separate adjudicator decides (`human_adjudicated`);
- reviewers also judge whether the text is a natural, plausible customer message and confirm
  its language variant; either failing excludes the row (never relabelled by software);
- a store-grounded answer must cite only chunk ids of the pinned KB version, call only taxonomy
  tools, match the adjudicated action and the preferred reply language, and pass every rubric
  criterion in >= 2 independent answer reviews.

Every delivered item ends either admitted or excluded with a reason; the report gives the
counts, coverage of the 35 intents, independent scenarios and human-human agreement.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.manifest import scan_files, write_model_json
from supportpilot.common.schemas import RawManifest, SourceEntry
from supportpilot.data.taxonomy import Taxonomy

SCHEMA = "human_intake/1"
Variant = Literal["en", "ru", "kk", "mixed"]
ReplyLang = Literal["en", "ru", "kk"]
Role = Literal["author", "reviewer", "adjudicator", "answer_author", "answer_reviewer"]
Competence = Literal["native", "fluent"]
FILES = (
    "batch.json",
    "contributors.jsonl",
    "requests.jsonl",
    "label_reviews.jsonl",
    "adjudications.jsonl",
    "answers.jsonl",
    "answer_reviews.jsonl",
)
OPTIONAL = frozenset({"adjudications.jsonl", "answers.jsonl", "answer_reviews.jsonl"})
# `spoken_asr` keeps the builder's `human_spoken_asr` meaning (human speech, machine text).
ORIGIN = {
    "written": "human_written",
    "spoken_human_transcript": "human_written",
    "spoken_asr": "human_spoken_asr",
}
RUBRIC = (
    "responsive",
    "complete",
    "claims_supported",
    "citations_correct",
    "abstains_when_unsupported",
    "language_ok",
)
MIN_REVIEWS = 2


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Batch(_Model):
    schema_version: Literal["human_intake/1"]
    batch_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    supplier: str = Field(min_length=2)
    collection_method: str = Field(min_length=20)
    rights_evidence: str = Field(min_length=20)
    consent_evidence: str = Field(min_length=20)
    kb_version: str = Field(pattern=r"^kb_[a-z0-9_]+-\d{8}-[0-9a-f]{8}$")
    delivered_at: date


class Contributor(_Model):
    contributor_id: str = Field(min_length=3)  # opaque, stable, no names or e-mails
    roles: list[Role] = Field(min_length=1)
    languages: dict[ReplyLang, Competence] = Field(min_length=1)
    competence_evidence: str = Field(min_length=10)
    identity_verified: bool
    independence_attested: bool
    no_llm_attested: bool

    @property
    def qualified(self) -> bool:
        return self.identity_verified and self.independence_attested and self.no_llm_attested

    def speaks(self, variant: str) -> bool:
        needed = {"ru", "kk"} if variant == "mixed" else {variant}
        return needed <= set(self.languages)


class Turn(_Model):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class Request(_Model):
    item_id: str = Field(min_length=1)
    author_id: str
    scenario_id: str = Field(min_length=3)  # author-side need: variants/paraphrases share it
    variant: Variant
    preferred_reply_lang: ReplyLang
    method: Literal["written", "spoken_human_transcript", "spoken_asr"]
    text: str = Field(min_length=1)
    context: list[Turn] = Field(default_factory=list)


class LabelReview(_Model):
    item_id: str
    reviewer_id: str
    natural: bool  # a person could plausibly have written this to a store's support
    variant: Variant
    intent: str
    action: str
    escalation_reason: str | None = None


class Adjudication(_Model):
    item_id: str
    adjudicator_id: str
    accept: bool
    variant: Variant
    intent: str
    action: str
    escalation_reason: str | None = None


class ToolCall(_Model):
    name: str
    arguments: dict[str, str] = Field(default_factory=dict)


class Answer(_Model):
    item_id: str
    author_id: str
    lang: ReplyLang
    behaviour: str  # an action of the taxonomy
    text: str = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)


class AnswerReview(_Model):
    item_id: str
    reviewer_id: str
    responsive: bool
    complete: bool
    claims_supported: bool
    citations_correct: bool
    abstains_when_unsupported: bool
    language_ok: bool

    @property
    def passed(self) -> bool:
        return all(getattr(self, c) for c in RUBRIC)


@dataclass
class Delivery:
    batch: Batch
    contributors: list[Contributor]
    requests: list[Request]
    label_reviews: list[LabelReview]
    adjudications: list[Adjudication] = field(default_factory=list)
    answers: list[Answer] = field(default_factory=list)
    answer_reviews: list[AnswerReview] = field(default_factory=list)


@dataclass(frozen=True)
class Admitted:
    request: Request
    intent: str
    action: str
    escalation_reason: str | None
    label_status: Literal["human_agreed", "human_adjudicated"]
    reviewers: tuple[str, ...]
    answer: Answer | None


@dataclass
class IntakeResult:
    admitted: list[Admitted]
    excluded: list[dict[str, str]]
    report: dict[str, Any]


# ----------------------------------------------------------------------------- loading


def _jsonl[M: BaseModel](path: Path, model: type[M]) -> list[M]:
    if not path.is_file():
        return []
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(model.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path.name}:{n}: {exc}") from exc
    return out


def load_delivery(root: Path) -> Delivery:
    missing = [f for f in FILES if f not in OPTIONAL and not (root / f).is_file()]
    if missing:
        raise ValueError(f"{root}: delivery lacks {missing}")
    unexpected = sorted(p.name for p in root.iterdir() if p.name not in {*FILES, "manifest.json"})
    if unexpected:
        raise ValueError(f"{root}: unexpected files {unexpected}")
    batch = Batch.model_validate_json((root / "batch.json").read_text(encoding="utf-8"))
    d = Delivery(
        batch,
        _jsonl(root / "contributors.jsonl", Contributor),
        _jsonl(root / "requests.jsonl", Request),
        _jsonl(root / "label_reviews.jsonl", LabelReview),
        _jsonl(root / "adjudications.jsonl", Adjudication),
        _jsonl(root / "answers.jsonl", Answer),
        _jsonl(root / "answer_reviews.jsonl", AnswerReview),
    )
    for name, ids in (
        ("contributor", [c.contributor_id for c in d.contributors]),
        ("request", [r.item_id for r in d.requests]),
        ("answer", [a.item_id for a in d.answers]),
        ("adjudication", [a.item_id for a in d.adjudications]),
        ("label review", [(r.item_id, r.reviewer_id) for r in d.label_reviews]),
        ("answer review", [(r.item_id, r.reviewer_id) for r in d.answer_reviews]),
    ):
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate {name} ids in {root}")
    return d


def kb_chunk_ids(data_dir: Path, kb_version: str) -> frozenset[str]:
    """Chunk ids of a published KB version (`kb_<name>-<date>-<hash>`)."""
    name = kb_version[3:].rsplit("-", 2)[0]
    path = data_dir / "kb" / name / "versions" / kb_version / "chunks.parquet"
    if not path.is_file():
        raise ValueError(f"KB version {kb_version} not found at {path}")
    return frozenset(pl.read_parquet(path, columns=["chunk_id"])["chunk_id"].to_list())


def seal_delivery(root: Path, entry: SourceEntry) -> RawManifest:
    """Write the raw manifest of a delivery so `verify_raw` detects any later change."""
    if (root / "manifest.json").exists():
        raise ValueError(f"{root} is already sealed")
    load_delivery(root)  # schema first; never seal an unreadable delivery
    m = RawManifest(
        source_id=entry.source_id,
        revision=entry.revision,
        url=entry.url,
        retrieved_at=datetime.now(UTC),
        licence=entry.licence,
        files=scan_files(root),
    )
    write_model_json(root / "manifest.json", m)
    return m


# ----------------------------------------------------------------------------- checks


def _label_problem(tax: Taxonomy, intent: str, action: str, reason: str | None) -> str | None:
    if intent not in tax.intents:
        return "unknown_intent"
    if action not in tax.actions:
        return "unknown_action"
    if (action == "escalate") != (reason is not None):
        return "escalation_reason_mismatch"
    if reason is not None and reason not in tax.escalation_reasons:
        return "unknown_escalation_reason"
    return None


def cohen_kappa(pairs: Sequence[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / len(pairs)
    chance = sum(left[c] * right[c] for c in left.keys() | right.keys()) / len(pairs) ** 2
    return round((observed - chance) / (1 - chance), 4) if chance < 1 else None


def _answer_problem(
    ans: Answer, req: Request, action: str, tax: Taxonomy, chunks: frozenset[str]
) -> str | None:
    if ans.lang != req.preferred_reply_lang:
        return "answer_language_mismatch"
    if ans.behaviour != action:
        return "answer_behaviour_mismatch"
    if set(ans.citations) - chunks:
        return "citation_not_in_kb"
    if {c.name for c in ans.tool_calls} - set(tax.tools):
        return "unknown_tool"
    if action == "answer_kb" and not ans.citations:
        return "answer_without_citation"
    if action == "call_tool" and not ans.tool_calls:
        return "tool_answer_without_call"
    return None


def assess(d: Delivery, tax: Taxonomy, chunks: frozenset[str]) -> IntakeResult:
    people = {c.contributor_id: c for c in d.contributors}
    reviews: dict[str, list[LabelReview]] = defaultdict(list)
    for lr in d.label_reviews:
        reviews[lr.item_id].append(lr)
    adjud = {a.item_id: a for a in d.adjudications}
    answers = {a.item_id: a for a in d.answers}
    a_reviews: dict[str, list[AnswerReview]] = defaultdict(list)
    for ar in d.answer_reviews:
        a_reviews[ar.item_id].append(ar)
    known = {r.item_id for r in d.requests}
    referenced = {
        *(x.item_id for x in d.label_reviews),
        *(x.item_id for x in d.adjudications),
        *(x.item_id for x in d.answers),
        *(x.item_id for x in d.answer_reviews),
    }
    orphans = sorted(referenced - known)
    if orphans:
        raise ValueError(f"records refer to unknown items: {orphans[:5]}")

    def able(pid: str, role: str, variant: str, *, not_in: Iterable[str]) -> str | None:
        c = people.get(pid)
        if c is None:
            return "unknown_contributor"
        if role not in c.roles:
            return "role_not_assigned"
        if not c.qualified:
            return "unattested_contributor"
        if not c.speaks(variant):
            return "language_not_competent"
        if pid in set(not_in):
            return "not_independent"
        return None

    admitted: list[Admitted] = []
    excluded: list[dict[str, str]] = []
    rejected_reviews: Counter[str] = Counter()
    pairs: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    agreed = adjudicated = 0
    a_seen: Counter[str] = Counter()
    a_rubric: dict[str, Counter[str]] = defaultdict(Counter)
    a_reviews_n: Counter[str] = Counter()

    def out(req: Request, reason: str, detail: str = "") -> None:
        excluded.append({"item_id": req.item_id, "reason": reason, "detail": detail})

    for req in sorted(d.requests, key=lambda r: r.item_id):
        if why := able(req.author_id, "author", req.variant, not_in=()):
            out(req, "author_" + why)
            continue
        valid: list[LabelReview] = []
        for lr in sorted(reviews[req.item_id], key=lambda x: x.reviewer_id):
            why = able(lr.reviewer_id, "reviewer", req.variant, not_in=[req.author_id])
            why = why or _label_problem(tax, lr.intent, lr.action, lr.escalation_reason)
            if why:
                rejected_reviews[why] += 1
            else:
                valid.append(lr)
        if len(valid) < MIN_REVIEWS:
            out(req, "insufficient_independent_review", f"{len(valid)} valid reviews")
            continue
        first, second = valid[0], valid[1]
        p = pairs[req.variant]
        p["intent"].append((first.intent, second.intent))
        p["action"].append((first.action, second.action))
        p["escalate"].append((str(first.action == "escalate"), str(second.action == "escalate")))
        if sum(not lr.natural for lr in valid) * 2 >= len(valid):
            out(req, "not_natural", f"{sum(not lr.natural for lr in valid)}/{len(valid)}")
            continue
        labels = {(lr.variant, lr.intent, lr.action, lr.escalation_reason) for lr in valid}
        status: Literal["human_agreed", "human_adjudicated"]
        if len(labels) == 1:
            variant, intent, action, reason = labels.pop()
            status = "human_agreed"
        else:
            adj = adjud.get(req.item_id)
            if adj is None:
                out(req, "unresolved_disagreement", f"{len(labels)} distinct labels")
                continue
            barred = [req.author_id, *(lr.reviewer_id for lr in valid)]
            why = able(adj.adjudicator_id, "adjudicator", req.variant, not_in=barred)
            why = why or _label_problem(tax, adj.intent, adj.action, adj.escalation_reason)
            if why:
                out(req, "adjudication_" + why)
                continue
            if not adj.accept:
                out(req, "adjudicator_rejected")
                continue
            variant, intent, action, reason = (
                adj.variant,
                adj.intent,
                adj.action,
                adj.escalation_reason,
            )
            status = "human_adjudicated"
        if variant != req.variant:
            out(req, "variant_mismatch", f"declared {req.variant}, reviewed {variant}")
            continue
        agreed += status == "human_agreed"
        adjudicated += status == "human_adjudicated"
        kept_answer = None
        ans = answers.get(req.item_id)
        if ans is not None:
            a_seen["delivered"] += 1
            why = able(ans.author_id, "answer_author", ans.lang, not_in=())
            why = why or _answer_problem(ans, req, action, tax, chunks)
            ok_reviews = [
                ar
                for ar in a_reviews[req.item_id]
                if able(ar.reviewer_id, "answer_reviewer", ans.lang, not_in=[ans.author_id]) is None
            ]
            for ar in ok_reviews:
                a_reviews_n[ans.lang] += 1
                for c in RUBRIC:
                    a_rubric[ans.lang][c] += bool(getattr(ar, c))
            if why is None and len(ok_reviews) < MIN_REVIEWS:
                why = "answer_insufficient_review"
            if why is None and not all(ar.passed for ar in ok_reviews):
                why = "answer_failed_review"
            if why is None:
                kept_answer = ans
                a_seen["admitted"] += 1
            else:
                a_seen[why] += 1
        admitted.append(
            Admitted(
                req,
                intent,
                action,
                reason,
                status,
                tuple(lr.reviewer_id for lr in valid),
                kept_answer,
            )
        )

    report = intake_report(d, tax, admitted, excluded, rejected_reviews, pairs)
    report["labels"] = {"human_agreed": agreed, "human_adjudicated": adjudicated}
    report["answers"] = {
        "outcome": dict(sorted(a_seen.items())),
        "rubric_pass_rate": {
            lang: {c: round(n[c] / a_reviews_n[lang], 4) for c in RUBRIC}
            for lang, n in sorted(a_rubric.items())
        },
        "valid_reviews": dict(sorted(a_reviews_n.items())),
    }
    return IntakeResult(admitted, excluded, report)


def intake_report(
    d: Delivery,
    tax: Taxonomy,
    admitted: Sequence[Admitted],
    excluded: Sequence[Mapping[str, str]],
    rejected_reviews: Mapping[str, int],
    pairs: Mapping[str, Mapping[str, list[tuple[str, str]]]],
) -> dict[str, Any]:
    by_variant = Counter(a.request.variant for a in admitted)
    intents = {a.intent for a in admitted}
    lang_intent = Counter(f"{a.request.variant}|{a.intent}" for a in admitted)
    roles: Counter[str] = Counter()
    for c in d.contributors:
        roles.update(c.roles)
    return {
        "schema_version": SCHEMA,
        "batch_id": d.batch.batch_id,
        "kb_version": d.batch.kb_version,
        "delivered": len(d.requests),
        "admitted": len(admitted),
        "admitted_share": round(len(admitted) / len(d.requests), 4) if d.requests else None,
        "excluded_reasons": dict(sorted(Counter(x["reason"] for x in excluded).items())),
        "rejected_label_reviews": dict(sorted(rejected_reviews.items())),
        "variant": dict(sorted(by_variant.items())),
        "intent": dict(sorted(Counter(a.intent for a in admitted).items())),
        "variant_x_intent": dict(sorted(lang_intent.items())),
        "intents_covered": len(intents),
        "intents_missing": sorted(set(tax.intents) - intents),
        "intents_missing_by_variant": {
            v: sorted(set(tax.intents) - {a.intent for a in admitted if a.request.variant == v})
            for v in ("en", "ru", "kk", "mixed")
        },
        "independent_scenarios": len({a.request.scenario_id for a in admitted}),
        "scenarios_by_variant": {
            v: len({a.request.scenario_id for a in admitted if a.request.variant == v})
            for v in sorted(by_variant)
        },
        "reviewed_share_of_admitted": 1.0 if admitted else None,  # admission requires it
        "agreement": {
            v: {
                f: {
                    "pairs": len(ps),
                    "raw": round(sum(a == b for a, b in ps) / len(ps), 4) if ps else None,
                    "kappa": cohen_kappa(ps),
                }
                for f, ps in sorted(fields.items())
            }
            for v, fields in sorted(pairs.items())
        },
        "contributors": {
            "total": len(d.contributors),
            "qualified": sum(c.qualified for c in d.contributors),
            "roles": dict(sorted(roles.items())),
            "kk_competent": sum("kk" in c.languages for c in d.contributors),
            "ru_kk_competent": sum(c.speaks("mixed") for c in d.contributors),
        },
    }


def admitted_records(result: IntakeResult) -> list[dict[str, Any]]:
    """Admitted items as plain rows (answers included) for reports and the builder."""
    rows = []
    for a in result.admitted:
        r = a.request
        rows.append(
            {
                "item_id": r.item_id,
                "scenario_id": r.scenario_id,
                "variant": r.variant,
                "preferred_reply_lang": r.preferred_reply_lang,
                "method": r.method,
                "text": r.text,
                "context_json": json.dumps([t.model_dump() for t in r.context], ensure_ascii=False),
                "intent": a.intent,
                "action": a.action,
                "escalation_reason": a.escalation_reason,
                "label_status": a.label_status,
                "reviews": len(a.reviewers),
                "answer": a.answer.model_dump() if a.answer is not None else None,
            }
        )
    return rows
