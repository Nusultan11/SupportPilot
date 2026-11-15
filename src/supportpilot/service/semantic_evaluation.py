"""Calibrated, independently labelled claim assessment; closed by default."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json


class ClaimReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    item_id: str
    family_id: str
    language: Literal["en", "ru", "kk", "mixed"]
    phase: Literal["calibration", "assessment"]
    author: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewer_kind: Literal["human"]
    fluent_languages: list[Literal["en", "ru", "kk"]]
    provenance: str = Field(min_length=1)
    judge_revision: str = Field(min_length=1)
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_support: Literal["supported", "unsupported", "unknown"]
    judge_support: Literal["supported", "unsupported", "unknown"]
    cited_claims: int = Field(ge=0)
    supported_cited_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    complete: bool


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    observed = sum(a == b for a, b in pairs) / len(pairs)
    chance = sum(left[c] * right[c] for c in left.keys() | right.keys()) / len(pairs) ** 2
    return (observed - chance) / (1 - chance) if chance < 1 else None


def evaluate_reviews(reviews: list[ClaimReview]) -> dict[str, object]:
    """Require distinct fluent humans, distinct assessment families and one pinned judge.

    Provenance is supplied by the review owner; software cannot certify a person's identity
    or fluency. Metrics stay closed if those declarations or calibration are absent.
    """
    ids = [(r.phase, r.language, r.item_id) for r in reviews]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate review items")
    for row in reviews:
        required = {"ru", "kk"} if row.language == "mixed" else {row.language}
        if row.author == row.reviewer or not required <= set(row.fluent_languages):
            raise ValueError("review must be independent and fluent in every required language")
        if row.supported_cited_claims > row.cited_claims:
            raise ValueError("supported citations exceed citation denominator")
        if row.human_support == "supported" and (
            row.unsupported_claims or row.supported_cited_claims != row.cited_claims
        ):
            raise ValueError("supported label contradicts claim/citation counts")
    if len({(r.judge_revision, r.rubric_sha256) for r in reviews}) > 1:
        raise ValueError("mixed judge/rubric identities")
    calibration = {r.family_id for r in reviews if r.phase == "calibration"}
    assessment = {r.family_id for r in reviews if r.phase == "assessment"}
    if calibration & assessment:
        raise ValueError("calibration and assessment families overlap")
    result: dict[str, object] = {}
    for language in ("en", "ru", "kk", "mixed"):
        cal = [r for r in reviews if r.language == language and r.phase == "calibration"]
        cohort = [r for r in reviews if r.language == language and r.phase == "assessment"]
        agreement = kappa([(r.human_support, r.judge_support) for r in cal])
        valid = len(cal) >= 100 and agreement is not None and agreement >= 0.6 and bool(cohort)
        cited = sum(r.cited_claims for r in cohort)
        result[language] = {
            "status": "EVALUATED" if valid else "NOT_EVALUATED",
            "calibration_items": len(cal),
            "calibration_kappa": agreement,
            "assessment_items": len(cohort),
            "semantic_citation_precision": sum(r.supported_cited_claims for r in cohort) / cited
            if valid and cited
            else None,
            "unsupported_claim_rate": sum(r.unsupported_claims > 0 for r in cohort) / len(cohort)
            if valid
            else None,
            "complete_answer_rate": sum(r.complete for r in cohort) / len(cohort)
            if valid
            else None,
            "reason": "fluent calibration >=100, kappa>=.6 and fresh assessment required"
            if not valid
            else "human labels; judge calibrated on separate families",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [ClaimReview.model_validate(r) for r in read_json(args.reviews)] if args.reviews else []
    result = {
        "reviews_sha256": sha256_file(args.reviews) if args.reviews else None,
        "review_identity_scope": "human identity/fluency declarations require owner verification",
        "metrics": evaluate_reviews(rows),
    }
    write_json(args.output, result)
    print(json.dumps({"path": str(args.output)}))


if __name__ == "__main__":
    main()
