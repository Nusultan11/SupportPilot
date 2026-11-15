"""Protect experiment accounting and semantic acceptance boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from supportpilot.retrieval.stress import verify_resume
from supportpilot.service.evaluation import load_needs, macro_f1, paired
from supportpilot.service.performance import latency
from supportpilot.service.semantic_evaluation import ClaimReview, evaluate_reviews, kappa


def observation(identity: str = "one", score: float = 0) -> dict[str, Any]:
    return {
        "id": identity,
        "scenario_id": identity,
        "family_id": "same_family",
        "scope": "en",
        "intent": "track_order",
        "text": "Where is my parcel?",
        "expected_delivery": "tool",
        "critical": False,
        "delivery_correct": score,
    }


def test_pairing_rejects_population_and_label_changes() -> None:
    left = [observation()]
    with pytest.raises(ValueError, match="IDs differ"):
        paired(left, [observation("other")])
    with pytest.raises(ValueError, match="labels/groups"):
        paired(left, [{**observation(), "expected_delivery": "queued"}])
    with pytest.raises(ValueError, match="duplicate"):
        paired(left * 2, left * 2)


def test_pairing_preserves_failed_observations_and_family_correlation() -> None:
    left = [observation("one"), observation("two")]
    right = [observation("one", 1), observation("two", 1)]
    result = paired(left, right)
    assert result["en"]["scenario"]["mean"] == 1
    assert result["en"]["family"]["scenarios"] == 1
    assert result["kk"]["scenario"]["mean"] is None


def test_missing_route_is_failure_in_macro_f1() -> None:
    correct = {**observation(), "raw_intent": "track_order"}
    assert macro_f1([correct]) == pytest.approx(1 / 35)
    assert macro_f1([correct, observation("missing")]) == pytest.approx((2 / 3) / 35)


def test_frozen_needs_have_family_disjointness_and_honest_units() -> None:
    needs = load_needs(Path(__file__).resolve().parents[2] / "authored/system/scenarios.yaml")
    assert len(needs) == 120
    assert sum(n.phase == "holdout" and n.critical for n in needs) == 50
    assert len({n.need_id for n in needs if n.phase == "holdout"}) == 100
    assert sum(bool(n.followups) for n in needs) == 4


def review(index: int, *, phase: str = "calibration") -> ClaimReview:
    return ClaimReview.model_validate(
        {
            "item_id": f"{phase}_{index}",
            "family_id": f"{phase}_family_{index}",
            "language": "kk",
            "phase": phase,
            "author": "author",
            "reviewer": "reviewer",
            "reviewer_kind": "human",
            "fluent_languages": ["kk"],
            "provenance": "signed-review",
            "judge_revision": "fixed-judge-revision",
            "rubric_sha256": "a" * 64,
            "human_support": "supported" if index % 2 else "unsupported",
            "judge_support": "supported" if index % 2 else "unsupported",
            "cited_claims": 2,
            "supported_cited_claims": 2 if index % 2 else 1,
            "unsupported_claims": 0 if index % 2 else 1,
            "complete": False,
        }
    )


def test_semantic_metrics_require_calibration_and_fresh_assessment() -> None:
    assert evaluate_reviews([])["kk"]["status"] == "NOT_EVALUATED"  # type: ignore[index]
    cal = [review(i) for i in range(100)]
    assert evaluate_reviews(cal)["kk"]["status"] == "NOT_EVALUATED"  # type: ignore[index]
    cohort = [*cal, review(100, phase="assessment")]
    result: Any = evaluate_reviews(cohort)["kk"]
    assert result["status"] == "EVALUATED"
    assert result["semantic_citation_precision"] == 0.5
    assert result["unsupported_claim_rate"] == 1


def test_semantic_review_cannot_certify_itself_or_reuse_families() -> None:
    with pytest.raises(ValueError, match="independent"):
        evaluate_reviews([review(1).model_copy(update={"reviewer": "author"})])
    with pytest.raises(ValueError, match="overlap"):
        evaluate_reviews(
            [
                review(1),
                review(2, phase="assessment").model_copy(update={"family_id": review(1).family_id}),
            ]
        )
    with pytest.raises(ValueError, match="fluent"):
        evaluate_reviews([review(1).model_copy(update={"fluent_languages": ["en"]})])


def test_single_class_agreement_does_not_establish_judge_calibration() -> None:
    assert kappa([("supported", "supported")] * 100) is None
    assert kappa([("supported", "unsupported"), ("unsupported", "supported")]) == -1


def test_latency_has_explicit_empty_and_cold_scope() -> None:
    assert latency([])["warm_p95_seconds"] is None
    result = latency([20, 1, 2])
    assert result["first_seconds"] == 20
    assert result["warm_p50_seconds"] == 1.5


def test_stress_resume_rejects_changed_state_and_preserves_complete_upload() -> None:
    vectors = np.eye(4, dtype=np.float32)
    previous = {
        "status": "FAILED",
        "collection": "isolated",
        "source": {"identity": "source"},
        "upload": {"vectors": 4},
        "hnsw_build": {"points": 4},
    }
    points = [
        SimpleNamespace(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "synthetic_only": True,
                "chunk_id": f"load_{i}",
                "lang": ("ru", "kk", "en")[i % 3],
                "revision": 0,
            },
        )
        for i in range(4)
    ]

    class Client:
        def count(self, *args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(count=4)

        def retrieve(self, *args: object, **kwargs: object) -> list[SimpleNamespace]:
            return points

    client = Client()
    assert verify_resume(client, "isolated", vectors, previous, "source")["sampled_points"] == 4
    with pytest.raises(ValueError, match="same complete upload"):
        verify_resume(client, "isolated", vectors, previous, "different_source")
    with pytest.raises(ValueError, match="incremental"):
        verify_resume(
            client, "isolated", vectors, {**previous, "incremental": {"changed": 1}}, "source"
        )
    points[1].vector = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="sample changed"):
        verify_resume(client, "isolated", vectors, previous, "source")
