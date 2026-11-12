from __future__ import annotations

import pytest

from supportpilot.retrieval.abstain import (
    balanced_accuracy,
    evidence_gate,
    preserves_multilingual_recall,
)
from supportpilot.retrieval.models import ScoredChunk
from tests.unit.test_retrieval import chunk


def evidence(score: float | None, doc: str) -> ScoredChunk:
    source = chunk(doc_id=doc)
    return ScoredChunk(
        chunk_id=source.chunk_id,
        kb_version="kb",
        index_version="index",
        chunk=source,
        score=score or 0,
        rerank_score=score,
        matched_unit_ids=["unit"],
    )


def test_query_gate_preserves_weak_background_when_direct_evidence_is_strong() -> None:
    rows = [evidence(0.8, "direct"), evidence(0.001, "background")]
    assert evidence_gate(rows, 0.5) == rows
    assert evidence_gate(rows, 0.9) == []
    assert evidence_gate(rows, 0.8) == rows
    assert evidence_gate([], 0.5) == []


def test_query_gate_cannot_apply_reranker_calibration_to_dense_scores() -> None:
    rows = [evidence(None, "policy")]
    assert evidence_gate(rows, None) == rows
    with pytest.raises(ValueError, match="requires reranker"):
        evidence_gate(rows, 0.5)


def test_answerability_calibration_groups_variants_and_ignores_holdout() -> None:
    correct = {"scenario_id": "a", "phase": "development", "answerable": True, "returned": 10}
    wrong = {"scenario_id": "b", "phase": "development", "answerable": True, "returned": 0}
    negative = {"scenario_id": "c", "phase": "development", "answerable": False, "returned": 0}
    rows = [correct] * 100 + [wrong, negative]
    assert balanced_accuracy(rows) == 0.75
    assert balanced_accuracy(rows + [{**correct, "phase": "holdout"}] * 1000) == 0.75


def test_gate_cannot_hide_cross_lingual_or_code_switch_regression() -> None:
    baseline = {
        scope: {"recall5": {"mean": 0.95}}
        for scope in ("ru", "kk", "en", "cross_lingual", "code_switch")
    }
    assert preserves_multilingual_recall(baseline, baseline)
    for scope in baseline:
        degraded = {**baseline, scope: {"recall5": {"mean": 0.84}}}
        assert not preserves_multilingual_recall(degraded, baseline)
