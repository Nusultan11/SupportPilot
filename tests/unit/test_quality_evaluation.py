"""Guard fresh confirmation isolation and paired experimental accounting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supportpilot.service.quality_evaluation import bootstrap, compare, load_inputs, summarise


def row(identity: str, score: float = 0) -> dict[str, Any]:
    return {
        "id": identity,
        "scenario_id": identity,
        "family_id": "shared",
        "scope": "en",
        "intent": "track_order",
        "raw_intent": "track_order" if score else None,
        "text": "inspect",
        "critical": False,
        "expected_tool": "get_order",
        "expected_delivery": "tool",
        "expected_action": "call_tool",
        "followup_expected": False,
        "intent_correct": score,
        "action_correct": score,
        "delivery_correct": score,
        "critical_recall": 0.0,
        "reason_correct": 0.0,
        "false_critical": 0.0,
        "tool_success": score,
        "dialogue_success": 0.0,
        "seconds": 1.0,
        "http_status": 200,
    }


def test_fresh_inputs_cover_both_splits_without_claiming_human_review() -> None:
    data = load_inputs(Path(__file__).resolve().parents[2] / "authored/system/quality.yaml")
    assert len(data["needs"]) == 90
    assert len(data["followups"]) == 6


def test_pairing_refuses_population_and_expected_operation_changes() -> None:
    with pytest.raises(ValueError, match="IDs"):
        compare({"rows": [row("a")]}, {"rows": [row("b")]})
    with pytest.raises(ValueError, match="labels"):
        compare({"rows": [row("a")]}, {"rows": [{**row("a"), "expected_tool": "get_payment"}]})


def test_family_uncertainty_is_unknown_for_one_family() -> None:
    a, b = [row("a"), row("b")], [row("a", 1), row("b", 1)]
    assert bootstrap(a, b, "delivery_correct", family=True) is None
    assert bootstrap(a, b, "delivery_correct") == [1, 1]
    result = compare({"rows": a}, {"rows": b})
    assert result["en"]["macro_f1_35"]["delta"] == pytest.approx(1 / 35)
    assert result["kk"]["tool_success"]["delta"] is None


def test_failures_stay_in_tool_denominator_and_empty_risks_are_unknown() -> None:
    result = summarise([row("failed"), row("passed", 1)])["en"]
    assert result["metrics"]["tool_success"]["numerator"] == 1
    assert result["metrics"]["tool_success"]["denominator"] == 2
    assert result["metrics"]["critical_recall"]["rate"] is None
    assert result["critical_wilson95"] is None
