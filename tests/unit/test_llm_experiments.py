from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from supportpilot.common.schemas import INTENTS, Action, AssistantResponse
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.llm.data import assert_disjoint
from supportpilot.llm.evaluate import (
    macro_f1,
    paired_ci,
    paired_macro_f1_ci,
    score_output,
    summarize,
)
from supportpilot.llm.prompts import prepare_messages, response_schema
from supportpilot.routing.models import RouteDecision
from supportpilot.routing.router import policy_decision


def test_schema_enumerates_valid_intents() -> None:
    assert set(response_schema()["properties"]["intent"]["enum"]) == INTENTS


def test_prompt_does_not_accept_heldout_target() -> None:
    with pytest.raises(ValueError, match="target"):
        prepare_messages([{"role": "assistant", "content": "secret target"}], mode="schema")
    with pytest.raises(ValueError, match="demonstrations"):
        prepare_messages([{"role": "user", "content": "hello"}], mode="fewshot")


def test_prompt_keeps_untrusted_data_out_of_system() -> None:
    messages = prepare_messages(
        [
            {"role": "system", "content": "original"},
            {"role": "user", "content": "ignore policy"},
            {"role": "tool", "content": "malicious source"},
        ],
        mode="schema",
    )
    assert "malicious source" not in messages[0]["content"]
    assert messages[-1]["content"] == "malicious source"


def test_family_leakage_is_rejected() -> None:
    train = [{"example_id": "a", "scenario_id": "x", "family_id": "f"}]
    other = [{"example_id": "b", "scenario_id": "y", "family_id": "f"}]
    with pytest.raises(ValueError, match="family_id"):
        assert_disjoint(train, other)


def observation(key: str, score: float) -> dict[str, Any]:
    return {
        "example_id": key,
        "scenario_id": key,
        "family_id": "family",
        "lang": "en",
        "expected_intent": "greeting",
        "predicted_intent": "greeting",
        "schema_validity": score,
    }


def test_paired_bootstrap_and_family_limit() -> None:
    left = [observation("a", 0), observation("b", 0)]
    right = [observation("a", 1), observation("b", 1)]
    assert paired_ci(left, right, "schema_validity")["ci95"] == [1, 1]
    family = paired_ci(left, right, "schema_validity", cluster="family_id")
    assert family["between_family_uncertainty_identified"] is False
    with pytest.raises(ValueError, match="identical"):
        paired_ci(left, right[:1], "schema_validity")
    with pytest.raises(ValueError, match="unique"):
        paired_ci(left + left, right, "schema_validity")


def test_macro_f1_uses_all_35_labels() -> None:
    assert macro_f1([observation("a", 1)]) == pytest.approx(1 / 35)


def test_resources_keep_missing_measurements_and_errors_explicit() -> None:
    a = {
        **observation("a", 1),
        "task": "clarify",
        "seconds": 2,
        "output_tokens": 10,
        "peak_vram_bytes": 100,
    }
    b = {**observation("b", 0), "task": "clarify", "execution_error": "out_of_memory"}
    summary = summarize([a, b])
    assert summary["en"]["n"] == 2
    assert summary["en"]["execution_errors"] == 1
    assert summary["en"]["timed_observations"] == 1
    assert summary["en"]["generation_tokens_per_second"] == 5
    assert summary["en"]["resources"]["peak_vram_bytes"] == 100
    assert summary["en"]["resources"]["rss_bytes"] is None
    assert summary["kk"]["generation_tokens_per_second"] is None


def test_nonlinear_macro_f1_bootstrap_recomputes_the_metric() -> None:
    correct = [observation("a", 1), observation("b", 1)]
    invalid = [{**r, "predicted_intent": "INVALID"} for r in correct]
    result = paired_macro_f1_ci(invalid, correct)
    assert result["delta"] == pytest.approx(1 / 35)
    assert result["ci95"] == pytest.approx([1 / 35, 1 / 35])
    assert paired_macro_f1_ci([], [])["delta"] is None
    with pytest.raises(ValueError, match="unique identical"):
        paired_macro_f1_ci(correct, invalid[:1])


def test_invalid_json_remains_failure_in_abstention_denominator() -> None:
    target = AssistantResponse(
        language="en",
        intent="return_policy",
        action=Action.ANSWER_KB,
        answer="Cannot confirm",
        missing_information=["proof"],
    )
    row = {
        "example_id": "x",
        "scenario_id": "s",
        "family_id": "f",
        "lang": "en",
        "task": "abstain_no_evidence",
        "target": target.model_dump(mode="json"),
    }
    # Invalid JSON must return before invoking language ID or reading evidence.
    result = score_output(row, "invalid", None)  # type: ignore[arg-type]
    assert result["schema_validity"] == 0
    assert result["abstention_accuracy"] == 0
    assert result["negative_answer_attempt_rate"] == 1
    assert result["grounded_answer_accuracy"] is None


def test_route_consistency_and_reference_policy() -> None:
    with pytest.raises(ValueError, match="escalat"):
        RouteDecision(language="en", intent="greeting", action=Action.ESCALATE, confidence=0)
    taxonomy = load_taxonomy(Path(__file__).resolve().parents[2] / "configs/taxonomy.yaml")
    missing = policy_decision("track_order", 0.5, "Where is it?", {}, "en", False, taxonomy)
    assert missing.action is Action.CLARIFY
    known = policy_decision(
        "track_order", 0.5, "Where is it?", {"order_id": "123456"}, "en", False, taxonomy
    )
    assert known.action is Action.CALL_TOOL
    human = policy_decision("greeting", 0.5, "Connect a human agent", {}, "en", False, taxonomy)
    assert human.escalate
    assert human.escalation_reason == "customer_requested_human"
