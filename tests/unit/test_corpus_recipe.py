"""Recipe validation, label decisions and action derivation (data/recipe.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supportpilot.common.config import find_project_root
from supportpilot.common.manifest import load_source_registry
from supportpilot.common.schemas import Action
from supportpilot.data.recipe import (
    LabelDecision,
    LabelSpec,
    SourceRecipe,
    decide_label,
    derive_action,
    has_order_reference,
    load_recipe,
)
from supportpilot.data.records import ActionRule, LabelStatus
from supportpilot.data.taxonomy import load_taxonomy

ROOT = find_project_root(Path(__file__).parent)


def _src(**labels: Any) -> SourceRecipe:
    return SourceRecipe.model_validate(
        {
            "domain": "retail_support",
            "origin": {"en": "human_written"},
            "evidence": "test",
            "veto": [{"pattern": r"\bolly\b", "reason": "wake word"}],
            "labels": labels,
        }
    )


SRC = _src(
    weather={"intent": "out_of_scope"},
    promo={"rules": [{"cue": r"\bpromo|coupon\b", "intent": "policy_faq"}]},
    stock={
        "veto": [{"pattern": "complain", "reason": "two intents fit"}],
        "rules": [{"cue": "in stock", "intent": "availability"}],
    },
    cost={"candidate": "delivery_options_cost"},
    subscription={"exclude": "no v1 intent"},
)


def _decide(label: str, text: str) -> LabelDecision:
    return decide_label("demo", SRC, label, text, len(text.split()), 3)


def test_direct_cue_and_unverified_paths() -> None:
    assert _decide("weather", "will it rain today").status is LabelStatus.MAPPED_DIRECT
    short = _decide("weather", "rain today")
    assert short.status is LabelStatus.UNVERIFIED
    assert short.rule.endswith(":short")
    cue = _decide("promo", "my promo code is invalid")
    assert (cue.intent, cue.status) == ("policy_faq", LabelStatus.MAPPED_CUE)
    no_cue = _decide("promo", "I have a problem please help")
    assert (no_cue.intent, no_cue.status) == ("policy_faq", LabelStatus.UNVERIFIED)
    bare = _decide("cost", "shipping was way too expensive")
    assert (bare.intent, bare.status) == ("delivery_options_cost", LabelStatus.UNVERIFIED)


def test_exclusions_and_vetoes() -> None:
    assert _decide("subscription", "pay my subscription now").reason == "no v1 intent"
    assert _decide("weather", "olly will it rain").reason == "wake word"
    assert _decide("stock", "I complain it is never in stock").reason == "two intents fit"
    assert _decide("unknown_label", "some text here").intent is None


def test_label_spec_needs_exactly_one_kind_and_known_intents() -> None:
    with pytest.raises(ValidationError):
        LabelSpec.model_validate({"intent": "greeting", "exclude": "x"})
    with pytest.raises(ValidationError):
        LabelSpec.model_validate({})
    with pytest.raises(ValidationError):
        LabelSpec.model_validate({"intent": "not_an_intent"})


def test_derive_action_follows_taxonomy_conditions() -> None:
    tax = load_taxonomy(ROOT / "configs" / "taxonomy.yaml")
    d = LabelDecision(intent="track_order", status=LabelStatus.MAPPED_CUE, rule="r")
    assert derive_action(tax, "track_order", "where is my order", d, None)[2] is (
        ActionRule.REFERENCE_MISSING
    )
    act, _, rule = derive_action(tax, "track_order", "where is order #4512345", d, None)
    assert (act, rule) == (Action.CALL_TOOL, ActionRule.REFERENCE_PRESENT)
    import re

    brand = re.compile(r"\bguess\b", re.IGNORECASE)
    assert derive_action(tax, "availability", "are Guess boots in stock", d, brand)[0] is (
        Action.CALL_TOOL
    )
    assert derive_action(tax, "product_info", "tell me about boots", d, brand)[0] is (
        Action.CLARIFY
    )
    act, reason, rule = derive_action(tax, "technical_issue", "site is down", d, None)
    assert (act, reason, rule) == (Action.ESCALATE, "technical_issue", ActionRule.TAXONOMY_DEFAULT)
    esc = LabelDecision(
        intent="payment_problem",
        status=LabelStatus.MAPPED_CUE,
        rule="r",
        action=Action.ESCALATE,
        escalation_reason="payment_dispute",
    )
    assert derive_action(tax, "payment_problem", "x", esc, None) == (
        Action.ESCALATE,
        "payment_dispute",
        ActionRule.RECIPE_OVERRIDE,
    )


def test_order_reference_detection() -> None:
    assert has_order_reference("my order number is 12345678")
    assert has_order_reference("заказ № 5512")
    assert has_order_reference("тапсырыс нөмірі 4411")
    assert not has_order_reference("I ordered 2 shirts")


def test_project_recipe_is_valid_and_registered() -> None:
    recipe = load_recipe(ROOT / "configs" / "corpus.yaml")
    registry = load_source_registry(ROOT / "configs" / "sources.yaml")
    selected = {s.source_id for s in registry.sources if s.status.value == "selected"}
    assert set(recipe.sources) == selected  # every selected source is decided in the recipe
    assert recipe.sources["xsid_kk"].placement == "heldout"
    assert not recipe.sources["mangi_kk_support_qa"].active
    assert "yes" in recipe.sources["clinc150"].labels  # YAML booleans are quoted
