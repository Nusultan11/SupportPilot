"""Placeholder catalogue, synthetic slot filling and the draft intent map."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supportpilot.common.config import find_project_root
from supportpilot.common.schemas import INTENTS
from supportpilot.data.labels import (
    IntentMap,
    IntentRule,
    has_reference,
    load_intent_map,
    map_labels,
)
from supportpilot.data.placeholders import (
    PlaceholderClass,
    fill_text,
    key_form,
    load_catalog,
    malformed,
    placeholders,
    resolve_response,
)

ROOT = find_project_root(Path(__file__).parent)
CATALOG = load_catalog(ROOT / "configs" / "placeholders.yaml")
IMAP = load_intent_map(ROOT / "configs" / "intent_map.yaml")


def test_placeholder_parsing_and_malformed() -> None:
    assert placeholders("cancel {{Order Number}} for {{ Person Name }}") == [
        "Order Number",
        "Person Name",
    ]
    assert malformed("cancel {{Order Number")
    assert malformed("cancel Order Number}}")
    assert not malformed("cancel {{Order Number}}")


def test_fill_is_deterministic_consistent_and_synthetic() -> None:
    tpl = (
        "cancel {{Order Number}} and {{Order Number}} for {{Person Name}}, "
        "{{Currency Symbol}}{{Refund Amount}}"
    )
    a = fill_text(tpl, CATALOG, "src:csv.1")
    b = fill_text(tpl, CATALOG, "src:csv.1")
    c = fill_text(tpl, CATALOG, "src:csv.2")
    assert a == b
    assert a.text != c.text  # values depend on the record
    order = a.slots["order_id"]
    assert a.text.count(order) == 2  # same placeholder -> same value in one row
    assert len(order) == 6
    assert order.isdigit()
    assert a.slots["person_name"] in CATALOG.slots["Person Name"].values
    assert a.slots["currency"] == "USD"
    assert "$" in a.text
    assert "{{" not in a.text
    assert a.unknown == []


def test_unknown_placeholder_is_reported_not_filled() -> None:
    f = fill_text("talk to {{Store Manager}}", CATALOG, "src:csv.1")
    assert f.unknown == ["Store Manager"]
    assert "{{Store Manager}}" in f.text


def test_key_form_masks_slot_values() -> None:
    assert key_form("track {{Order Number}} now", CATALOG) == "track  <order_id>  now"
    assert "<unknown>" in key_form("{{Whatever}}", CATALOG)


def test_response_resolution_fills_own_slots_and_classifies_the_rest() -> None:
    f = fill_text("track {{Order Number}}", CATALOG, "src:csv.9")
    r = resolve_response(
        "Order {{Order Number}}: call {{Customer Support Phone Number}}, open {{Order History}}, "
        "{{fill in the specific process, e.g., contacting our team}}",
        f.shown,
        CATALOG,
    )
    assert f.slots["order_id"] in r.text
    assert r.kept == {
        PlaceholderClass.KB_FACT.value: ["Customer Support Phone Number"],
        PlaceholderClass.TEMPLATE_INSTRUCTION.value: [
            "fill in the specific process, e.g., contacting our team"
        ],
        PlaceholderClass.UI_LABEL.value: ["Order History"],
    }


def test_real_intent_map_is_valid_and_uses_known_intents() -> None:
    mapped = [r for s in IMAP.sources.values() for r in s.values() if r.intent]
    assert mapped
    assert all(r.intent in INTENTS for r in mapped)
    assert IMAP.status.startswith("draft")


def test_needs_reference_decides_per_row() -> None:
    with_slot = map_labels(
        IMAP, "bitext_customer_support", "track_order", "where is 482913", {"order_id": "482913"}
    )
    literal = map_labels(IMAP, "bitext_customer_support", "check_invoice", "find bill #12588", {})
    without = map_labels(IMAP, "bitext_customer_support", "track_order", "where is my order", {})
    assert (with_slot.intent, with_slot.action) == ("track_order", "call_tool")
    assert literal.action == "call_tool"
    assert without.action == "clarify"


def test_escalation_and_unmapped_labels() -> None:
    esc = map_labels(IMAP, "bitext_retail_ecom", "human_agent", "agent please", {})
    assert esc.escalate
    assert esc.escalation_reason == "customer_requested_human"
    un = map_labels(IMAP, "bitext_retail_ecom", "technical_issue", "app broken", {})
    assert un.status == "unmapped"
    assert un.intent is None
    assert un.detail
    assert map_labels(IMAP, "belebele", None, "q", {}).status == "no_label"
    assert (
        map_labels(IMAP, "bitext_retail_ecom", "brand_new_intent", "x", {}).status == "not_in_map"
    )


def test_suspect_rule_rejects_rows_without_label_evidence() -> None:
    ok = map_labels(
        IMAP, "bitext_retail_ecom", "return_product_in_store", "how do I return it in store", {}
    )
    bad = map_labels(
        IMAP,
        "bitext_retail_ecom",
        "return_product_in_store",
        "information about a product in store",
        {},
    )
    assert ok.status == "mapped"
    assert bad.status == "suspect"
    assert bad.action is None


def test_has_reference() -> None:
    assert has_reference("order 1234567", {})
    assert has_reference("x", {"invoice_id": "INV-1"})
    assert not has_reference("cancel my order", {})


@pytest.mark.parametrize(
    "rule",
    [
        {"intent": "not_an_intent", "action": "answer_kb"},
        {"intent": None},  # no reason
        {"intent": "human_agent", "action": "escalate"},  # escalate flag missing
        {"intent": "complaint", "action": "escalate", "escalate": True},  # no reason
        {"intent": "track_order", "action": "teleport"},
        {"intent": "track_order", "action": "answer_kb", "suspect_unless": "("},
    ],
)
def test_invalid_intent_rules_are_rejected(rule: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        IntentRule.model_validate(rule)


def test_intent_map_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        IntentMap.model_validate({"status": "draft", "sources": {}, "extra": 1})
