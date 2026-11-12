"""Source label -> SupportPilot intent/action mapping (draft v0, ADR-021).

Nothing is inferred from text semantics. A row gets labels only through an explicit entry
in `configs/intent_map.yaml`; the one per-row decision is `needs_reference`, which checks
whether the message contains an order/invoice reference (a filled `{{Order Number}}` /
`{{Invoice Number}}` slot or a literal "#12345" / "order 12345").
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from supportpilot.common.schemas import INTENTS_V0, Action

NEEDS_REFERENCE = "needs_reference"
REFERENCE_SLOTS = frozenset({"order_id", "invoice_id"})
_LITERAL_REF_RE = re.compile(
    r"#\s?\d{3,}|\b(?:order|purchase|invoice|bill|receipt)\s*(?:number|no\.?|#)?\s*#?\d{4,}\b",
    re.IGNORECASE,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentRule(_Model):
    intent: str | None
    action: str | None = None
    escalate: bool = False
    escalation_reason: str | None = None
    reason: str | None = None  # why there is no v0 intent
    note: str | None = None
    # Measured source label noise: a row whose text does NOT match this regex is not
    # trusted (status "suspect", routed to pending). Never used to assign a label.
    suspect_unless: str | None = None

    @model_validator(mode="after")
    def _valid(self) -> IntentRule:
        if self.intent is None:
            if not self.reason:
                raise ValueError("an unmapped label (intent: null) needs a reason")
            if self.action is not None or self.escalate:
                raise ValueError("an unmapped label cannot have an action or escalation")
            return self
        if self.intent not in INTENTS_V0:
            raise ValueError(f"unknown v0 intent {self.intent!r}")
        allowed = {a.value for a in Action} | {NEEDS_REFERENCE}
        if self.action not in allowed:
            raise ValueError(f"action must be one of {sorted(allowed)}")
        if (self.action == Action.ESCALATE.value) != self.escalate:
            raise ValueError("escalate must be true exactly when action is escalate")
        if self.escalate and not self.escalation_reason:
            raise ValueError("escalate needs escalation_reason")
        if self.suspect_unless:
            try:
                re.compile(self.suspect_unless)
            except re.error as exc:
                raise ValueError(f"suspect_unless is not a valid regex: {exc}") from exc
        return self


class IntentMap(_Model):
    schema_version: Literal["intent_map/1"] = "intent_map/1"
    status: str
    sources: dict[str, dict[str, IntentRule]]

    def rule(self, source_id: str, source_intent: str | None) -> IntentRule | None:
        if source_intent is None:
            return None
        return self.sources.get(source_id, {}).get(source_intent)


def load_intent_map(path: Path) -> IntentMap:
    return IntentMap.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def has_reference(text: str, slots: dict[str, str]) -> bool:
    return bool(REFERENCE_SLOTS & slots.keys()) or bool(_LITERAL_REF_RE.search(text))


class Labels(BaseModel):
    status: Literal["mapped", "unmapped", "no_label", "not_in_map", "suspect"]
    intent: str | None = None
    action: str | None = None
    escalate: bool = False
    escalation_reason: str | None = None
    detail: str | None = None


def map_labels(
    imap: IntentMap, source_id: str, source_intent: str | None, text: str, slots: dict[str, str]
) -> Labels:
    if source_intent is None:
        return Labels(status="no_label", detail="source has no intent label")
    rule = imap.rule(source_id, source_intent)
    if rule is None:
        return Labels(status="not_in_map", detail=f"{source_id}:{source_intent} not in intent map")
    if rule.intent is None:
        return Labels(status="unmapped", detail=rule.reason)
    if rule.suspect_unless and not re.search(rule.suspect_unless, text, re.IGNORECASE):
        return Labels(
            status="suspect",
            intent=rule.intent,
            detail=f"text does not match the evidence pattern for {source_intent}",
        )
    action = rule.action
    if action == NEEDS_REFERENCE:
        action = Action.CALL_TOOL.value if has_reference(text, slots) else Action.CLARIFY.value
    return Labels(
        status="mapped",
        intent=rule.intent,
        action=action,
        escalate=rule.escalate,
        escalation_reason=rule.escalation_reason,
    )
