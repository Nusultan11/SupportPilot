"""Intent taxonomy v1 and the source-label -> v1 relabelling (Stage 4, ADR-023).

`configs/taxonomy.yaml` is the single source of the v1 taxonomy: intents with their default
action, tool, KB documents and the clarify / escalate / refuse conditions, plus the mapping
of every train-source label. Labels are never guessed:
- a label maps to one intent, or
- ordered `rules` (regexes on the message) split a label whose texts carry two requests;
  a text that matches no rule stays unlabelled, or
- the label is excluded with a reason.
The only per-row decision besides rules is `needs_reference` (call_tool when the message
carries an order/invoice reference, else clarify), shared with the Stage 3 draft map.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.schemas import INTENTS, Action
from supportpilot.data.labels import NEEDS_REFERENCE, has_reference


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentSpec(_Model):
    group: str
    action: str
    description: str
    tool: str | None = None
    kb: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    clarify_when: str | None = None
    escalate_when: str | None = None
    refuse_when: str | None = None


class LabelRule(_Model):
    pattern: str
    intent: str | None = None
    exclude: str | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.intent is None) == (self.exclude is None):
            raise ValueError("a rule sets exactly one of intent / exclude")
        re.compile(self.pattern)
        return self


class LabelMap(_Model):
    intent: str | None = None
    action: str | None = None  # overrides the intent's default action for this label
    rules: list[LabelRule] = Field(default_factory=list)
    exclude: str | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        kinds = sum((self.intent is not None, bool(self.rules), self.exclude is not None))
        if kinds != 1:
            raise ValueError("a source label has exactly one of intent / rules / exclude")
        return self


class Taxonomy(_Model):
    schema_version: Literal["taxonomy/1"] = "taxonomy/1"
    version: str
    frozen_at: str
    actions: list[str]
    tools: list[str]
    escalation_reasons: dict[str, str]
    intents: dict[str, IntentSpec]
    sources: dict[str, dict[str, LabelMap]]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if set(self.actions) != {a.value for a in Action}:
            raise ValueError("actions must equal the Action enum")
        if set(self.intents) != INTENTS:
            missing, extra = INTENTS - set(self.intents), set(self.intents) - INTENTS
            raise ValueError(f"intents differ from schemas.INTENTS: -{missing} +{extra}")
        allowed = {*self.actions, NEEDS_REFERENCE}
        for name, spec in self.intents.items():
            if spec.action not in allowed:
                raise ValueError(f"{name}: unknown action {spec.action!r}")
            if spec.tool is not None and spec.tool not in self.tools:
                raise ValueError(f"{name}: unknown tool {spec.tool!r}")
            if (spec.action == Action.ESCALATE.value) != (spec.escalation_reason is not None):
                raise ValueError(f"{name}: escalate intents need exactly one escalation_reason")
            if spec.escalation_reason and spec.escalation_reason not in self.escalation_reasons:
                raise ValueError(f"{name}: unknown escalation reason {spec.escalation_reason!r}")
            if spec.action == NEEDS_REFERENCE and spec.tool is None:
                raise ValueError(f"{name}: needs_reference requires a tool")
        for sid, labels in self.sources.items():
            for label, m in labels.items():
                targets = [m.intent] + [r.intent for r in m.rules]
                for t in targets:
                    if t is not None and t not in self.intents:
                        raise ValueError(f"{sid}:{label}: unknown intent {t!r}")
                if m.action is not None and m.action not in allowed:
                    raise ValueError(f"{sid}:{label}: unknown action {m.action!r}")
        return self

    def escalation_reason(self, intent: str) -> str | None:
        return self.intents[intent].escalation_reason


def load_taxonomy(path: Path) -> Taxonomy:
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class V1Label(BaseModel):
    status: Literal["mapped", "excluded", "unlabelled", "not_in_map"]
    intent: str | None = None
    action: str | None = None
    escalate: bool = False
    escalation_reason: str | None = None
    rule: str | None = None  # the rule pattern that decided, if any
    detail: str | None = None


def resolve_action(tax: Taxonomy, intent: str, action: str | None, has_ref: bool) -> str:
    a = action or tax.intents[intent].action
    if a == NEEDS_REFERENCE:
        return Action.CALL_TOOL.value if has_ref else Action.CLARIFY.value
    return a


def relabel(
    tax: Taxonomy, source_id: str, source_intent: str | None, text: str, slots: dict[str, str]
) -> V1Label:
    m = tax.sources.get(source_id, {}).get(source_intent or "")
    if m is None:
        return V1Label(status="not_in_map", detail=f"{source_id}:{source_intent} not in taxonomy")
    if m.exclude is not None:
        return V1Label(status="excluded", detail=m.exclude)
    intent, rule_used = m.intent, None
    if m.rules:
        for r in m.rules:
            if re.search(r.pattern, text, re.IGNORECASE):
                if r.exclude is not None:
                    return V1Label(status="excluded", detail=r.exclude, rule=r.pattern)
                intent, rule_used = r.intent, r.pattern
                break
        else:
            return V1Label(status="unlabelled", detail="no rule matched the text")
    assert intent is not None
    action = resolve_action(tax, intent, m.action, has_reference(text, slots))
    reason = tax.escalation_reason(intent) if action == Action.ESCALATE.value else None
    return V1Label(
        status="mapped",
        intent=intent,
        action=action,
        escalate=action == Action.ESCALATE.value,
        escalation_reason=reason,
        rule=rule_used,
    )
