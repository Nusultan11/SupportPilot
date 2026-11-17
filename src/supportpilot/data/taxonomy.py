"""Intent taxonomy v1 (`configs/taxonomy.yaml`): intents, default actions, tools, escalation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from supportpilot.common.schemas import INTENTS, Action

NEEDS_REFERENCE = "needs_reference"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntentSpec(_Model):
    group: str
    action: str
    description: str
    tool: str | None = None
    escalation_reason: str | None = None
    clarify_when: str | None = None
    escalate_when: str | None = None
    refuse_when: str | None = None


class Taxonomy(_Model):
    schema_version: Literal["taxonomy/1"] = "taxonomy/1"
    version: str
    frozen_at: str
    actions: list[str]
    tools: list[str]
    escalation_reasons: dict[str, str]
    intents: dict[str, IntentSpec]

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
        return self


def load_taxonomy(path: Path) -> Taxonomy:
    return Taxonomy.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
