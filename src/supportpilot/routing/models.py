"""Stage 7 RouteDecision boundary for Stage 8 integration."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.schemas import Action, Intent, LanguageCode


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["route_decision/1"] = "route_decision/1"
    language: LanguageCode
    code_switch: bool = False
    intent: Intent
    action: Action
    confidence: float | None = Field(default=None, ge=0, le=1)
    slots: dict[str, str] = Field(default_factory=dict)
    escalate: bool = False
    escalation_reason: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.escalate != bool(self.escalation_reason):
            raise ValueError("escalation needs exactly one reason")
        if self.action is Action.ESCALATE and not self.escalate:
            raise ValueError("escalate action requires escalate flag")
        return self
