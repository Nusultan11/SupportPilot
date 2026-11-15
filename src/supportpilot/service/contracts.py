"""Public service boundaries; caller identity never comes from message content."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.schemas import AssistantResponse, LanguageCode


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChatRequest(Contract):
    message: str = Field(min_length=1, max_length=2000, pattern=r"\S")
    session_id: UUID | None = None
    request_id: UUID
    language: LanguageCode | None = None
    confirm_cancellation: bool = False


class Source(Contract):
    chunk_id: str
    doc_id: str
    locator: str
    text: str
    kb_version: str
    index_version: str


class ChatResult(Contract):
    session_id: UUID
    request_id: UUID
    response: AssistantResponse
    sources: list[Source] = Field(default_factory=list)
    escalation_id: UUID | None = None
    code_switch: bool = False
    delivery: Literal["tool", "clarification", "queued", "fallback", "refusal"]
    diagnostic: str


class Escalation(Contract):
    escalation_id: UUID
    session_id: UUID
    status: Literal["open", "in_progress", "resolved"]
    reason: str
    language: LanguageCode
    context: list[str]
    evidence: list[str]
    created_at: float


class EscalationUpdate(Contract):
    status: Literal["in_progress", "resolved"]


class ServiceError(Exception):
    def __init__(self, status: int, code: str) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
