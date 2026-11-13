from __future__ import annotations

import json
from datetime import date

import pytest

from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import Action, AssistantResponse, KBChunk, Locator
from supportpilot.llm.generation import build_messages, generate_grounded
from supportpilot.retrieval.models import RetrievalContext, ScoredChunk


class FakeBackend:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.calls = 0

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return self.raw


@pytest.fixture
def context() -> RetrievalContext:
    text = "Returns are accepted within 14 days."
    chunk = KBChunk(
        chunk_id="returns:aaaaaaaa:0000",
        doc_id="returns",
        version="a" * 64,
        lang="en",
        position=0,
        section_id="window",
        section_path=["Returns"],
        title="Returns",
        text=text,
        locator=Locator(line_range=(1, 1)),
        doc_type="policy",
        format="txt",
        source_path="returns.txt",
        effective_from=date(2026, 1, 1),
        token_count=12,
        translation_group="returns.window",
        checksum=sha256_text(text),
    )
    return RetrievalContext(
        kb_version="kb",
        index_version="index",
        text=text,
        token_count=12,
        budget=1024,
        truncated=False,
        abstain=False,
        evidence=[
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                chunk=chunk,
                kb_version="kb",
                index_version="index",
                score=1,
                matched_unit_ids=["unit"],
            )
        ],
    )


def response(answer: str = "Returns are accepted within 14 days.") -> AssistantResponse:
    return AssistantResponse(
        language="en",
        intent="return_policy",
        action=Action.ANSWER_KB,
        answer=answer,
        citations=["returns:aaaaaaaa:0000"],
    )


def test_empty_context_does_not_call_model(context: RetrievalContext) -> None:
    backend = FakeBackend("invalid")
    result = generate_grounded(
        backend,
        "Returns?",
        "en",
        context.model_copy(update={"evidence": []}),
        identify_language=lambda _: "en",
    )
    assert result.fallback
    assert backend.calls == 0
    assert result.response.needs_human


def test_digits_are_not_semantic_proof(context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(response().model_dump_json()),
        "Returns?",
        "en",
        context,
        identify_language=lambda _: "en",
    )
    assert result.fallback
    assert result.semantic_support == "unknown"
    assert "semantic_support:unknown" in result.problems


def test_explicit_support_can_pass(context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(response().model_dump_json()),
        "Returns?",
        "en",
        context,
        identify_language=lambda _: "en",
        verify_claims=lambda *_: "supported",
    )
    assert not result.fallback
    assert result.response == response()


@pytest.mark.parametrize("raw", ["{}", "```json\n{}\n```", "[]", "not JSON"])
def test_invalid_outputs_fail_closed(raw: str, context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(raw),
        "Returns?",
        "en",
        context,
        identify_language=lambda _: "en",
    )
    assert result.problems == ("invalid_schema",)


def test_customer_number_cannot_authorize_policy(context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(response("You can return within 90 days.").model_dump_json()),
        "Can I return within 90 days?",
        "en",
        context,
        identify_language=lambda _: "en",
        verify_claims=lambda *_: "supported",
    )
    assert result.fallback
    assert "unsupported_number:90" in result.problems


def test_nonnumeric_fabrication_needs_support(context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(response("Opened cosmetics are always returnable.").model_dump_json()),
        "Opened cosmetics?",
        "en",
        context,
        identify_language=lambda _: "en",
        verify_claims=lambda *_: "unsupported",
    )
    assert result.fallback
    assert result.semantic_support == "unsupported"


def test_required_escalation_and_language(context: RetrievalContext) -> None:
    result = generate_grounded(
        FakeBackend(response().model_dump_json()),
        "Get a human",
        "ru",
        context,
        identify_language=lambda _: "en",
        verify_claims=lambda *_: "supported",
        mandatory_escalation=True,
    )
    assert result.fallback
    assert "language_mismatch" in result.problems
    assert "mandatory_escalation_missing" in result.problems
    assert result.response.language == "ru"


def test_unshown_context_text_is_not_exposed(context: RetrievalContext) -> None:
    messages = build_messages("Returns?", "en", context.model_copy(update={"text": "secret"}))
    payload = json.loads(messages[1]["content"])
    assert "secret" not in messages[1]["content"]
    assert payload["evidence"][0]["chunk_id"] == "returns:aaaaaaaa:0000"
