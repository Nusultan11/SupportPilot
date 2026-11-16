from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import read_json
from supportpilot.common.schemas import Action, AssistantResponse, KBChunk, Locator
from supportpilot.retrieval.models import RetrievalContext, ScoredChunk
from supportpilot.service.manual_testing import (
    InspectionRequest,
    ManualWorkbench,
    inspect_completion,
)


@pytest.fixture
def packed() -> RetrievalContext:
    text = "Returns are accepted within 14 days."
    chunk = KBChunk(
        chunk_id="returns:aaaaaaaa:0000",
        doc_id="returns",
        version="a" * 64,
        lang="en",
        position=0,
        section_id="returns",
        section_path=["Returns"],
        title="Returns",
        text=text,
        locator=Locator(line_range=(1, 1)),
        doc_type="policy",
        format="txt",
        source_path="returns.txt",
        effective_from=date(2026, 1, 1),
        token_count=10,
        translation_group="returns",
        checksum=sha256_text(text),
    )
    return RetrievalContext(
        kb_version="kb",
        index_version="index",
        text=text,
        token_count=10,
        budget=1024,
        truncated=False,
        abstain=False,
        evidence=[
            ScoredChunk(
                chunk_id=chunk.chunk_id,
                kb_version="kb",
                index_version="index",
                chunk=chunk,
                score=1,
                matched_unit_ids=["unit"],
            )
        ],
    )


def candidate(**updates: Any) -> str:
    return (
        AssistantResponse(
            language="en",
            intent="return_policy",
            action="answer_kb",
            answer="Returns are accepted within 14 days.",
            citations=["returns:aaaaaaaa:0000"],
        )
        .model_copy(update=updates)
        .model_dump_json()
    )


@pytest.mark.parametrize("language", ["ru", "kk", "en"])
def test_diagnostics_never_qualify_or_execute(packed: RetrievalContext, language: Any) -> None:
    result = inspect_completion(
        candidate(language=language),
        InspectionRequest(query="Returns?", language=language),
        packed,
        lambda _: language,
    )
    assert result["schema"] == "VALID"
    assert result["deterministic_problems"] == []
    assert result["semantic_quality"] == "NOT_EVALUATED"
    assert result["factual_delivery_allowed"] is False
    assert result["service_invoked"] is False
    assert result["tools_executed"] is False


@pytest.mark.parametrize("raw", ["not JSON", '{"answer":"incomplete"}', "<script>bad()</script>"])
def test_invalid_raw_is_not_repaired(packed: RetrievalContext, raw: str) -> None:
    result = inspect_completion(
        raw, InspectionRequest(query="Returns?", language="en"), packed, lambda _: "en"
    )
    assert result["candidate"] is None
    assert "invalid_schema" in result["deterministic_problems"]


def test_consistency_error_remains_json_serializable(packed: RetrievalContext) -> None:
    result = inspect_completion(
        candidate(needs_human=True),
        InspectionRequest(query="Returns?", language="en"),
        packed,
        lambda _: "en",
    )
    assert "invalid_schema" in result["deterministic_problems"]
    assert json.loads(json.dumps(result))["candidate"] is None


def test_citation_number_language_and_tool_failures(packed: RetrievalContext) -> None:
    result = inspect_completion(
        candidate(
            answer="Return within 999 days",
            citations=["missing:aaaaaaaa:0000"],
            action=Action.CALL_TOOL,
            facts_from_tools=["payment=999"],
        ),
        InspectionRequest(query="Refund?", language="ru"),
        packed,
        lambda _: "en",
    )
    errors = result["deterministic_problems"]
    assert "citation_not_in_context:1" in errors
    assert "unsupported_number:999" in errors
    assert "language_mismatch" in errors
    assert "no_authenticated_tool_results" in errors


def test_empty_and_changed_evidence_fail_closed(packed: RetrievalContext) -> None:
    request = InspectionRequest(query="Returns?", language="en")
    empty = packed.model_copy(update={"evidence": [], "abstain": True})
    assert (
        "empty_evidence"
        in inspect_completion(candidate(), request, empty, lambda _: "en")["deterministic_problems"]
    )
    row = packed.evidence[0]
    corrupt = row.model_copy(update={"chunk": row.chunk.model_copy(update={"text": "changed"})})
    stale = packed.model_copy(update={"evidence": [corrupt]})
    assert (
        "invalid_or_expired_provenance"
        in inspect_completion(candidate(), request, stale, lambda _: "en")["deterministic_problems"]
    )


class ProcessDouble:
    def __init__(self) -> None:
        self.alive = True
        self.terminated = False
        self.exitcode = None

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False
        self.terminated = True

    def join(self, timeout: float) -> None:
        pass

    def close(self) -> None:
        pass


class ConnectionDouble:
    def __init__(self, *, timeout: bool) -> None:
        self.timeout = timeout

    def send(self, request: object) -> None:
        pass

    def poll(self, timeout: float) -> bool:
        return not self.timeout

    def recv(self) -> dict[str, Any]:
        return {"status": "FAILED", "error_type": "RuntimeError", "error": "retained failure"}

    def close(self) -> None:
        pass


@pytest.mark.parametrize("timeout", [False, True])
def test_failure_receipt_and_worker_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: bool
) -> None:
    monkeypatch.setattr("supportpilot.service.manual_testing.sha256_file", lambda _: "a" * 64)
    workbench = ManualWorkbench(tmp_path, timeout_s=0.1)
    process: Any = ProcessDouble()
    connection: Any = ConnectionDouble(timeout=timeout)
    workbench.process, workbench.connection = process, connection
    result = workbench.inspect(InspectionRequest(query="Returns?", language="en"))
    assert result["status"] == ("TIMED_OUT" if timeout else "FAILED")
    assert result["model_calls"] == "UNKNOWN"
    assert read_json(Path(result["record_path"])) == result
    assert process.terminated
    assert workbench.process is None
    assert workbench.connection is None
    assert result["service_invoked"] is False
    workbench.close()


def test_parallel_request_rejected_without_second_worker(tmp_path: Path) -> None:
    workbench = ManualWorkbench(tmp_path)
    with workbench.lock, pytest.raises(RuntimeError, match="in progress"):
        workbench.inspect(InspectionRequest(query="Test", language="en"))
    workbench.close()
