"""Read-only local model inspection; never a customer-delivery or qualification path."""

from __future__ import annotations

import atexit
import json
import multiprocessing as mp
import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, date, datetime
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from supportpilot.common.config import Settings, load_settings
from supportpilot.common.grounding import Evidence, check_response
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import Action, AssistantResponse, LanguageCode
from supportpilot.llm.generation import Backend, build_messages, safe_fallback
from supportpilot.llm.models import InferenceConfig
from supportpilot.retrieval.models import RetrievalContext


class InspectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    query: str = Field(min_length=1, max_length=2000)
    language: LanguageCode


class _Connection(Protocol):
    def send(self, obj: Any) -> None: ...
    def recv(self) -> Any: ...
    def poll(self, timeout: float) -> bool: ...
    def close(self) -> None: ...


def inspect_completion(
    raw: str,
    request: InspectionRequest,
    context: RetrievalContext,
    identify_language: Callable[[str], str],
) -> dict[str, Any]:
    """Reuse deterministic checks without treating them as semantic verification."""
    problems = []
    if context.abstain or not context.evidence:
        problems.append("empty_evidence")
    today = date.today()
    for item in context.evidence:
        if (
            item.chunk_id != item.chunk.chunk_id
            or item.kb_version != context.kb_version
            or item.index_version != context.index_version
            or item.chunk.checksum != sha256_text(item.chunk.text)
            or item.chunk.effective_from > today
            or (item.chunk.effective_to is not None and item.chunk.effective_to < today)
        ):
            problems.append("invalid_or_expired_provenance")
    response = None
    detected = None
    schema_error = None
    try:
        response = AssistantResponse.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        problems.append("invalid_schema")
        schema_error = exc.errors(include_url=False, include_input=False, include_context=False)
    if response is not None:
        evidence = Evidence(
            chunks={
                e.chunk_id: " ".join(
                    [
                        e.chunk.text,
                        e.chunk.effective_from.isoformat(),
                        e.chunk.effective_to.isoformat() if e.chunk.effective_to else "",
                    ]
                )
                for e in context.evidence
            }
        )
        problems.extend(check_response(response, evidence))
        detected = identify_language(response.answer)
        if response.language != request.language or detected != request.language:
            problems.append("language_mismatch")
        if response.action is Action.CALL_TOOL or response.facts_from_tools:
            problems.append("no_authenticated_tool_results")
    return {
        "schema": "VALID" if response is not None else "INVALID",
        "schema_errors": schema_error,
        "candidate": response.model_dump(mode="json") if response else None,
        "detected_reply_language": detected,
        "deterministic_problems": list(dict.fromkeys(problems)),
        "semantic_quality": "NOT_EVALUATED",
        "semantic_support": "unknown",
        "factual_delivery_allowed": False,
        "service_invoked": False,
        "tools_executed": False,
        "fallback_preview": safe_fallback(request.language).model_dump(mode="json"),
    }


def selected_config(settings: Settings) -> tuple[str, InferenceConfig]:
    """Verify the unchanged selection and adapter before loading local weights."""
    seal = read_json(settings.root / "artifacts/reports/llm/selection.json")
    identity = str(seal.pop("identity"))
    if sha256_text(json.dumps(seal, sort_keys=True)) != identity:
        raise ValueError("selection identity mismatch")
    config = InferenceConfig.model_validate(seal["configurations"]["adapter"])
    if (
        config.adapter is None
        or sha256_file(config.adapter / "adapter_manifest.json")
        != seal["adapter_manifests"]["adapter"]
    ):
        raise ValueError("adapter differs from sealed selection")
    # Operational resource profile only; do not rewrite sealed configuration or model files.
    return identity, config.model_copy(update={"quantization": "nf4", "max_seconds": 45.0})


@contextmanager
def local_components(settings: Settings) -> Iterator[tuple[Backend, Any, dict[str, Any]]]:
    from supportpilot.llm.backend import TransformersBackend
    from supportpilot.retrieval import open_retriever

    endpoint = urlparse(settings.qdrant.url)
    if settings.qdrant.mode != "server" or endpoint.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("manual testing requires loopback Qdrant")
    identity, config = selected_config(settings)
    retrieval = settings.retrieval.model_copy(update={"device": "cpu"})
    with open_retriever(settings.model_copy(update={"retrieval": retrieval})) as retriever:
        backend = TransformersBackend(config)
        try:
            yield (
                backend,
                retriever,
                {
                    "selection_identity": identity,
                    "inference_config": config.model_dump(mode="json"),
                    "retrieval_config": retrieval.model_dump(mode="json"),
                    "retrieval_device": "cpu",
                },
            )
        finally:
            backend.close()


def _worker(connection: _Connection, root: str) -> None:
    """Only this disposable child owns models; a hung child can be stopped safely."""
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    from supportpilot.data.langid import LanguageIdentifier

    try:
        settings = load_settings(Path(root) / "configs/base.yaml", root=Path(root))
        with local_components(settings) as (backend, retriever, identity):
            lid = LanguageIdentifier()
            while True:
                payload = connection.recv()
                if payload is None:
                    break
                request = InspectionRequest.model_validate(payload)
                started = time.perf_counter()
                record: dict[str, Any] = {"identity": identity, "raw": None, "model_calls": 0}
                try:
                    context = retriever.context(request.query, request.language)
                    messages = build_messages(
                        request.query, request.language, context, prompt_mode="schema"
                    )
                    record.update(
                        context=context.model_dump(mode="json"),
                        messages=messages,
                        query_language=asdict(lid.detect(request.query)),
                    )
                    record["model_calls"] = 1
                    raw = backend.generate(messages)
                    record.update(raw=raw, resources=dict(getattr(backend, "last_observation", {})))
                    record["checks"] = inspect_completion(
                        raw, request, context, lambda text: lid.detect(text).lang
                    )
                    record["status"] = "COMPLETE"
                except Exception as exc:
                    record.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
                record["seconds"] = time.perf_counter() - started
                connection.send(record)
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        connection.send({"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc)})
    finally:
        connection.close()


class ManualWorkbench:
    """One serialized local worker; retain attempts and automatically recover after timeout."""

    def __init__(self, root: Path, *, timeout_s: float = 180) -> None:
        self.root = root.resolve()
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.process: BaseProcess | None = None
        self.connection: _Connection | None = None
        atexit.register(self.close)

    def close(self) -> None:
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=5)
            self.process.close()
            self.process = None
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def inspect(self, request: InspectionRequest) -> dict[str, Any]:
        if not request.query.strip():
            raise ValueError("query must contain text")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("another local inference is in progress")
        try:
            return self._inspect(request)
        finally:
            self.lock.release()

    def _inspect(self, request: InspectionRequest) -> dict[str, Any]:
        attempt = uuid4().hex
        path = self.root / "artifacts/reports/manual_testing/attempts" / attempt / "record.json"
        record: dict[str, Any] = {
            "schema_version": "manual_llm_inspection/1",
            "attempt": attempt,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "RUNNING",
            "request": request.model_dump(),
            "semantic_quality": "NOT_EVALUATED",
            "independent_qualification": False,
            "service_invoked": False,
            "tools_executed": False,
            "record_path": str(path),
            "model_calls": "UNKNOWN",
            "source_hashes": {
                name: sha256_file(self.root / name)
                for name in (
                    "src/supportpilot/service/manual_testing.py",
                    "apps/manual_testing.py",
                    "src/supportpilot/llm/generation.py",
                    "src/supportpilot/llm/backend.py",
                    "src/supportpilot/service/acceptance.py",
                    "configs/base.yaml",
                    "uv.lock",
                )
            },
        }
        write_json(path, record)
        started = time.perf_counter()
        try:
            if self.process is None or not self.process.is_alive():
                self.close()
                spawn = mp.get_context("spawn")
                self.connection, child = spawn.Pipe()
                self.process = spawn.Process(
                    target=_worker, args=(child, str(self.root)), daemon=True
                )
                self.process.start()
                child.close()
            assert self.connection is not None
            self.connection.send(request.model_dump())
            if not self.connection.poll(self.timeout_s):
                record.update(status="TIMED_OUT", error="Local worker deadline exceeded")
                self.close()
            else:
                record.update(self.connection.recv())
                if record["status"] == "FAILED":
                    self.close()
        except Exception as exc:
            record.update(
                status="FAILED", error_type=type(exc).__name__, error=str(exc) or type(exc).__name__
            )
            if self.process is not None:
                self.process.join(timeout=1)
                record["worker_exitcode"] = self.process.exitcode
            self.close()
        finally:
            record["wall_seconds"] = time.perf_counter() - started
            write_json(path, record)
        return record
