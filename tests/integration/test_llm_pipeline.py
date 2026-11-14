from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from supportpilot.common.schemas import Action, AssistantResponse
from supportpilot.llm.experiments import run_inference
from supportpilot.llm.models import InferenceConfig, ModelSpec


@pytest.fixture
def inference_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, Any]], InferenceConfig, dict[str, int]]:
    target = AssistantResponse(
        language="en",
        intent="return_policy",
        action=Action.ANSWER_KB,
        answer="According to the policy, the window is 14 days.",
        citations=["returns_en:12345678:0000"],
    )
    unsupported = target.model_copy(update={"answer": "The return window is 30 days."})
    outputs = iter(["invalid JSON", unsupported.model_dump_json()])
    calls = {"generate": 0}

    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            return list(text.encode())

    class Backend:
        def __init__(self, config: InferenceConfig) -> None:
            self.tokenizer = Tokenizer()
            self.load_seconds = 0.0
            self.last_observation = {"seconds": 0.01, "output_tokens": 5}

        def generate(self, messages: list[dict[str, str]]) -> str:
            assert all(message["content"] != target.model_dump_json() for message in messages)
            calls["generate"] += 1
            return next(outputs)

        def close(self) -> None:
            pass

    monkeypatch.setattr("supportpilot.llm.experiments.TransformersBackend", Backend)
    monkeypatch.setattr(
        "supportpilot.llm.experiments.LanguageIdentifier",
        lambda: SimpleNamespace(detect=lambda text: SimpleNamespace(lang="en")),
    )
    rows = [
        {
            "example_id": key,
            "scenario_id": "scenario_" + key,
            "family_id": "family_" + key,
            "lang": "en",
            "task": "grounded_answer",
            "user_text": "What is the return window?",
            "target": target.model_dump(mode="json"),
            "messages": [
                {"role": "user", "content": "What is the return window?"},
                {
                    "role": "tool",
                    "name": "kb_search",
                    "content": json.dumps(
                        {
                            "results": [
                                {
                                    "chunk_id": "returns_en:12345678:0000",
                                    "title": "Returns",
                                    "text": "The return window is 14 days.",
                                    "effective_from": "2026-11-13",
                                    "effective_to": None,
                                },
                            ]
                        }
                    ),
                },
                {"role": "assistant", "content": target.model_dump_json()},
            ],
        }
        for key in ("a", "b")
    ]
    config = InferenceConfig(
        model=ModelSpec(
            repo_id="example/model",
            revision="a" * 40,
            licence="apache-2.0",
        )
    )
    return rows, config, calls


def test_runner_preserves_raw_failures_and_reuses_completed_prediction_prefix(
    tmp_path: Path,
    inference_fixture: tuple[list[dict[str, Any]], InferenceConfig, dict[str, int]],
) -> None:
    rows, config, calls = inference_fixture
    result = run_inference(rows, config, tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    original = predictions.read_bytes()
    observations = [json.loads(line) for line in original.splitlines()]
    assert result["count"] == 2
    assert result["summary"]["en"]["metrics"]["schema_validity"]["value"] == 0.5
    assert observations[0]["raw"] == "invalid JSON"
    assert observations[1]["schema_validity"] == 1.0
    assert observations[1]["diagnostic_grounding"] == 0.0
    assert observations[1]["citation_membership"] == 1.0
    assert any("number" in problem for problem in observations[1]["problems"])
    assert all(row["fallback_required"] for row in observations)
    assert all(row["semantic_support"] == "NOT_EVALUATED" for row in observations)
    run_inference(rows, config, tmp_path)
    assert calls["generate"] == 2
    assert predictions.read_bytes() == original


def test_runner_rejects_changed_inputs_and_corrupted_prediction_prefix(
    tmp_path: Path,
    inference_fixture: tuple[list[dict[str, Any]], InferenceConfig, dict[str, int]],
) -> None:
    rows, config, calls = inference_fixture
    run_inference(rows, config, tmp_path)
    changed = [{**rows[0], "user_text": "different input"}, rows[1]]
    with pytest.raises(ValueError, match="input/config mismatch"):
        run_inference(changed, config, tmp_path)
    predictions = tmp_path / "predictions.jsonl"
    observations = [json.loads(line) for line in predictions.read_text("utf-8").splitlines()]
    observations[0]["example_id"] = "different_example"
    predictions.write_text(
        "\n".join(json.dumps(row) for row in observations) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not an input prefix"):
        run_inference(rows, config, tmp_path)
    assert calls["generate"] == 2
