from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from supportpilot.common.schemas import Action, AssistantResponse
from supportpilot.llm.adapters import validate_adapter, write_adapter_manifest
from supportpilot.llm.models import InferenceConfig, ModelSpec, TrainingConfig
from supportpilot.llm.selection import require_seal, seal
from supportpilot.llm.train import prepare_training_rows


class Tokenizer:
    eos_token = "EOS"

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        return json.dumps(messages)

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        return list(text.encode())


def config(**kwargs: Any) -> TrainingConfig:
    model = ModelSpec(repo_id="example/model", revision="a" * 40, licence="apache-2.0")
    return TrainingConfig(inference=InferenceConfig(model=model), **kwargs)


def row(split: str = "train") -> dict[str, Any]:
    response = AssistantResponse(
        language="en",
        intent="greeting",
        action=Action.CLARIFY,
        answer="How can I help?",
        clarification_question="How can I help?",
    )
    return {
        "example_id": "a",
        "split": split,
        "lang": "en",
        "task": "clarify",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": response.model_dump_json()},
        ],
    }


def test_training_never_silently_truncates_a_target() -> None:
    with pytest.raises(ValueError, match="no complete training targets"):
        prepare_training_rows([row()], Tokenizer(), config(max_length=128))
    prepared, excluded = prepare_training_rows([row()], Tokenizer(), config(max_length=8192))
    assert len(prepared) == 1
    assert not excluded
    assert prepared[0]["completion"].endswith("EOS")


def test_training_cannot_read_a_test_row_or_current_store_facts() -> None:
    with pytest.raises(ValueError, match="train-only"):
        prepare_training_rows([row("test")], Tokenizer(), config())
    current = row()
    current["task"] = "grounded_answer"
    with pytest.raises(ValueError, match="training-KB"):
        prepare_training_rows([current], Tokenizer(), config())


def test_adapter_bytes_and_base_revision_are_checked(tmp_path: Path) -> None:
    model = config().inference.model
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"trusted-test-weights")
    write_adapter_manifest(tmp_path, model, "training-identity")
    validate_adapter(tmp_path, model)
    with pytest.raises(ValueError, match="revision"):
        validate_adapter(tmp_path, model.model_copy(update={"revision": "b" * 40}))
    weights.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        validate_adapter(tmp_path, model)


def test_selection_binds_adapter_bytes_and_manifest(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    weights = adapter / "adapter.safetensors"
    weights.write_bytes(b"selected")
    model = config().inference.model
    write_adapter_manifest(adapter, model, "selected-training")
    selection = tmp_path / "selection.json"
    seal(
        selection,
        {"selected": InferenceConfig(model=model, adapter=adapter)},
        prompt_modes={"selected": "schema"},
        decision={},
        validation_reports=[],
    )
    require_seal(selection)
    weights.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="integrity"):
        require_seal(selection)
    write_adapter_manifest(adapter, model, "different-training")
    with pytest.raises(ValueError, match="changed after selection"):
        require_seal(selection)
