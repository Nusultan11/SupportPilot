from __future__ import annotations

import json
from pathlib import Path

import pytest

from supportpilot.llm.decoding import prefix_filter, tokenizer_data


class CharacterTokenizer:
    eos_token_id = 128
    all_special_ids = (128,)

    def __len__(self) -> int:
        return 129

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids if i < 128)


class Sequence:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def tolist(self) -> list[int]:
        return self.ids


def test_public_core_adapter_enforces_schema_without_transformers_import_hack() -> None:
    tokenizer = CharacterTokenizer()
    schema = {
        "type": "object",
        "properties": {"language": {"enum": ["en", "ru", "kk"]}},
        "required": ["language"],
        "additionalProperties": False,
    }
    allowed = prefix_filter(tokenizer_data(tokenizer), schema)
    ids = tokenizer.encode("untrusted prompt")
    text = json.dumps({"language": "kk"})
    for character in text:
        assert ord(character) in allowed(0, Sequence(ids))
        ids.append(ord(character))
    assert tokenizer.eos_token_id in allowed(0, Sequence(ids))


def test_enforcer_rejects_an_unknown_language_value() -> None:
    tokenizer = CharacterTokenizer()
    schema = {
        "type": "object",
        "properties": {"language": {"enum": ["en", "ru", "kk"]}},
        "required": ["language"],
    }
    allowed = prefix_filter(tokenizer_data(tokenizer), schema)
    ids = tokenizer.encode("prompt")
    for character in '{"language":"':
        assert ord(character) in allowed(0, Sequence(ids))
        ids.append(ord(character))
    assert ord("x") not in allowed(0, Sequence(ids))


def test_root_verdict_enum_cannot_append_a_period() -> None:
    tokenizer = CharacterTokenizer()
    allowed = prefix_filter(
        tokenizer_data(tokenizer),
        {"type": "string", "enum": ["supported", "unsupported", "unknown"]},
    )
    ids = tokenizer.encode("untrusted prompt")
    for char in '"unsupported"':
        assert ord(char) in allowed(0, Sequence(ids))
        ids.append(ord(char))
    assert tokenizer.eos_token_id in allowed(0, Sequence(ids))
    assert ord(".") not in allowed(0, Sequence(ids))


def test_backend_schema_override_cannot_silently_disable_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from supportpilot.llm import backend
    from supportpilot.llm.models import InferenceConfig, load_models

    models = load_models(Path(__file__).resolve().parents[2] / "configs/llm_models.yaml")

    def forbidden_import(name: str) -> None:
        raise AssertionError("No model or Torch import is allowed before validation")

    config = InferenceConfig(model=next(iter(models.values())), device="cpu", constrained=False)
    monkeypatch.setattr("supportpilot.llm.backend.importlib.import_module", forbidden_import)
    with pytest.raises(ValueError, match="requires constrained"):
        backend.TransformersBackend(config, output_schema={"type": "string"})
