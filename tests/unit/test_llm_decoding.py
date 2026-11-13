from __future__ import annotations

import json

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
