"""Transformers 5 adapter to LM Format Enforcer's public core API.

The packaged 0.11.3 Transformers integration imports a removed tokenizer symbol.
This adapter uses TokenEnforcer directly; no monkey patch or model-code execution.
Reference: https://github.com/noamgat/lm-format-enforcer (MIT).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def tokenizer_data(tokenizer: Any) -> Any:
    core: Any = importlib.import_module("lmformatenforcer")
    prefix = tokenizer.encode("0", add_special_tokens=False)[-1]
    special = set(tokenizer.all_special_ids)
    vocabulary = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        alone = str(tokenizer.decode([token_id]))
        contextual = str(tokenizer.decode([prefix, token_id]))[1:]
        vocabulary.append((token_id, contextual, len(contextual) > len(alone)))

    def decode(ids: list[int]) -> str:
        return str(tokenizer.decode(ids)).rstrip("\ufffd")

    return core.TokenEnforcerTokenizerData(
        vocabulary, decode, tokenizer.eos_token_id, False, len(tokenizer)
    )


def prefix_filter(data: Any, schema: dict[str, Any]) -> Callable[[int, Any], list[int]]:
    core: Any = importlib.import_module("lmformatenforcer")
    enforcer = core.TokenEnforcer(data, core.JsonSchemaParser(schema))

    def allowed(batch_id: int, sequence: Any) -> list[int]:
        return list(enforcer.get_allowed_tokens(sequence.tolist()).allowed_tokens)

    return allowed
