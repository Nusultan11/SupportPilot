"""Template placeholders: inventory, synthetic slot filling and response classification.

Bitext writes entities as `{{Order Number}}`. The canonical contract forbids unresolved
placeholders in customer text, so each one is replaced by a synthetic value from
`configs/placeholders.yaml` (deterministic per record). See ADR-020 for the policy.
"""

from __future__ import annotations

import random
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.hashing import sha256_text

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SlotSpec(_Model):
    slot: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
    kind: Literal["digits", "template", "integer", "fixed", "choice"]
    length: int | None = Field(default=None, ge=1, le=18)
    template: str | None = None
    min: int | None = None
    max: int | None = None
    step: int = Field(default=1, ge=1)
    text: str | None = None
    value: str | None = None
    values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete(self) -> SlotSpec:
        need = {
            "digits": self.length is not None,
            "template": bool(self.template) and "{digits6}" in (self.template or ""),
            "integer": self.min is not None and self.max is not None and self.min <= self.max,
            "fixed": bool(self.text) and bool(self.value),
            "choice": len(self.values) > 0,
        }
        if not need[self.kind]:
            raise ValueError(f"slot {self.slot}: incomplete spec for kind={self.kind}")
        return self


class PlaceholderClass(StrEnum):
    SLOT = "slot"  # filled with a synthetic value (text placeholders)
    KB_FACT = "kb_fact"  # store fact; must come from the KB/tools in Stage 4
    TEMPLATE_INSTRUCTION = "template_instruction"  # generation artefact
    UI_LABEL = "ui_label"  # name of a site section or button


class PlaceholderCatalog(_Model):
    schema_version: Literal["placeholders/1"] = "placeholders/1"
    slots: dict[str, SlotSpec]
    instruction_patterns: list[str] = Field(min_length=1)
    kb_fact_patterns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _patterns_compile(self) -> PlaceholderCatalog:
        for p in (*self.instruction_patterns, *self.kb_fact_patterns):
            try:
                re.compile(p)
            except re.error as exc:
                raise ValueError(f"invalid pattern {p!r}: {exc}") from exc
        return self

    def classify_response(self, name: str) -> PlaceholderClass:
        """Class of a response placeholder that is *not* one of this row's text slots."""
        if any(re.search(p, name, re.IGNORECASE) for p in self.instruction_patterns):
            return PlaceholderClass.TEMPLATE_INSTRUCTION
        if any(re.search(p, name, re.IGNORECASE) for p in self.kb_fact_patterns):
            return PlaceholderClass.KB_FACT
        return PlaceholderClass.UI_LABEL


def load_catalog(path: Path) -> PlaceholderCatalog:
    return PlaceholderCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def placeholders(text: str) -> list[str]:
    return [m.group(1) for m in PLACEHOLDER_RE.finditer(text)]


def malformed(text: str) -> bool:
    """`{{` without `}}` (or the reverse) after removing well-formed placeholders."""
    rest = PLACEHOLDER_RE.sub("", text)
    return "{{" in rest or "}}" in rest


def key_form(text: str, catalog: PlaceholderCatalog) -> str:
    """Placeholders replaced by `<slot>` markers, so dedup ignores the synthetic values."""

    def repl(m: re.Match[str]) -> str:
        spec = catalog.slots.get(m.group(1))
        return f" <{spec.slot if spec else 'unknown'}> "

    return PLACEHOLDER_RE.sub(repl, text)


def _rng(seed_key: str, name: str) -> random.Random:
    return random.Random(int(sha256_text(f"{seed_key}|{name}")[:16], 16))  # noqa: S311 - synthetic values


def synth_value(spec: SlotSpec, rng: random.Random) -> tuple[str, str]:
    """(text shown in the message, slot value). Both synthetic."""
    match spec.kind:
        case "digits":
            n = spec.length or 6
            v = str(rng.randrange(10 ** (n - 1), 10**n))
            return v, v
        case "template":
            v = (spec.template or "").replace("{digits6}", str(rng.randrange(100_000, 1_000_000)))
            return v, v
        case "integer":
            lo, hi = spec.min or 0, spec.max or 0
            v = str(lo + rng.randrange(0, (hi - lo) // spec.step + 1) * spec.step)
            return v, v
        case "fixed":
            return spec.text or "", spec.value or ""
        case "choice":
            v = spec.values[rng.randrange(len(spec.values))]
            return v, v
    raise AssertionError(spec.kind)  # pragma: no cover


class Filled(BaseModel):
    text: str
    slots: dict[str, str]  # slot -> value (e.g. order_id -> "482913")
    shown: dict[str, str]  # placeholder name -> text inserted (e.g. "Currency Symbol" -> "₸")
    unknown: list[str]


def fill_text(template: str, catalog: PlaceholderCatalog, seed_key: str) -> Filled:
    """Replace known placeholders by synthetic values (same name -> same value in a row)."""
    shown: dict[str, str] = {}
    slots: dict[str, str] = {}
    unknown: list[str] = []

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        spec = catalog.slots.get(name)
        if spec is None:
            unknown.append(name)
            return m.group(0)
        if name not in shown:
            text, value = synth_value(spec, _rng(seed_key, name))
            shown[name], slots[spec.slot] = text, value
        return shown[name]

    out = PLACEHOLDER_RE.sub(repl, template)
    return Filled(text=out, slots=slots, shown=shown, unknown=unknown)


class Resolved(BaseModel):
    text: str
    kept: dict[str, list[str]]  # class -> placeholder names left in the text


def resolve_response(response: str, shown: dict[str, str], catalog: PlaceholderCatalog) -> Resolved:
    """Fill the row's own text slots; keep and classify every other placeholder."""
    kept: dict[str, list[str]] = {}

    def repl(m: re.Match[str]) -> str:
        name = m.group(1)
        if name in shown:
            return shown[name]
        kept.setdefault(catalog.classify_response(name).value, []).append(name)
        return m.group(0)

    out = PLACEHOLDER_RE.sub(repl, response)
    return Resolved(text=out, kept={k: sorted(set(v)) for k, v in sorted(kept.items())})
