"""The rendered-source register and the extraction ground truth.

`register.json` (kb_register/1) is the document catalogue a store CMS would export: one entry
per file with doc_id, language, type, title, effective date, the sha256 of the bytes and the
section ids with their headings. Parsers do not read it; `structure` uses it only to attach
the stable section ids to the headings a parser found (and the title / date of a CSV file,
which has no place for them).

`expected.json` (kb_expected/1) is what rendering put into each file: fact placements with
their exact surface strings and every content string. Only the gates read it.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

from supportpilot.common.schemas import (
    DocFormat,
    DocType,
    LanguageCode,
    NonEmptyStr,
    RelPath,
    Sha256,
    Slug,
)

REGISTER = "register.json"
EXPECTED = "expected.json"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RegisterSection(_Model):
    id: Slug
    heading: NonEmptyStr  # without the printed number
    level: int = Field(ge=1, le=2)


class RegisterEntry(_Model):
    path: RelPath
    doc_id: Slug
    lang: LanguageCode
    doc_type: DocType
    format: DocFormat
    title: NonEmptyStr
    effective_from: date
    sha256: Sha256
    bytes: NonNegativeInt
    key_column: NonNegativeInt | None = None  # record documents
    sections: list[RegisterSection] = Field(default_factory=list)
    records: list[str] = Field(default_factory=list)


class Register(_Model):
    schema_version: Literal["kb_register/1"] = "kb_register/1"
    kb: Slug
    render_params: dict[str, Any]
    documents: list[RegisterEntry] = Field(min_length=1)


class ExpectedPlacement(_Model):
    key: NonEmptyStr
    value: Any
    surface: NonEmptyStr
    section_id: str | None
    where: str


class ExpectedDoc(_Model):
    title: NonEmptyStr
    effective_from: date
    placements: list[ExpectedPlacement]
    content: list[str]


class Expected(_Model):
    schema_version: Literal["kb_expected/1"] = "kb_expected/1"
    facts: dict[str, Any]
    documents: dict[str, ExpectedDoc]  # by register path
