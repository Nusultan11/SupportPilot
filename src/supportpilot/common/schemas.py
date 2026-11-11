"""Pydantic data contracts (DATA_CONTRACTS §2-4, §9). Breaking changes bump `schema_version`."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    model_validator,
)

from supportpilot.common.hashing import SHA256_RE
from supportpilot.common.ids import (
    DATASET_VERSION_RE,
    EXAMPLE_ID_RE,
    RECORD_ID_RE,
    validate_slug,
)


def _pattern(regex_name: str, regex: re.Pattern[str]) -> AfterValidator:
    def check(value: str) -> str:
        if not regex.fullmatch(value):
            raise ValueError(f"invalid {regex_name}: {value!r}")
        return value

    return AfterValidator(check)


def _rel_posix(value: str) -> str:
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or value in {"", "."}:
        raise ValueError(f"path must be relative POSIX without '..': {value!r}")
    return value


Slug = Annotated[str, AfterValidator(validate_slug)]
Sha256 = Annotated[str, _pattern("sha256", SHA256_RE)]
ExampleId = Annotated[str, _pattern("example_id", EXAMPLE_ID_RE)]
RecordId = Annotated[str, _pattern("record_id", RECORD_ID_RE)]
DatasetVersion = Annotated[str, _pattern("dataset_version", DATASET_VERSION_RE)]
RelPath = Annotated[str, AfterValidator(_rel_posix)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

LanguageCode = Literal["en", "ru", "kk"]
LANGUAGES: tuple[LanguageCode, ...] = ("en", "ru", "kk")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- §2 source registry


class AllowedUse(StrEnum):
    TRAIN = "train"
    EVAL_ONLY = "eval_only"
    REFERENCE_ONLY = "reference_only"
    BLOCKED = "blocked"


class SourceSize(_Contract):
    rows: NonNegativeInt | None = None
    files: NonNegativeInt | None = None
    bytes: NonNegativeInt | None = None


class SourceEntry(_Contract):
    source_id: Slug
    url: NonEmptyStr
    revision: NonEmptyStr
    licence: NonEmptyStr
    allowed_use: AllowedUse
    languages: list[LanguageCode] = Field(min_length=1)
    size: SourceSize = SourceSize()
    retrieved_at: AwareDatetime | None = None
    purpose: NonEmptyStr
    notes: str = ""

    @model_validator(mode="after")
    def _unknown_licence_blocks_use(self) -> Self:
        if self.licence.strip().lower() == "unknown" and self.allowed_use is not AllowedUse.BLOCKED:
            raise ValueError("licence 'unknown' requires allowed_use='blocked'")
        return self


class SourceRegistry(_Contract):
    schema_version: Literal["sources/1"] = "sources/1"
    sources: list[SourceEntry]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        ids = [s.source_id for s in self.sources]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate source_id: {dupes}")
        return self

    def get(self, source_id: str) -> SourceEntry:
        for s in self.sources:
            if s.source_id == source_id:
                return s
        raise KeyError(source_id)


# ---------------------------------------------------------------- §3 raw storage / manifests


class FileEntry(_Contract):
    path: RelPath
    bytes: NonNegativeInt
    sha256: Sha256


class RawManifest(_Contract):
    schema_version: Literal["raw_manifest/1"] = "raw_manifest/1"
    source_id: Slug
    revision: NonEmptyStr
    url: NonEmptyStr
    retrieved_at: AwareDatetime
    files: list[FileEntry] = Field(min_length=1)


class DatasetManifest(_Contract):
    """§9: one per processed dataset or index."""

    schema_version: Literal["dataset_manifest/1"] = "dataset_manifest/1"
    dataset_version: DatasetVersion
    data_schema_version: NonEmptyStr
    code_commit: NonEmptyStr
    inputs: list[NonEmptyStr] = Field(min_length=1)
    params: dict[str, object] = Field(default_factory=dict)
    row_counts: dict[str, NonNegativeInt]
    files: list[FileEntry] = Field(min_length=1)
    created_at: AwareDatetime
    quality_report: str | None = None


# ---------------------------------------------------------------- §4 canonical example


class Action(StrEnum):
    ANSWER_KB = "answer_kb"
    CALL_TOOL = "call_tool"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    REFUSE_OOS = "refuse_oos"


class Origin(StrEnum):
    ORIGINAL = "original"
    MACHINE_TRANSLATED = "machine_translated"
    LLM_GENERATED = "llm_generated"
    POST_EDITED = "post_edited"
    HUMAN_WRITTEN = "human_written"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


# Intent taxonomy draft v0 (DATA_CONTRACTS §7). Frozen as v1 in Stage 4.
INTENTS: frozenset[str] = frozenset(
    {
        "track_order", "cancel_order", "change_order", "order_history", "place_order",
        "delivery_time", "delivery_options_cost", "delivery_problem",
        "return_policy", "return_product", "exchange_product", "request_refund", "refund_status",
        "payment_methods", "payment_problem", "invoice",
        "product_info", "availability",
        "store_info", "policy_faq",
        "account_help",
        "human_agent", "complaint", "feedback",
        "unclear", "out_of_scope", "greeting",
    }
)  # fmt: skip


def _known_intent(value: str) -> str:
    if value not in INTENTS:
        raise ValueError(f"unknown intent {value!r}")
    return value


Intent = Annotated[str, AfterValidator(_known_intent)]


class Quality(_Contract):
    lid_conf: float = Field(ge=0.0, le=1.0)
    mt_qe: float | None = None
    intent_agree: bool | None = None
    dup_cluster: str | None = None
    flags: list[str] = Field(default_factory=list)


class CanonicalExample(_Contract):
    schema_version: Literal["canonical/1"] = "canonical/1"
    example_id: ExampleId
    record_id: RecordId
    source_id: Slug
    scenario_id: NonEmptyStr
    family_id: NonEmptyStr
    lang: LanguageCode
    code_switch: bool
    text: NonEmptyStr
    reference_response: str | None = None
    intent: Intent
    source_intent: str | None = None
    source_category: str | None = None
    action: Action
    escalate: bool
    escalation_reason: str | None = None
    slots: dict[str, str] = Field(default_factory=dict)
    origin: Origin
    translation_of: ExampleId | None = None
    translation_system: str | None = None
    quality: Quality
    split: Split | None = None
    created_at: datetime | None = None

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if not self.record_id.startswith(f"{self.source_id}:"):
            raise ValueError("record_id must start with '{source_id}:'")
        if self.escalate and not (self.escalation_reason and self.escalation_reason.strip()):
            raise ValueError("escalate=True requires escalation_reason")
        if self.action is Action.ESCALATE and not self.escalate:
            raise ValueError("action='escalate' requires escalate=True")
        if self.origin is Origin.MACHINE_TRANSLATED and not (
            self.translation_of and self.translation_system
        ):
            raise ValueError("machine_translated requires translation_of and translation_system")
        if "{{" in self.text:
            raise ValueError("unresolved {{placeholder}} in text")
        return self
