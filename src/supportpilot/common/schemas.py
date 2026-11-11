"""Pydantic data contracts (DATA_CONTRACTS §2-4, §9). Breaking changes bump `schema_version`."""

from __future__ import annotations

import re
from datetime import date, datetime
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

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

LanguageCode = Literal["en", "ru", "kk"]
LANGUAGES: tuple[LanguageCode, ...] = ("en", "ru", "kk")


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- §2 source registry


class AllowedUse(StrEnum):
    """Ordered from most to least permissive (see `USE_RANK`)."""

    TRAIN = "train"
    EVAL_ONLY = "eval_only"
    REFERENCE_ONLY = "reference_only"
    BLOCKED = "blocked"


USE_RANK: dict[AllowedUse, int] = {
    AllowedUse.TRAIN: 3,
    AllowedUse.EVAL_ONLY: 2,
    AllowedUse.REFERENCE_ONLY: 1,
    AllowedUse.BLOCKED: 0,
}

# Licence policy (ADR-018): the most permissive use each licence allows in this project.
# A licence not listed here is rejected by validation, so new licences are a conscious choice.
LICENCE_MAX_USE: dict[str, AllowedUse] = {
    "cdla-sharing-1.0": AllowedUse.TRAIN,  # share-alike for published derived data
    "cc-by-4.0": AllowedUse.TRAIN,
    "cc-by-sa-4.0": AllowedUse.TRAIN,  # share-alike
    "cc0-1.0": AllowedUse.TRAIN,
    "mit": AllowedUse.TRAIN,
    "apache-2.0": AllowedUse.TRAIN,
    "cc-by-nc-4.0": AllowedUse.EVAL_ONLY,
    "cc-by-nc-sa-4.0": AllowedUse.EVAL_ONLY,
    "unknown": AllowedUse.BLOCKED,
    "conflicting": AllowedUse.BLOCKED,  # declared licence contradicts the upstream data's licence
}


class SourceStatus(StrEnum):
    SELECTED = "selected"  # acquired and converted in Stage 2
    CANDIDATE = "candidate"  # usable, not acquired yet (reason in `decision`)
    REJECTED = "rejected"  # researched and not used (reason in `decision`)
    UNAVAILABLE = "unavailable"  # wanted, but access is blocked (e.g. gated without token)


class Access(StrEnum):
    PUBLIC = "public"
    GATED = "gated"  # needs an HF account that accepted the terms (HF_TOKEN)


class FetchSpec(_Contract):
    type: Literal["hf_dataset"] = "hf_dataset"
    repo_id: Annotated[str, Field(pattern=r"^[\w.-]+/[\w.-]+$")]
    include: list[NonEmptyStr] = Field(min_length=1, description="fnmatch patterns of repo files")


class SourceSize(_Contract):
    rows: NonNegativeInt | None = None
    files: NonNegativeInt | None = None
    bytes: NonNegativeInt | None = None


class SourceEntry(_Contract):
    source_id: Slug
    url: NonEmptyStr
    revision: NonEmptyStr
    licence: NonEmptyStr
    licence_evidence: NonEmptyStr
    allowed_use: AllowedUse
    status: SourceStatus
    access: Access = Access.PUBLIC
    languages: list[LanguageCode] = Field(min_length=1)
    size: SourceSize = SourceSize()
    reviewed_at: date
    purpose: NonEmptyStr
    decision: NonEmptyStr
    notes: str = ""
    fetch: FetchSpec | None = None

    @model_validator(mode="after")
    def _policy(self) -> Self:
        lic = self.licence.strip().lower()
        if lic not in LICENCE_MAX_USE:
            raise ValueError(f"licence {self.licence!r} is not in the licence policy (ADR-018)")
        if USE_RANK[self.allowed_use] > USE_RANK[LICENCE_MAX_USE[lic]]:
            raise ValueError(
                f"allowed_use={self.allowed_use.value} exceeds what licence {lic} permits "
                f"({LICENCE_MAX_USE[lic].value})"
            )
        if self.status is SourceStatus.SELECTED:
            if self.allowed_use is AllowedUse.BLOCKED:
                raise ValueError("a blocked source cannot be selected")
            if self.fetch is None:
                raise ValueError("a selected source needs a fetch spec")
        if (
            self.fetch is not None
            and self.fetch.type == "hf_dataset"
            and not _GIT_SHA_RE.fullmatch(self.revision)
        ):
            raise ValueError("HF sources must pin `revision` to a full 40-char commit sha")
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
    licence: str | None = None
    repo_id: str | None = None
    files: list[FileEntry] = Field(min_length=1)


class RecordType(StrEnum):
    SINGLE_TURN = "single_turn"  # one customer message (+ optional agent response)
    DIALOGUE = "dialogue"  # multi-turn; full turns in `extra_json["messages"]`
    READING_COMPREHENSION = "reading_comprehension"  # passage + question + options
    POLICY_DECISION = "policy_decision"  # message + policies/facts -> structured decision


class RawRecord(_Contract):
    """Stage 2 output: one source row in a uniform shape, values exactly as in the source."""

    schema_version: Literal["raw_record/1"] = "raw_record/1"
    record_id: RecordId
    source_id: Slug
    revision: NonEmptyStr
    source_file: RelPath
    source_row: NonNegativeInt
    source_split: str | None
    record_type: RecordType
    lang: LanguageCode
    text: NonEmptyStr
    response: str | None
    source_intent: str | None
    source_category: str | None
    source_tags: str | None
    extra_json: str
    text_sha256: Sha256

    @model_validator(mode="after")
    def _lineage(self) -> Self:
        if not self.record_id.startswith(f"{self.source_id}:"):
            raise ValueError("record_id must start with '{source_id}:'")
        return self


class RecordsManifest(_Contract):
    """Manifest of `data/interim/records/{source_id}/{revision}/`."""

    schema_version: Literal["records_manifest/1"] = "records_manifest/1"
    source_id: Slug
    revision: NonEmptyStr
    allowed_use: AllowedUse
    converter: NonEmptyStr
    converter_version: NonNegativeInt
    raw_manifest_sha256: Sha256
    code_commit: str | None
    created_at: AwareDatetime
    row_count: NonNegativeInt
    rejected_count: NonNegativeInt
    rejected_reasons: dict[str, NonNegativeInt]
    counts: dict[str, NonNegativeInt]
    content_sha256: Sha256
    files: list[FileEntry] = Field(min_length=1)


class PendingExample(_Contract):
    """§4c: a cleaned, deduplicated example that cannot be made canonical yet (reason given)."""

    schema_version: Literal["pending/1"] = "pending/1"
    example_id: ExampleId
    record_id: RecordId
    source_id: Slug
    revision: NonEmptyStr
    record_type: RecordType
    source_split: str | None = None
    scenario_id: NonEmptyStr
    family_id: NonEmptyStr
    lang: LanguageCode
    code_switch: bool
    langs: list[str]
    text: NonEmptyStr
    reference_response: str | None = None
    slots: dict[str, str] = Field(default_factory=dict)
    source_intent: str | None = None
    source_category: str | None = None
    source_tags: str | None = None
    candidate_intent: str | None = None  # v0 intent if the label maps, else None
    candidate_action: str | None = None
    context_json: str | None = None
    allowed_use: AllowedUse
    reason: NonEmptyStr
    reason_detail: str | None = None
    quality: Quality

    @model_validator(mode="after")
    def _lineage(self) -> Self:
        if not self.record_id.startswith(f"{self.source_id}:"):
            raise ValueError("record_id must start with '{source_id}:'")
        if "{{" in self.text:
            raise ValueError("unresolved {{placeholder}} in text")
        return self


class RejectedRecord(_Contract):
    """§4d: an input record excluded in Stage 3, with the stage and reason."""

    schema_version: Literal["rejected/1"] = "rejected/1"
    record_id: RecordId
    source_id: Slug
    revision: NonEmptyStr
    example_id: ExampleId | None = None
    stage: Literal["clean", "dedup", "validate"]
    reason: NonEmptyStr
    detail: str | None = None
    duplicate_of: ExampleId | None = None
    family_id: str | None = None
    allowed_use: AllowedUse
    text_sha256: Sha256

    @model_validator(mode="after")
    def _dup(self) -> Self:
        if (self.reason == "exact_duplicate") != (self.duplicate_of is not None):
            raise ValueError("duplicate_of is set exactly for reason=exact_duplicate")
        return self


class CleanManifest(_Contract):
    """Manifest of `data/interim/clean/{source_id}/{revision}/` (Stage 3 checkpoint)."""

    schema_version: Literal["clean_manifest/1"] = "clean_manifest/1"
    source_id: Slug
    revision: NonEmptyStr
    allowed_use: AllowedUse
    records_content_sha256: Sha256  # lineage: the RawRecord content this was built from
    clean_version: NonNegativeInt
    params: dict[str, object]
    params_sha256: Sha256
    code_commit: str | None
    created_at: AwareDatetime
    row_count: NonNegativeInt
    rejected_count: NonNegativeInt
    reject_reasons: dict[str, NonNegativeInt]
    flag_counts: dict[str, NonNegativeInt]
    counts: dict[str, NonNegativeInt]
    content_sha256: Sha256
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
