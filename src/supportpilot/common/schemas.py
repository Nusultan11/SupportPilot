"""Pydantic data contracts (DATA_CONTRACTS §2-4, §8-9). Breaking changes bump `schema_version`."""

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
    PositiveInt,
    model_validator,
)

from supportpilot.common.hashing import SHA256_RE, sha256_text
from supportpilot.common.ids import (
    CHUNK_ID_RE,
    DATASET_VERSION_RE,
    EXAMPLE_ID_RE,
    RECORD_ID_RE,
    make_chunk_id,
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


# Intent taxonomy draft v0 (Stage 3, ADR-021). The Stage 3 build and its `labels` gate keep
# using it, so the Stage 3 outputs stay byte-identical.
INTENTS_V0: frozenset[str] = frozenset(
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

# Intent taxonomy v1 (frozen in Stage 4, ADR-023): a superset of v0. The authoritative list
# with actions and conditions is configs/taxonomy.yaml; a unit test keeps both equal.
INTENTS: frozenset[str] = INTENTS_V0 | frozenset(
    {
        "change_address", "product_defect", "contact_info", "personal_data", "newsletter",
        "cart_help", "app_help", "technical_issue",
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


# ---------------------------------------------------------------- §5-6 SFT and model output

CHUNK_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*:[0-9a-f]{8}:\d{4,}$")


class Task(StrEnum):
    GROUNDED_ANSWER = "grounded_answer"
    TOOL_ANSWER = "tool_answer"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    ABSTAIN_NO_EVIDENCE = "abstain_no_evidence"
    REFUSE_OOS = "refuse_oos"
    CONFLICTING_CONTEXT = "conflicting_context"
    INJECTION_RESIST = "injection_resist"


class AssistantResponse(_Contract):
    """§6: the structured model output (and API output). Grounding is checked separately
    against the evidence by `supportpilot.common.grounding.check_response`."""

    language: LanguageCode
    intent: Intent
    action: Action
    answer: NonEmptyStr
    citations: list[str] = Field(default_factory=list)
    facts_from_tools: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    needs_human: bool = False
    escalation_reason: str | None = None
    clarification_question: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.needs_human and not self.escalation_reason:
            raise ValueError("needs_human requires escalation_reason")
        if self.escalation_reason and not self.needs_human:
            raise ValueError("escalation_reason without needs_human")
        if self.action is Action.ESCALATE and not self.needs_human:
            raise ValueError("action='escalate' requires needs_human")
        if (self.action is Action.CLARIFY) != (self.clarification_question is not None):
            raise ValueError("clarification_question is set exactly for action='clarify'")
        for c in self.citations:
            if not CHUNK_ID_PATTERN.fullmatch(c):
                raise ValueError(f"citation is not a chunk id: {c!r}")
        if self.action is Action.REFUSE_OOS and self.citations:
            raise ValueError("a refusal cites nothing")
        return self


class SFTMessage(_Contract):
    role: Literal["system", "user", "assistant", "tool"]
    content: NonEmptyStr
    name: str | None = None  # tool name for role=tool

    @model_validator(mode="after")
    def _tool(self) -> Self:
        if (self.role == "tool") != (self.name is not None):
            raise ValueError("name is set exactly for tool messages")
        return self


class SFTExample(_Contract):
    """§5: one supervised fine-tuning example (`sft/1`)."""

    schema_version: Literal["sft/1"] = "sft/1"
    example_id: ExampleId
    scenario_id: NonEmptyStr
    family_id: NonEmptyStr
    lang: LanguageCode
    code_switch: bool
    task: Task
    messages: list[SFTMessage] = Field(min_length=3)
    context_chunk_ids: list[str] = Field(default_factory=list)
    distractor_chunk_ids: list[str] = Field(default_factory=list)
    target: AssistantResponse
    split: Split
    # lineage and provenance
    source_id: Slug
    record_id: RecordId
    origin: Origin
    translation_of: ExampleId | None = None
    translation_system: str | None = None
    kb_variant: str | None = None
    system_prompt_id: NonEmptyStr
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _shape(self) -> Self:
        if self.messages[0].role != "system" or self.messages[-1].role != "assistant":
            raise ValueError("messages start with system and end with the assistant target")
        if not any(m.role == "user" for m in self.messages):
            raise ValueError("an SFT example needs a user message")
        if AssistantResponse.model_validate_json(self.messages[-1].content) != self.target:
            raise ValueError("the last assistant message must equal `target`")
        if self.target.language != self.lang and not self.code_switch:
            raise ValueError("target.language must equal lang")
        if not set(self.target.citations) <= set(self.context_chunk_ids):
            raise ValueError("citations must be a subset of context_chunk_ids")
        if set(self.distractor_chunk_ids) & set(self.target.citations):
            raise ValueError("a distractor chunk is cited")
        if self.origin is Origin.MACHINE_TRANSLATED and not (
            self.translation_of and self.translation_system
        ):
            raise ValueError("machine_translated requires translation_of and translation_system")
        return self


class ReviewStatus(StrEnum):
    """Gold review levels (EVALUATION §1). Only `verified` means author + independent human."""

    DRAFT = "draft"  # authored; automatic checks passed; no second review
    LLM_REVIEWED = "llm_reviewed"  # re-read by an LLM reviewer; never "independently verified"
    HUMAN_REVIEWED = "human_reviewed"  # one fluent human review
    VERIFIED = "verified"  # author + independent fluent human reviewer agree


class GoldTurn(_Contract):
    role: Literal["user", "assistant"]
    content: NonEmptyStr


class GoldFact(_Contract):
    key: NonEmptyStr  # demo-store fact key or tool field, e.g. return_window_days
    value: str  # expected surface value, e.g. "14"


class GoldEvidence(_Contract):
    doc_id: Slug
    section: NonEmptyStr


class GoldReview(_Contract):
    status: ReviewStatus
    author: NonEmptyStr
    reviewers: list[str] = Field(default_factory=list)
    notes: str = ""


class GoldItem(_Contract):
    """One gold benchmark item (`gold/1`, EVALUATION §1)."""

    schema_version: Literal["gold/1"] = "gold/1"
    gold_id: Annotated[str, Field(pattern=r"^g_[a-z0-9_]+_(?:en|ru|kk)$")]
    scenario_id: NonEmptyStr
    lang: LanguageCode
    code_switch: bool = False
    category: Literal[
        "grounded", "tool", "clarify", "escalate", "abstain", "out_of_scope",
        "conflicting", "injection", "multi_turn", "code_switch",
    ]  # fmt: skip
    difficulty: Literal["easy", "medium", "hard"]
    turns: list[GoldTurn] = Field(min_length=1)
    intent: Intent
    action: Action
    escalate: bool
    escalation_reason: str | None = None
    required_facts: list[GoldFact] = Field(default_factory=list)
    tool_state: dict[str, object] | None = None
    accepted_evidence: list[GoldEvidence] = Field(default_factory=list)
    must_not: list[str] = Field(default_factory=list)  # claims that make an answer wrong
    kb: Literal["demo_store"] = "demo_store"
    review: GoldReview

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.turns[-1].role != "user":
            raise ValueError("the last turn is the user's message to answer")
        if self.escalate != bool(self.escalation_reason):
            raise ValueError("escalate requires escalation_reason (and only then)")
        if self.action is Action.ESCALATE and not self.escalate:
            raise ValueError("action='escalate' requires escalate=True")
        if not self.gold_id.endswith(f"_{self.lang}"):
            raise ValueError("gold_id must end with _<lang>")
        if self.review.status is ReviewStatus.VERIFIED and not self.review.reviewers:
            raise ValueError("verified requires a named independent reviewer")
        return self


# ---------------------------------------------------------------- §8 KB document and chunk


ChunkId = Annotated[str, _pattern("chunk_id", CHUNK_ID_RE)]


class DocType(StrEnum):
    POLICY = "policy"
    FAQ = "faq"
    CATALOG = "catalog"
    TERMS = "terms"


class DocFormat(StrEnum):
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    JSON = "json"
    TXT = "txt"


Span = tuple[PositiveInt, PositiveInt]


class Locator(_Contract):
    """Where a piece of content sits in its source file. 1-based, inclusive ranges.

    pdf: page (+ page_end); docx: block_range over body blocks (paragraphs and tables in
    document order); xlsx: sheet + row_range (Excel rows); csv: row_range (file lines);
    json: json_pointer (RFC 6901); txt: line_range. `record_id` is the row key of a record.
    """

    page: PositiveInt | None = None
    page_end: PositiveInt | None = None
    sheet: str | None = None
    row_range: Span | None = None
    block_range: Span | None = None
    line_range: Span | None = None
    json_pointer: str | None = None
    record_id: str | None = None

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if all(v is None for v in self.model_dump().values()):
            raise ValueError("an empty locator points nowhere")
        for span in (self.row_range, self.block_range, self.line_range):
            if span is not None and span[0] > span[1]:
                raise ValueError(f"range start after end: {span}")
        if self.page_end is not None and (self.page is None or self.page_end < self.page):
            raise ValueError("page_end needs page <= page_end")
        if self.json_pointer is not None and not self.json_pointer.startswith("/"):
            raise ValueError("json_pointer must start with '/'")
        return self


class KBTable(_Contract):
    header: list[str] = Field(min_length=1)
    rows: list[list[str]]
    locator: Locator

    @model_validator(mode="after")
    def _rectangular(self) -> Self:
        if any(len(r) != len(self.header) for r in self.rows):
            raise ValueError("every table row needs one cell per header column")
        return self


class KBSection(_Contract):
    """One node of the document tree. Records (catalogue rows) are level-1 sections whose
    single-row table holds the record; `section_id` is the record key."""

    section_id: Slug | None = None
    heading: str | None = None
    level: NonNegativeInt
    text: str = ""
    tables: list[KBTable] = Field(default_factory=list)
    locator: Locator
    children: list[KBSection] = Field(default_factory=list)


class KBDocument(_Contract):
    schema_version: Literal["kb_document/1"] = "kb_document/1"
    doc_id: Slug
    version: Sha256  # sha256 of the source file bytes
    lang: LanguageCode
    title: NonEmptyStr
    doc_type: DocType
    format: DocFormat
    source_path: RelPath
    effective_from: date
    effective_to: date | None = None
    parser: NonEmptyStr  # library/version + parser revision
    sections: list[KBSection] = Field(min_length=1)  # level 0 = text before the first heading

    @model_validator(mode="after")
    def _dates(self) -> Self:
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to before effective_from")
        return self


class KBChunk(_Contract):
    schema_version: Literal["kb_chunk/1"] = "kb_chunk/1"
    chunk_id: ChunkId
    doc_id: Slug
    version: Sha256
    lang: LanguageCode
    position: NonNegativeInt
    part: NonNegativeInt = 0  # > 0 when a long section was split
    section_id: Slug | None
    section_path: list[str]
    title: NonEmptyStr
    text: NonEmptyStr
    locator: Locator
    doc_type: DocType
    format: DocFormat
    source_path: RelPath
    effective_from: date
    effective_to: date | None = None
    token_count: PositiveInt
    translation_group: NonEmptyStr
    checksum: Sha256  # sha256 of `text`

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.chunk_id != make_chunk_id(self.doc_id, self.version, self.position):
            raise ValueError("chunk_id must be {doc_id}:{version[:8]}:{position:04d}")
        if self.checksum != sha256_text(self.text):
            raise ValueError("checksum must be sha256(text)")
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("effective_to before effective_from")
        return self
