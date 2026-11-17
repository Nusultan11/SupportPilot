"""Corpus record contracts (DATA_CONTRACTS §6-8). Breaking changes bump `schema_version`."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from supportpilot.common.schemas import INTENTS, Action, FileEntry, LanguageCode, Sha256, Slug

CORPUS_NAME = "corpus"
NOT_EVALUATED = "NOT_EVALUATED"


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Origin(StrEnum):
    """How the customer text was produced. None of them is synthetic or machine-translated."""

    HUMAN_WRITTEN = "human_written"  # crowd workers wrote the request from a prompt
    HUMAN_ROLEPLAY = "human_roleplay"  # crowd workers role-played a customer in a live chat
    HUMAN_LOCALISED = "human_localised"  # human translation/localisation of another utterance
    HUMAN_SPOKEN_ASR = "human_spoken_asr"  # human speech, machine (ASR) transcription


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class Pool(StrEnum):
    CORE = "core"  # label from a mapping that the text supports; enters train/validation/test
    UNVERIFIED = "unverified"  # plausible label the text does not confirm; never in core files


class LabelStatus(StrEnum):
    MAPPED_DIRECT = "mapped_direct"  # the source label alone determines the v1 intent
    MAPPED_CUE = "mapped_cue"  # the source label plus a cue found in the text
    UNVERIFIED = "unverified"  # candidate intent; no supporting cue or too short


class ActionRule(StrEnum):
    TAXONOMY_DEFAULT = "taxonomy_default"
    RECIPE_OVERRIDE = "recipe_override"
    REFERENCE_PRESENT = "needs_reference:reference_present"  # call_tool
    REFERENCE_MISSING = "needs_reference:reference_missing"  # clarify
    NAMED_PRODUCT = "named_product"  # product_info -> answer_kb, availability -> call_tool
    UNNAMED_PRODUCT = "unnamed_product"  # clarify


class Quality(_Record):
    words: NonNegativeInt
    chars: NonNegativeInt
    lang_guess: Literal["en", "ru", "kk", "unknown"]
    lang_consistent: bool
    pii_masked: dict[str, int] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)
    human_review: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"


class IntentExample(_Record):
    """One customer request with a v1 intent/action label (`corpus_example/1`)."""

    schema_version: Literal["corpus_example/1"] = "corpus_example/1"
    example_id: str = Field(pattern=r"^ex-[0-9a-f]{16}$")
    source_id: Slug
    source_revision: str
    source_item_id: str
    source_split: str | None
    source_label: str
    text: str = Field(min_length=1)
    text_sha256: Sha256
    lang: LanguageCode
    code_switch: bool
    origin: Origin
    is_translation: bool
    parallel_id: str | None
    domain: str
    intent: str
    action: Action
    escalation_reason: str | None
    label_status: LabelStatus
    label_rule: str
    action_rule: ActionRule
    pool: Pool
    group_id: str
    family_id: str
    split: Split
    quality: Quality

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.intent not in INTENTS:
            raise ValueError(f"unknown intent {self.intent!r}")
        if (self.pool is Pool.CORE) == (self.label_status is LabelStatus.UNVERIFIED):
            raise ValueError("core rows need a verified mapping; unverified rows stay out of core")
        if self.is_translation != (self.origin is Origin.HUMAN_LOCALISED):
            raise ValueError("is_translation must equal origin == human_localised")
        if (self.action is Action.ESCALATE) != (self.escalation_reason is not None):
            raise ValueError("escalate needs exactly one escalation_reason")
        return self


class Exclusion(_Record):
    """A source row that did not enter the corpus, with the reason (`corpus_exclusion/1`)."""

    schema_version: Literal["corpus_exclusion/1"] = "corpus_exclusion/1"
    source_id: Slug
    source_item_id: str
    lang: LanguageCode
    source_label: str
    reason: str
    detail: str = ""
    duplicate_of: str | None = None


class Evidence(_Record):
    span_id: str
    text: str


class Turn(_Record):
    role: Literal["user", "agent"]
    text: str


class GroundedExample(_Record):
    """One document-grounded agent response with its evidence spans (`grounded_turn/1`)."""

    schema_version: Literal["grounded_turn/1"] = "grounded_turn/1"
    example_id: str = Field(pattern=r"^gr-[0-9a-f]{16}$")
    source_id: Slug
    source_revision: str
    dialogue_id: str
    turn_id: int
    lang: LanguageCode
    origin: Origin
    domain: str
    doc_id: str
    context: list[Turn] = Field(min_length=1)
    response: str = Field(min_length=1)
    dialogue_act: str
    evidence: list[Evidence] = Field(min_length=1)
    group_id: str
    split: Split
    flags: list[str] = Field(default_factory=list)
    human_review: Literal["NOT_EVALUATED"] = "NOT_EVALUATED"


class CorpusManifest(_Record):
    """`corpus_manifest/1`: binds the published files to inputs, recipe, code and hashes."""

    schema_version: Literal["corpus_manifest/1"] = "corpus_manifest/1"
    version: str = Field(pattern=r"^corpus-\d{8}-[0-9a-f]{8}$")
    created_at: AwareDatetime
    code_commit: str | None
    recipe_sha256: Sha256
    taxonomy_sha256: Sha256
    registry_sha256: Sha256
    seed: int
    inputs: dict[str, str] = Field(description="source_id -> raw manifest sha256")
    counts: dict[str, int]
    content_sha256: Sha256
    files: list[FileEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _version_matches_content(self) -> Self:
        if not self.version.endswith(self.content_sha256[:8]):
            raise ValueError("version suffix must be content_sha256[:8]")
        return self
