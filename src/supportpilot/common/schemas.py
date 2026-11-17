"""Pydantic data contracts (DATA_CONTRACTS §1-3, §5). Breaking changes bump `schema_version`."""

from __future__ import annotations

import re
from datetime import date
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
RelPath = Annotated[str, AfterValidator(_rel_posix)]
NonEmptyStr = Annotated[str, Field(min_length=1)]

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

LanguageCode = Literal["en", "ru", "kk"]


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
    "cc-by-3.0": AllowedUse.TRAIN,  # attribution only, like 4.0 (ADR-043)
    "cc-by-sa-4.0": AllowedUse.TRAIN,  # share-alike
    "cc0-1.0": AllowedUse.TRAIN,
    "mit": AllowedUse.TRAIN,
    "apache-2.0": AllowedUse.TRAIN,
    "cc-by-nc-4.0": AllowedUse.EVAL_ONLY,
    "cc-by-nc-sa-4.0": AllowedUse.EVAL_ONLY,
    # commissioned work: the contract assigns all rights, incl. ML training/evaluation (ADR-049)
    "commissioned-assignment": AllowedUse.TRAIN,
    "unknown": AllowedUse.BLOCKED,
    "conflicting": AllowedUse.BLOCKED,  # declared licence contradicts the upstream data's licence
}


class SourceStatus(StrEnum):
    SELECTED = "selected"  # acquired into data/raw
    CANDIDATE = "candidate"  # usable, not acquired yet (reason in `decision`)
    REJECTED = "rejected"  # researched and not used (reason in `decision`)
    UNAVAILABLE = "unavailable"  # wanted, but access is blocked (e.g. gated without token)


class Access(StrEnum):
    PUBLIC = "public"
    GATED = "gated"  # needs an HF account that accepted the terms (HF_TOKEN)


class FetchSpec(_Contract):
    """`hf_dataset`: a Hugging Face dataset repo; `github_repo`: a public GitHub repository.
    Both are pinned to a full commit sha and verified against upstream git/LFS hashes.
    `delivery`: a commissioned delivery taken in by `data collection intake` (ADR-049),
    pinned to the sha256 of its content; it has no remote."""

    type: Literal["hf_dataset", "github_repo", "delivery"] = "hf_dataset"
    repo_id: Annotated[str, Field(pattern=r"^[\w.-]+/[\w.-]+$")] | None = None
    include: list[NonEmptyStr] = Field(min_length=1, description="fnmatch patterns of repo files")

    @model_validator(mode="after")
    def _repo(self) -> Self:
        if (self.repo_id is None) != (self.type == "delivery"):
            raise ValueError("remote fetch types need repo_id; a delivery has none")
        return self


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
        if self.fetch is not None and self.fetch.type == "delivery":
            if not SHA256_RE.fullmatch(self.revision):
                raise ValueError("delivery sources pin `revision` to their content sha256")
        elif self.fetch is not None and not _GIT_SHA_RE.fullmatch(self.revision):
            raise ValueError(
                f"{self.fetch.type} sources must pin `revision` to a full 40-char commit sha"
            )
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


# ---------------------------------------------------------------- intent taxonomy


class Action(StrEnum):
    ANSWER_KB = "answer_kb"
    CALL_TOOL = "call_tool"
    CLARIFY = "clarify"
    ESCALATE = "escalate"
    REFUSE_OOS = "refuse_oos"


# Intent taxonomy v1. configs/taxonomy.yaml holds actions and conditions; a test keeps both equal.
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
        "change_address", "product_defect", "contact_info", "personal_data", "newsletter",
        "cart_help", "app_help", "technical_issue",
    }
)  # fmt: skip
