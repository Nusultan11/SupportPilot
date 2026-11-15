"""Explicit service acceptance; independent qualification is never inferred from a verdict."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal, Protocol, Self
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.grounding import Evidence
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import read_json
from supportpilot.common.schemas import Action, AssistantResponse, LanguageCode
from supportpilot.llm.generation import (
    Backend,
    GenerationResult,
    SupportVerdict,
    generate_grounded,
    safe_fallback,
)
from supportpilot.retrieval.models import RetrievalContext
from supportpilot.service.semantic_evaluation import ClaimReview, evaluate_reviews

Scope = Literal["en", "ru", "kk", "mixed"]
Digest = str


class QualificationApproval(BaseModel):
    """Trusted offline owner approval, never accepted from an HTTP request.

    Software checks declarations and calibration, not the truth of human identities,
    fluency or adjudication. The named owner must verify the referenced private receipts.
    No approval or qualified implementation ships with the default runtime.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["verifier_approval/1"] = "verifier_approval/1"
    approved_by: str = Field(min_length=1)
    reviews_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    overlap_adjudication_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    human_identities_and_fluency_verified: Literal[True]
    overlap_independence_verified: Literal[True]
    judge_revision: str = Field(min_length=1)
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scopes: tuple[Scope, ...] = Field(min_length=1)
    kb_version: str = Field(min_length=1)
    index_version: str = Field(min_length=1)
    approved_at: AwareDatetime
    valid_until: AwareDatetime

    @model_validator(mode="after")
    def valid(self) -> Self:
        if self.valid_until <= self.approved_at or len(set(self.scopes)) != len(self.scopes):
            raise ValueError("invalid approval interval or duplicate scopes")
        return self


class ClaimAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    verdict: SupportVerdict
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    qualification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    judge_revision: str = Field(min_length=1)
    rubric_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    issued_at: AwareDatetime
    valid_until: AwareDatetime


class ServiceClaimVerifier(Protocol):
    def __call__(
        self,
        response: AssistantResponse,
        query: str,
        evidence: Evidence,
        *,
        binding_sha256: str,
        qualification_sha256: str,
    ) -> ClaimAssessment:
        """Assess every claim, relevance, conflict, exceptions and completeness.

        Return a fresh result bound to this call. Merely echoing the binding is not
        semantic verification; a separately qualified implementation is mandatory.
        """
        ...


@dataclass(frozen=True)
class QualifiedVerifier:
    approval: QualificationApproval
    identity: Digest
    assess: ServiceClaimVerifier

    def current(self, scope: Scope) -> bool:
        now = datetime.now(UTC)
        return (
            scope in self.approval.scopes
            and self.approval.approved_at <= now < self.approval.valid_until
        )


@dataclass(frozen=True)
class ServiceGenerationResult(GenerationResult):
    valid_until: datetime | None = None


def qualify_verifier(
    reviews: Path,
    approval: Path,
    assessor: ServiceClaimVerifier,
    *,
    judge_revision: str,
    rubric_sha256: str,
    reviewer_registry: Path,
    overlap_adjudication: Path,
) -> QualifiedVerifier:
    """Validate an explicitly supplied packet; this does not certify human declarations."""
    owner = QualificationApproval.model_validate(read_json(approval))
    if sha256_file(reviews) != owner.reviews_sha256:
        raise ValueError("review packet differs from owner approval")
    if (
        sha256_file(reviewer_registry) != owner.reviewer_registry_sha256
        or sha256_file(overlap_adjudication) != owner.overlap_adjudication_sha256
    ):
        raise ValueError("independent provenance receipts differ from owner approval")
    if (judge_revision, rubric_sha256) != (owner.judge_revision, owner.rubric_sha256):
        raise ValueError("verifier implementation differs from qualified judge/rubric")
    rows = [ClaimReview.model_validate(r) for r in read_json(reviews)]
    if any((r.judge_revision, r.rubric_sha256) != (judge_revision, rubric_sha256) for r in rows):
        raise ValueError("review judge/rubric differs from approval")
    metrics = evaluate_reviews(rows)
    for scope in owner.scopes:
        values = metrics[scope]
        if not isinstance(values, dict) or values["status"] != "EVALUATED":
            raise ValueError("independent calibration and assessment required for every scope")
    identity = sha256_text(owner.model_dump_json())
    qualified = QualifiedVerifier(owner, identity, assessor)
    if not all(qualified.current(scope) for scope in owner.scopes):
        raise ValueError("qualification is not currently valid")
    return qualified


def generate_for_service(
    backend: Backend,
    query: str,
    language: LanguageCode,
    context: RetrievalContext,
    *,
    identify_language: Callable[[str], str],
    verifier: QualifiedVerifier | None = None,
    code_switch: bool = False,
) -> ServiceGenerationResult:
    """Adapt a qualified, call-bound assessment to the existing ClaimVerifier boundary."""
    scope: Scope = "mixed" if code_switch else language
    today = date.today()
    valid_context = bool(context.evidence) and all(
        e.chunk.effective_from <= today
        and (e.chunk.effective_to is None or e.chunk.effective_to >= today)
        and e.chunk.checksum == sha256_text(e.chunk.text)
        for e in context.evidence
    )
    if verifier is not None and (
        not verifier.current(scope)
        or not valid_context
        or (context.kb_version, context.index_version)
        != (verifier.approval.kb_version, verifier.approval.index_version)
        or len({e.chunk_id for e in context.evidence}) != len(context.evidence)
    ):
        return ServiceGenerationResult(
            safe_fallback(language), None, ("verifier_not_qualified_for_context",), True, "unknown"
        )
    started = datetime.now(UTC)
    nonce = uuid4().hex
    accepted_until: datetime | None = None

    def verify(response: AssistantResponse, query: str, evidence: Evidence) -> SupportVerdict:
        nonlocal accepted_until
        if verifier is None or not verifier.current(scope):
            return "unknown"
        if (
            response.action != Action.ANSWER_KB
            or response.needs_human
            or response.missing_information
            or response.facts_from_tools
            or not response.citations
            or len(set(response.citations)) != len(response.citations)
        ):
            return "unknown"
        binding = sha256_text(
            json.dumps(
                {
                    "nonce": nonce,
                    "query": query,
                    "language": language,
                    "code_switch": code_switch,
                    "response": response.model_dump(mode="json"),
                    "context": context.model_dump(mode="json"),
                    "qualification": verifier.identity,
                },
                sort_keys=True,
                ensure_ascii=False,
            )
        )
        # A verifier is not allowed to mutate the candidate or its packed evidence.
        observation = verifier.assess(
            response.model_copy(deep=True),
            query,
            Evidence(chunks=dict(evidence.chunks)),
            binding_sha256=binding,
            qualification_sha256=verifier.identity,
        )
        if not isinstance(observation, ClaimAssessment):
            return "unknown"
        now = datetime.now(UTC)
        if (
            observation.binding_sha256 != binding
            or observation.qualification_sha256 != verifier.identity
            or observation.judge_revision != verifier.approval.judge_revision
            or observation.rubric_sha256 != verifier.approval.rubric_sha256
            or not started <= observation.issued_at <= now < observation.valid_until
            or observation.valid_until > verifier.approval.valid_until
            or not verifier.current(scope)
        ):
            return "unknown"
        if observation.verdict == "supported":
            accepted_until = observation.valid_until
        return observation.verdict

    result = generate_grounded(
        backend,
        query,
        language,
        context,
        identify_language=identify_language,
        verify_claims=verify if verifier is not None else None,
        prompt_mode="schema",
    )
    return ServiceGenerationResult(
        result.response,
        result.raw,
        result.problems,
        result.fallback,
        result.semantic_support,
        accepted_until,
    )
