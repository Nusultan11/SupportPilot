"""Corpus recipe (`configs/corpus.yaml`, `corpus_recipe/1`) and label decisions.

A label is an authored mapping of the *source's own* label to a v1 intent, optionally
confirmed by a cue in the text. It is never independent human adjudication. A row whose
text does not support the mapping goes to the `unverified` pool with its candidate intent.
"""

from __future__ import annotations

import re
from functools import cached_property
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.schemas import INTENTS, Action, LanguageCode
from supportpilot.data.records import ActionRule, LabelStatus, Origin
from supportpilot.data.taxonomy import NEEDS_REFERENCE, Taxonomy

_FLAGS = re.IGNORECASE | re.UNICODE


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, _FLAGS)


class Rule(_Model):
    cue: str
    intent: str
    action: Action | None = None
    escalation_reason: str | None = None

    @cached_property
    def regex(self) -> re.Pattern[str]:
        return compile_pattern(self.cue)


class Veto(_Model):
    pattern: str
    reason: str

    @cached_property
    def regex(self) -> re.Pattern[str]:
        return compile_pattern(self.pattern)


class LabelSpec(_Model):
    """Exactly one of `exclude`, `intent` (direct mapping), `rules` (cue-confirmed) or a bare
    `candidate` (always unverified: the source label does not determine one v1 intent)."""

    exclude: str | None = None
    intent: str | None = None
    rules: list[Rule] = Field(default_factory=list)
    candidate: str | None = None  # intent recorded for unverified rows; default: first rule
    veto: list[Veto] = Field(default_factory=list)
    min_words: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        bare_candidate = self.candidate is not None and not self.rules
        kinds = [self.exclude is not None, self.intent is not None, bool(self.rules)]
        if sum(kinds) + bare_candidate != 1:
            raise ValueError("a label needs exactly one of exclude / intent / rules / candidate")
        for name in [self.intent, self.candidate, *(r.intent for r in self.rules)]:
            if name is not None and name not in INTENTS:
                raise ValueError(f"unknown intent {name!r}")
        return self


class SourceRecipe(_Model):
    kind: Literal["intent", "grounded"] = "intent"
    active: bool = True
    inactive_reason: str | None = None
    domain: str
    origin: dict[LanguageCode, Origin]
    placement: Literal["any", "heldout"] = "any"
    min_words: int | None = Field(default=None, ge=1)
    evidence: str
    named_product: str | None = None  # regex: the request names a concrete product
    veto: list[Veto] = Field(default_factory=list)  # source-wide exclusions
    labels: dict[str, LabelSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if not self.active and not self.inactive_reason:
            raise ValueError("an inactive source needs inactive_reason")
        if self.kind == "intent" and self.active and not self.labels:
            raise ValueError("an active intent source needs a label table")
        return self


class SplitRatios(_Model):
    train: float = Field(gt=0, lt=1)
    validation: float = Field(gt=0, lt=1)
    test: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def _sum(self) -> Self:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1")
        return self


class NearDup(_Model):
    shingle: int = Field(default=4, ge=2)
    jaccard: float = Field(default=0.7, gt=0, le=1)
    num_perm: int = Field(default=64, ge=16)
    bands: int = Field(default=16, ge=2)

    @model_validator(mode="after")
    def _bands(self) -> Self:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be divisible by bands")
        return self


class Recipe(_Model):
    schema_version: Literal["corpus_recipe/1"] = "corpus_recipe/1"
    seed: int
    split: SplitRatios
    neardup: NearDup = NearDup()
    min_words: int = Field(default=3, ge=1)
    max_chars: int = Field(default=1000, ge=50)
    review_sample_per_language: int = Field(default=100, ge=10)
    sources: dict[str, SourceRecipe]


def load_recipe(path: Path) -> Recipe:
    return Recipe.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- decisions


class LabelDecision(_Model):
    """`intent is None` means excluded (reason in `reason`)."""

    intent: str | None
    status: LabelStatus | None
    rule: str
    action: Action | None = None
    escalation_reason: str | None = None
    reason: str = ""


def decide_label(
    source_id: str, source: SourceRecipe, label: str, text: str, n_words: int, min_words: int
) -> LabelDecision:
    rule_id = f"{source_id}.{label}"
    for v in source.veto:
        if v.regex.search(text):
            return LabelDecision(intent=None, status=None, rule=rule_id, reason=v.reason)
    spec = source.labels.get(label)
    if spec is None:
        return LabelDecision(
            intent=None, status=None, rule=rule_id, reason="label not in recipe (unmapped)"
        )
    if spec.exclude is not None:
        return LabelDecision(intent=None, status=None, rule=rule_id, reason=spec.exclude)
    for v in spec.veto:
        if v.regex.search(text):
            return LabelDecision(intent=None, status=None, rule=rule_id, reason=v.reason)
    min_words = spec.min_words or source.min_words or min_words
    if spec.intent is not None:
        if n_words < min_words:
            return LabelDecision(
                intent=spec.intent, status=LabelStatus.UNVERIFIED, rule=rule_id + ":short"
            )
        return LabelDecision(intent=spec.intent, status=LabelStatus.MAPPED_DIRECT, rule=rule_id)
    for i, r in enumerate(spec.rules):
        if r.regex.search(text):
            status = LabelStatus.MAPPED_CUE if n_words >= min_words else LabelStatus.UNVERIFIED
            suffix = f":rule{i}" + ("" if status is LabelStatus.MAPPED_CUE else ":short")
            return LabelDecision(
                intent=r.intent,
                status=status,
                rule=rule_id + suffix,
                action=r.action,
                escalation_reason=r.escalation_reason,
            )
    candidate = spec.candidate or spec.rules[0].intent
    return LabelDecision(intent=candidate, status=LabelStatus.UNVERIFIED, rule=rule_id + ":no_cue")


_ORDER_REF = compile_pattern(
    r"\b\d{6,}\b|<number>|<order_id>|\border\s*(?:id|number|no\.?|#|№)\s*[:#№]?\s*\w*\d"
    r"|(?:заказ|тапсырыс)\w*\s*(?:№|номер|нөмір\w*)\s*\d"
)


def has_order_reference(text: str) -> bool:
    return _ORDER_REF.search(text) is not None


def derive_action(
    taxonomy: Taxonomy,
    intent: str,
    text: str,
    decision: LabelDecision,
    named_product: re.Pattern[str] | None,
) -> tuple[Action, str | None, ActionRule]:
    """Action from taxonomy v1 conditions; a recipe rule may override it (e.g. escalation)."""
    spec = taxonomy.intents[intent]
    if decision.action is not None:
        reason = decision.escalation_reason if decision.action is Action.ESCALATE else None
        return decision.action, reason, ActionRule.RECIPE_OVERRIDE
    if spec.action == NEEDS_REFERENCE:
        if has_order_reference(text):
            return Action.CALL_TOOL, None, ActionRule.REFERENCE_PRESENT
        return Action.CLARIFY, None, ActionRule.REFERENCE_MISSING
    if intent in {"product_info", "availability"}:
        if named_product is not None and named_product.search(text):
            named = Action.ANSWER_KB if intent == "product_info" else Action.CALL_TOOL
            return named, None, ActionRule.NAMED_PRODUCT
        return Action.CLARIFY, None, ActionRule.UNNAMED_PRODUCT
    action = Action(spec.action)
    return action, spec.escalation_reason, ActionRule.TAXONOMY_DEFAULT
