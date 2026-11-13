"""Frozen multilingual encoder + trained linear head, with explicit policy guards."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import INTENTS, Action, AssistantResponse, LanguageCode
from supportpilot.data.labels import has_reference
from supportpilot.data.taxonomy import Taxonomy
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.models import ModelSpec
from supportpilot.routing.models import RouteDecision

_HUMAN = re.compile(
    r"\b(?:operator|human agent|оператор\w*|менеджер\w*|адаммен|қызметкермен)\b", re.I
)
_LEGAL = re.compile(r"\b(?:lawyer|sue|court|адвокат\w*|суд\w*|заңгер\w*|сотқа)\b", re.I)
_SAFETY = re.compile(
    r"\b(?:injury|injured|electric shock|fire|burned|травм\w*|ожог\w*|өрт|күйік|ток соқты)\b", re.I
)


def policy_decision(
    intent: str,
    confidence: float,
    message: str,
    session: dict[str, str],
    language: LanguageCode,
    code_switch: bool,
    taxonomy: Taxonomy,
) -> RouteDecision:
    reason = None
    if _SAFETY.search(message):
        reason = "safety_risk"
    elif _LEGAL.search(message):
        reason = "legal_threat"
    elif _HUMAN.search(message):
        reason = "customer_requested_human"
    spec = taxonomy.intents[intent]
    action = spec.action
    if reason:
        action = Action.ESCALATE
    elif action == "needs_reference":
        action = Action.CALL_TOOL if has_reference(message, session) else Action.CLARIFY
    elif action == Action.ESCALATE:
        reason = spec.escalation_reason
    return RouteDecision(
        language=language,
        code_switch=code_switch,
        intent=intent,
        action=Action(action),
        confidence=confidence,
        slots=session,
        escalate=bool(reason),
        escalation_reason=reason,
    )


def llm_decision(response: AssistantResponse, *, code_switch: bool = False) -> RouteDecision:
    """LLM confidence is unknown; do not fabricate a calibrated probability."""
    return RouteDecision(
        language=response.language,
        code_switch=code_switch,
        intent=response.intent,
        action=response.action,
        confidence=None,
        escalate=response.needs_human,
        escalation_reason=response.escalation_reason,
    )


def fit_head(
    vectors: npt.NDArray[np.float32],
    labels: list[str],
    *,
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Deterministic multinomial linear classifier on frozen encoder representations."""
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

    if set(labels) != INTENTS:
        raise ValueError("encoder router training must cover all 35 intents")
    classifier = LogisticRegression(C=regularization, max_iter=1000, random_state=20261113)
    classifier.fit(vectors, labels)
    return {
        "labels": classifier.classes_.tolist(),
        "weights": classifier.coef_.tolist(),
        "bias": classifier.intercept_.tolist(),
        "regularization": regularization,
    }


class EncoderRouter:
    def __init__(
        self, encoder: ModelSpec, head: dict[str, Any], taxonomy: Taxonomy, *, device: str = "cpu"
    ) -> None:
        self.encoder = Transformer(encoder, device=device)
        self.head = head
        self.taxonomy = taxonomy
        self.last_seconds = 0.0

    def route(
        self,
        message: str,
        session: dict[str, str],
        *,
        language: LanguageCode,
        code_switch: bool = False,
    ) -> RouteDecision:
        started = time.perf_counter()
        vector = self.encoder.encode([message], query=True)[0]
        logits = np.asarray(self.head["weights"]) @ vector + np.asarray(self.head["bias"])
        scores = np.exp(logits - logits.max())
        scores /= scores.sum()
        index = int(scores.argmax())
        decision = policy_decision(
            str(self.head["labels"][index]),
            float(scores[index]),
            message,
            session,
            language,
            code_switch,
            self.taxonomy,
        )
        self.last_seconds = time.perf_counter() - started
        return decision

    def save(self, path: Path) -> None:
        write_json(
            path,
            {
                "schema_version": "encoder_router/1",
                "encoder": self.encoder.spec.model_dump(mode="json"),
                "head": self.head,
            },
        )

    @classmethod
    def load(cls, path: Path, taxonomy: Taxonomy, *, device: str = "cpu") -> EncoderRouter:
        data = read_json(path)
        if data["schema_version"] != "encoder_router/1":
            raise ValueError("unsupported router artifact")
        return cls(ModelSpec.model_validate(data["encoder"]), data["head"], taxonomy, device=device)
