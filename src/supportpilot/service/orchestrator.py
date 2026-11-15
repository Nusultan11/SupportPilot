"""Bounded sessions and fail-closed delivery around the existing research libraries."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol, TypeVar
from uuid import UUID, uuid4

from supportpilot.common.config import ServingSettings
from supportpilot.common.schemas import Action, AssistantResponse, LanguageCode
from supportpilot.data.langid import LanguageIdentifier, LanguageResult
from supportpilot.data.taxonomy import Taxonomy
from supportpilot.llm.generation import Backend, safe_fallback
from supportpilot.retrieval.models import RetrievalContext
from supportpilot.routing.models import RouteDecision
from supportpilot.routing.router import policy_decision
from supportpilot.service.acceptance import QualifiedVerifier, generate_for_service
from supportpilot.service.contracts import ChatRequest, ChatResult, ServiceError, Source
from supportpilot.service.decisions import critical_request, explicit_intent, service_decision
from supportpilot.service.store import Store

T = TypeVar("T")


class Router(Protocol):
    def route(
        self,
        message: str,
        session: dict[str, str],
        *,
        language: LanguageCode,
        code_switch: bool = False,
    ) -> RouteDecision: ...


class ContextProvider(Protocol):
    def context(self, query: str, lang: str | None = None) -> RetrievalContext: ...


@dataclass
class Session:
    customer: str
    updated: float
    language: LanguageCode = "en"
    order: str | None = None
    sku: str | None = None
    pending: str | None = None
    failures: int = 0
    history: list[str] = field(default_factory=list)
    decision: RouteDecision | None = None


# Historical Stage 9 guard identity for immutable evaluation compatibility only.
# Active service interpretation is in service.decisions; no unrestricted-risk claim.
_RISK = re.compile(
    r"\b(?:operator|human|agent|fraud|stolen|injur\w*|burn\w*|fire|shock|smoke|"
    r"overheat\w*|swollen|bleed\w*|allerg\w*|court|lawyer|police|privacy|password|"
    r"cvv|again|repeated|ignore)\b|\b(?:оператор|менеджер|адаммен|қызметкер|жалоб|"
    r"complaint|шағым|мошенн|алаяқ|взлом|ұрла|травм|ожог|горит|дым|нагрев|взду|"
    r"кров|өрт|күйік|түтін|жарақат|суд|адвокат|полици|сот|заңгер|персональн|"
    r"парол|құпиясөз)\w*|charged twice|дважды|екі рет|personal data|жеке дерек|"
    r"delete.*account|удали.*аккаунт|\b(?:снова|опять|қайта)\b|system prompt|игнор|"
    r"нұсқау|customer[_ -]?id|demo_bob|demo_alice",
    re.IGNORECASE,
)
_ORDER = re.compile(r"\bSP-\d{4}\b", re.IGNORECASE)
_SKU = re.compile(r"\b(?!SP-)[A-Z]{2}-\d{4}\b", re.IGNORECASE)
_COPY: dict[str, dict[LanguageCode, str]] = {
    "confirm": {
        "en": "Confirm a cancellation request using the confirmation control.",
        "ru": "Подтвердите запрос на отмену с помощью поля подтверждения.",
        "kk": "Болдырмау сұрауын растау өрісі арқылы растаңыз.",
    },
    "reference": {
        "en": "Please provide one order ID (SP-1001) or SKU (TV-5501).",
        "ru": "Укажите один номер заказа (SP-1001) или SKU (TV-5501).",
        "kk": "Бір тапсырыс нөмірін (SP-1001) немесе SKU (TV-5501) көрсетіңіз.",
    },
    "clarify": {
        "en": "What would you like to know about your order, stock or store policy?",
        "ru": "Что вы хотите узнать о заказе, наличии товара или правилах магазина?",
        "kk": "Тапсырыс, тауар қоры немесе дүкен ережелері туралы не білгіңіз келеді?",
    },
    "queued": {
        "en": (
            "Your request is recorded in the local operator queue. No response time is confirmed."
        ),
        "ru": "Обращение записано в локальную очередь оператора. Время ответа не подтверждено.",
        "kk": "Өтініш жергілікті оператор кезегіне тіркелді. Жауап беру уақыты расталмаған.",
    },
    "tool": {
        "en": "Authorized synthetic tool result",
        "ru": "Результат авторизованного синтетического инструмента",
        "kk": "Рұқсат етілген синтетикалық құрал нәтижесі",
    },
}


class SupportService:
    def __init__(
        self,
        store: Store,
        settings: ServingSettings,
        taxonomy: Taxonomy,
        router: Router | None = None,
        retriever: ContextProvider | None = None,
        backend: Backend | None = None,
        *,
        verifier: QualifiedVerifier | None = None,
    ) -> None:
        self.store, self.settings, self.taxonomy = store, settings, taxonomy
        self.router, self.retriever, self.backend = router, retriever, backend
        self.verifier = verifier
        self.sessions: dict[UUID, Session] = {}
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="support-components")
        self.pending_call: Future[Any] | None = None
        self.lid = LanguageIdentifier()
        self.closed = False

    def close(self) -> None:
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)

    def external(self, call: Callable[[], T], deadline: float) -> T:
        # A timeout never starts an unbounded second GPU/thread job. Component calls have no
        # business writes; late results are discarded. Restart for a permanently stuck driver.
        if self.pending_call is not None and not self.pending_call.done():
            raise ServiceError(503, "component_busy")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError
        future = self.pool.submit(call)
        self.pending_call = future
        return future.result(timeout=remaining)

    def ready(self) -> bool:
        if self.closed or (self.pending_call is not None and not self.pending_call.done()):
            return False
        try:
            with closing(self.store.connect()) as db:
                return bool(db.execute("PRAGMA quick_check").fetchone()[0] == "ok")
        except Exception:
            return False

    def chat(self, customer: str, request: ChatRequest) -> ChatResult:
        if customer == "operator":
            raise ServiceError(403, "customer_required")
        if self.closed or not self.lock.acquire(blocking=False):
            raise ServiceError(503, "service_busy")
        try:
            return self._chat(customer, request)
        finally:
            self.lock.release()

    def _chat(self, customer: str, request: ChatRequest) -> ChatResult:
        now = time.monotonic()
        deadline = now + self.settings.request_timeout_s
        fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        with closing(self.store.connect()) as db:
            receipt = db.execute(
                "SELECT fingerprint,result FROM receipts WHERE customer=? AND request=?",
                (customer, str(request.request_id)),
            ).fetchone()
        if receipt:
            if receipt["fingerprint"] != fingerprint:
                raise ServiceError(409, "request_id_reused")
            return ChatResult.model_validate_json(receipt["result"])
        self.sessions = {
            k: v for k, v in self.sessions.items() if now - v.updated < self.settings.session_ttl_s
        }
        identity = request.session_id or uuid4()
        session = self.sessions.get(identity)
        if request.session_id and (session is None or session.customer != customer):
            raise ServiceError(404, "session_unavailable")
        if session is None:
            if len(self.sessions) >= self.settings.max_sessions:
                raise ServiceError(503, "session_capacity")
            session = Session(customer, now)
        # Copy before mutation: failed publication cannot leave a partially advanced session.

        session = replace(session, history=list(session.history))
        session.decision = None
        language_failed = False
        try:
            detected = self.external(lambda: self.lid.detect(request.message), deadline)
        except Exception:
            detected = LanguageResult("und", 0, "None", False, ())
            language_failed = True
        lang: LanguageCode = request.language or (
            detected.lang if detected.lang in {"ru", "kk", "en"} else session.language
        )  # type: ignore[assignment]
        session.language = lang
        sources: list[Source] = []
        tool: str | None = None
        ref: str | None = None
        reason: str | None = None
        diagnostic = "deterministic"
        intent = "unclear"
        accepted_until: datetime | None = None
        orders = [value.upper() for value in _ORDER.findall(request.message)]
        skus = [value.upper() for value in _SKU.findall(request.message)]
        if language_failed:
            reason, diagnostic = "language_unavailable", "component_timeout"
        elif critical := critical_request(request.message):
            reason, intent = critical
            session.decision = service_decision(
                intent, request.message, {}, lang, detected.code_switch, self.taxonomy
            )
        elif session.failures >= 2:
            reason = "repeated_failure"
        else:
            malformed = any(
                not (_ORDER.fullmatch(value) or _SKU.fullmatch(value))
                for value in re.findall(r"\b[A-Z]{2,3}-[\w-]+", request.message, re.I)
            )
            if malformed or len(set(orders)) > 1 or len(set(skus)) > 1 or (orders and skus):
                response = self.clarify(lang, "reference")
                session.pending = None
            else:
                try:
                    slots = {
                        key: value
                        for key, value in {
                            "order_id": orders[0] if orders else session.order,
                            "sku": skus[0] if skus else session.sku,
                        }.items()
                        if value is not None
                    }
                    if session.pending and re.fullmatch(
                        r"\s*(?:SP-\d{4}|(?!SP-)[A-Z]{2}-\d{4})\s*[.!]?",
                        request.message,
                        re.I,
                    ):
                        slots["pending_intent"] = session.pending
                        route = service_decision(
                            session.pending,
                            request.message,
                            slots,
                            lang,
                            detected.code_switch,
                            self.taxonomy,
                        )
                    else:
                        route = self.external(
                            lambda: self.route(request.message, lang, detected.code_switch, slots),
                            deadline,
                        )
                    session.decision = route
                    intent = route.intent
                    session.pending = None
                    if route.escalate:
                        reason = route.escalation_reason or "routing_escalation"
                    elif route.action == Action.CALL_TOOL:
                        tool = self.taxonomy.intents[intent].tool
                        if tool is None:
                            reason = "unsupported_operation"
                        elif (tool == "get_stock" and orders) or (tool != "get_stock" and skus):
                            # An explicitly supplied reference of the wrong type must
                            # never fall back to an unrelated carried order or SKU.
                            tool = None
                            session.pending = intent
                            response = self.clarify(lang, "reference")
                        elif tool == "request_cancellation" and not request.confirm_cancellation:
                            session.pending = intent
                            tool = None
                            response = self.clarify(lang, "confirm")
                        ref = slots.get("sku" if tool == "get_stock" else "order_id")
                        if tool is not None and ref is None:
                            session.pending = intent
                            response = self.clarify(lang, "reference")
                    elif route.action == Action.ANSWER_KB:
                        if self.retriever is None:
                            reason, diagnostic = "evidence_unavailable", "retrieval_unavailable"
                        else:
                            context = self.external(
                                lambda: self.context(request.message, lang), deadline
                            )
                            sources = [
                                Source(
                                    chunk_id=e.chunk_id,
                                    doc_id=e.chunk.doc_id,
                                    locator=e.chunk.locator.model_dump_json(exclude_none=True),
                                    text=e.chunk.text,
                                    kb_version=e.kb_version,
                                    index_version=e.index_version,
                                )
                                for e in context.evidence
                            ]
                            if self.backend is not None:
                                generated = self.external(
                                    lambda: generate_for_service(
                                        self.backend_required(),
                                        request.message,
                                        lang,
                                        context,
                                        identify_language=lambda text: self.lid.detect(text).lang,
                                        verifier=self.verifier,
                                        code_switch=detected.code_switch,
                                    ),
                                    deadline,
                                )
                                diagnostic = ";".join(generated.problems) or "qualified_support"
                                if (
                                    not generated.fallback
                                    and generated.semantic_support == "supported"
                                    and generated.valid_until is not None
                                ):
                                    response = generated.response
                                    accepted_until = generated.valid_until
                                    session.failures = 0
                                else:
                                    reason = "unverified_generation"
                            else:
                                diagnostic = "generation_disabled:semantic_unknown"
                                reason = "unverified_generation"
                    elif route.action == Action.REFUSE_OOS:
                        response = AssistantResponse(
                            language=lang,
                            intent=intent,
                            action=Action.REFUSE_OOS,
                            answer={
                                "en": "I can help with store support.",
                                "ru": "Я могу помочь с поддержкой магазина.",
                                "kk": "Дүкен қызметі бойынша көмектесе аламын.",
                            }[lang],
                        )
                    else:
                        if self.taxonomy.intents[intent].tool:
                            session.pending = intent
                        response = self.clarify(lang, "reference" if session.pending else "clarify")
                except Exception as exc:
                    reason = "component_failure"
                    diagnostic = (
                        "component_timeout"
                        if isinstance(exc, TimeoutError)
                        else "component_unavailable"
                    )
        escalation = None
        with self.store.transaction() as db:
            if accepted_until is not None and (
                datetime.now(UTC) >= accepted_until or time.monotonic() >= deadline
            ):
                reason, diagnostic = "unverified_generation", "expired_verification"
            if tool and ref and not reason:
                try:
                    values = self.store.tool(db, customer, tool, ref)
                    response = AssistantResponse(
                        language=lang,
                        intent=intent,
                        action=Action.CALL_TOOL,
                        answer=_COPY["tool"][lang]
                        + ": "
                        + "; ".join(f"{k}={v}" for k, v in values.items()),
                        facts_from_tools=list(values),
                    )
                    if tool == "get_stock":
                        session.sku = ref
                    else:
                        session.order = ref
                    session.pending, session.failures = None, 0
                except ServiceError as exc:
                    # A failed/foreign reference is never carried forward or revealed.
                    reason, diagnostic = "tool_unavailable", exc.code
            if reason:
                # Persist only structured allowlisted context, never raw messages/model output.
                context_lines = [*session.history, f"intent={intent};reason={reason}"]
                if session.order or session.sku:
                    context_lines.append(
                        f"authorized_order={session.order};authorized_sku={session.sku}"
                    )
                if reason == "independent_safety_guard":
                    cues = [
                        word
                        for word in (
                            "operator",
                            "оператор",
                            "swollen",
                            "smoke",
                            "injury",
                            "ожог",
                            "дым",
                            "түтін",
                            "өрт",
                            "fraud",
                            "мошенн",
                            "алаяқ",
                            "password",
                            "парол",
                            "құпиясөз",
                            "privacy",
                            "персональн",
                            "жеке дерек",
                            "court",
                            "суд",
                            "сот",
                            "жалоб",
                            "complaint",
                            "шағым",
                            "charged twice",
                            "дважды",
                            "екі рет",
                        )
                        if word in request.message.lower()
                    ]
                    context_lines.append("guard_cues=" + ",".join(cues))
                escalation = self.store.enqueue(
                    db,
                    customer,
                    identity,
                    reason,
                    lang,
                    context_lines[-self.settings.memory_turns :],
                    [s.chunk_id for s in sources],
                )
                # Reused tickets may retain a higher-priority reason. Acknowledge the
                # actual committed queue state, not the lower-priority followup reason.
                reason = str(
                    db.execute(
                        "SELECT reason FROM escalations WHERE id=?", (str(escalation),)
                    ).fetchone()[0]
                )
                response = safe_fallback(lang).model_copy(
                    update={
                        "answer": _COPY["queued"][lang],
                        "escalation_reason": reason,
                        "intent": intent,
                    }
                )
                session.failures += 1
            elif response.action == Action.CLARIFY:
                session.failures += 1
            delivery = (
                "queued"
                if escalation
                else "tool"
                if response.action == Action.CALL_TOOL
                else "clarification"
                if response.action == Action.CLARIFY
                else "grounded"
                if response.action == Action.ANSWER_KB
                else "refusal"
            )
            result = ChatResult(
                session_id=identity,
                request_id=request.request_id,
                response=response,
                sources=sources,
                escalation_id=escalation,
                code_switch=detected.code_switch,
                delivery=delivery,
                diagnostic=diagnostic,
            )
            db.execute(
                "INSERT INTO receipts VALUES(?,?,?,?)",
                (customer, str(request.request_id), fingerprint, result.model_dump_json()),
            )
        session.history = [
            *session.history,
            f"intent={intent};delivery={delivery};reason={reason}",
        ][-self.settings.memory_turns :]
        session.updated = time.monotonic()
        self.sessions[identity] = session
        return result

    def clarify(self, language: LanguageCode, key: str) -> AssistantResponse:
        return AssistantResponse(
            language=language,
            intent="unclear",
            action=Action.CLARIFY,
            answer=_COPY[key][language],
            clarification_question=_COPY[key][language],
        )

    def route(
        self,
        text: str,
        lang: LanguageCode,
        mixed: bool,
        slots: dict[str, str] | None = None,
    ) -> RouteDecision:
        references = dict(slots or {})
        intent = explicit_intent(text)
        confidence = None
        if intent is None:
            if self.router is None:
                raise ServiceError(503, "router_unavailable")
            raw = self.router.route(text, references, language=lang, code_switch=mixed)
            intent, confidence = raw.intent, raw.confidence
        return service_decision(intent, text, references, lang, mixed, self.taxonomy, confidence)

    def context(self, text: str, lang: LanguageCode) -> RetrievalContext:
        if self.retriever is None:
            raise ServiceError(503, "retrieval_unavailable")
        return self.retriever.context(text, lang)

    def backend_required(self) -> Backend:
        if self.backend is None:
            raise ServiceError(503, "generation_unavailable")
        return self.backend


class ClarificationRouter:
    """Explicit limited mode when no frozen encoder can be loaded; never guesses facts."""

    def __init__(self, taxonomy: Taxonomy) -> None:
        self.taxonomy = taxonomy

    def route(
        self,
        message: str,
        session: dict[str, str],
        *,
        language: LanguageCode,
        code_switch: bool = False,
    ) -> RouteDecision:
        return policy_decision("unclear", 0, message, session, language, code_switch, self.taxonomy)
