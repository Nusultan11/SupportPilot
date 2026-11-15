"""Service-only interpretation; the sealed research router is never modified."""

from __future__ import annotations

import re

from supportpilot.common.schemas import Action, LanguageCode
from supportpilot.data.taxonomy import Taxonomy
from supportpilot.routing.models import RouteDecision


def matches(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.IGNORECASE) is not None


# Match an expressed incident/request, not an isolated product noun or retry word.
_CRITICAL: tuple[tuple[str, str, str], ...] = (
    (
        "safety_risk",
        "product_defect",
        r"\b(?:injur\w*|burned (?:my|me)|burnt (?:my|me)|electric shock|"
        r"swollen|overheat\w*|bleeding|on fire|caught fire|"
        r"обж[её]г\w*|ожог\w*|загорел\w*|горит|травм\w*|"
        r"ударило током|күйдір\w*|күйік\w*|ток соқты|жарақат\w*)\b|"
        r"(?:smoke|дым|түтін).{0,30}(?:coming|ид[её]т|шығ)|"
        r"(?:ид[её]т|выходит).{0,15}дым",
    ),
    (
        "fraud_suspected",
        "account_help",
        r"\b(?:fraud|stolen|hacked|taken over|мошенн\w*|взлом\w*|"
        r"алаяқ\w*|ұрлан\w*)\b|аккаунт\w*.{0,20}бұзыл|"
        r"(?:order|purchase).{0,35}(?:did not|didn't|never) (?:place|make|order)|"
        r"заказ.{0,30}(?:не оформлял|не делал)|мен рәсімдемеген тапсырыс",
    ),
    (
        "payment_dispute",
        "payment_problem",
        r"charged twice|charged.{0,15}(?:two times|double)|"
        r"(?:списал\w*|снял\w*).{0,20}дважды|дважды.{0,20}спис|"
        r"екі рет.{0,20}(?:алын|шешіл|төлен)|"
        r"money.{0,20}taken.{0,25}no order",
    ),
    (
        "privacy_request",
        "personal_data",
        r"personal data|персональн\w* данн\w*|жеке дерек\w*|"
        r"delete.{0,30}(?:account|profile)|удал\w*.{0,20}аккаунт|"
        r"аккаунт\w*.{0,20}(?:жой|өші)",
    ),
    (
        "legal_threat",
        "complaint",
        r"\b(?:sue|lawsuit|my lawyer|адвокат\w*|заңгер\w*|сотқа)\b|"
        r"(?:подам|обращ\w*|пойду).{0,15}(?:в суд|полици)|"
        r"(?:take|taking).{0,20}(?:to court|legal action)",
    ),
    (
        "repeated_failure",
        "unclear",
        r"(?:two|three|multiple|repeated).{0,30}(?:support|repl\w*|attempt\w*)"
        r".{0,45}(?:not solved|failed|unresolved)|"
        r"(?:два|три|повторн\w*).{0,35}(?:обращ\w*|ответ\w*).{0,45}не реш|"
        r"(?:екі|үш).{0,30}(?:өтініш|жауап).{0,35}шеш\w*(?:педі|лмеді)",
    ),
    (
        "customer_requested_human",
        "human_agent",
        r"\b(?:connect|transfer|speak|talk|need|want).{0,35}"
        r"(?:human|person|agent|supervisor|operator)\b|"
        r"^(?:operator|human agent)\b|\boperator please\b|"
        r"(?:соедин\w*|переключ\w*|нужен|позов\w*).{0,35}"
        r"(?:оператор\w*|менеджер\w*|человек\w*)|"
        r"^(?:оператор\w*|менеджер\w*)\b|"
        r"(?:адаммен|қызметкер\w*|оператор\w*).{0,25}(?:сөйлес|қос)",
    ),
    (
        "technical_issue",
        "technical_issue",
        r"(?:website|app|checkout|сайт|приложен\w*|страниц\w*|қолданба|рәсімдеу беті)"
        r".{0,50}(?:crash\w*|not work|blank page|вылет\w*|не работа\w*|пуст\w* страниц|"
        r"жабыл\w*|жұмыс істем\w*|бос бет)",
    ),
    ("complaint", "complaint", r"\b(?:complain\w*|жалоб\w*|пожалова\w*|шағым\w*)\b"),
)
_INJECTION = (
    r"system prompt|(?:ignore|disregard).{0,20}(?:instruction|rule)|"
    r"игнор\w*.{0,25}(?:инструкц|правил)|customer[_ -]?id|demo_bob|demo_alice"
)


def critical_request(text: str) -> tuple[str, str] | None:
    """Return reason and intent for explicitly covered critical requests only."""
    if matches(_INJECTION, text):
        return "independent_safety_guard", "unclear"
    return next(((reason, intent) for reason, intent, p in _CRITICAL if matches(p, text)), None)


_EXPLICIT: tuple[tuple[str, str], ...] = (
    ("cancel_order", r"\b(?:cancel\w*|отмен\w*|болдырма\w*|бас тарт\w*)"),
    ("invoice", r"\b(?:invoice|receipt|фискальн\w*|чек\w*|шот-фактура\w*|фискалдық)"),
    (
        "refund_status",
        r"(?:refund|возврат\w*|қайтарым\w*).{0,30}(?:status|статус|күй)|"
        r"(?:status|статус|күй).{0,30}(?:refund|возврат|қайтарым)",
    ),
    ("request_refund", r"money.back|возврат\w* денег|ақш\w*.{0,20}қайтар"),
    ("availability", r"\b(?:stock|availability|налич\w*|остат\w*|қор\w*|бар ма|қалған сан)"),
    ("payment_problem", r"\b(?:payment|оплат\w*|плат[её]ж\w*|төлем\w*)"),
)
_POLICY = (
    r"\b(?:polic\w*|methods|rules|правил\w*|способ\w*|услов\w*|"
    r"ереж\w*|әдіс\w*|шарт\w*)|"
    r"(?:how|как|қалай).{0,30}(?:cancel|отмен|болдырма)|"
    r"(?:instalment|рассрочк|бөліп төл|cash on|наличными|қолма-қол)"
)


def explicit_intent(text: str) -> str | None:
    """Narrow commands disambiguate operations; several operations require clarification."""
    if matches(
        r"(?:do not|don't|never).{0,15}cancel|не отмен\w*|отмен\w*.{0,15}не (?:делай|нужн)",
        text,
    ):
        return "unclear"
    if matches(_POLICY, text):
        return None
    found = {intent for intent, pattern in _EXPLICIT if matches(pattern, text)}
    # Refund/invoice vocabulary includes payment words; prefer the specific request.
    if found & {"invoice", "refund_status", "request_refund"}:
        found.discard("payment_problem")
    if "refund_status" in found:
        found.discard("request_refund")
    if len(found) > 1:
        return "unclear"
    if found:
        return next(iter(found))
    if matches(r"\bSP-\d{4}\b", text) and matches(
        r"\b(?:where|status|stage|hub|этап\w*|статус\w*|қайда|кезең\w*)", text
    ):
        return "track_order"
    if matches(r"\b(?!SP-)[A-Z]{2}-\d{4}\b", text) and matches(
        r"\b(?:ports?|dimensions?|compatib\w*|разъ[её]м\w*|габарит\w*|қосқыш\w*|өлшем\w*)", text
    ):
        return "product_info"
    if matches(r"\b(?:order|track\w*|отслед\w*|заказ\w*|тапсырыс\w*)", text) and not matches(
        r"how|как|қалай|change|измен|өзгерт|history|истор|бұрын|стар|old|"
        r"missing|wrong|не положили|басқа|жоқ|broke|broken|перестал|жұмыс істемей",
        text,
    ):
        return "track_order"
    return None


def service_decision(
    intent: str,
    text: str,
    slots: dict[str, str],
    language: LanguageCode,
    mixed: bool,
    taxonomy: Taxonomy,
    confidence: float | None = None,
) -> RouteDecision:
    """Reconcile a class with typed, authorized references; never trust predicted slots."""
    spec = taxonomy.intents[intent]
    reason = spec.escalation_reason if spec.action == Action.ESCALATE else None
    critical = critical_request(text)
    if critical:
        reason, intent = critical
        action = Action.ESCALATE
    elif spec.tool and matches(_POLICY, text):
        action = Action.ANSWER_KB
    elif spec.action == "needs_reference" or intent in {"payment_problem", "availability"}:
        reference = slots.get("sku" if spec.tool == "get_stock" else "order_id")
        action = Action.CALL_TOOL if reference else Action.CLARIFY
    elif intent == "product_info" and slots.get("sku"):
        action = Action.ANSWER_KB
    else:
        action = Action(spec.action)
    # A model's cancellation label alone is never authority to mutate state.
    if (
        intent == "cancel_order"
        and explicit_intent(text) != "cancel_order"
        and slots.get("pending_intent") != "cancel_order"
    ):
        action = Action.CLARIFY
    return RouteDecision(
        language=language,
        code_switch=mixed,
        intent=intent,
        action=action,
        confidence=confidence,
        slots=slots,
        escalate=bool(reason),
        escalation_reason=reason,
    )
