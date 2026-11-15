"""Exercise model-to-service action integration and authority at the HTTP boundary."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import ServingSettings, Settings
from supportpilot.common.schemas import Action, LanguageCode
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.routing.models import RouteDecision
from supportpilot.service.orchestrator import SupportService
from supportpilot.service.store import Store

ROOT = Path(__file__).resolve().parents[2]


class FixedRouter:
    def __init__(self, intent: str = "product_defect", action: Action = Action.CALL_TOOL) -> None:
        self.intent, self.action = intent, action
        self.calls: list[dict[str, str]] = []

    def route(
        self,
        message: str,
        session: dict[str, str],
        *,
        language: LanguageCode,
        code_switch: bool = False,
    ) -> RouteDecision:
        self.calls.append(dict(session))
        return RouteDecision(
            language=language, intent=self.intent, action=self.action, slots={"order_id": "SP-1002"}
        )


@pytest.fixture
def runtime(tmp_path: Path) -> Iterator[tuple[TestClient, SupportService, FixedRouter]]:
    cfg = ServingSettings(
        database=tmp_path / "state.sqlite", tokens={"demo_alice": SecretStr("a" * 32)}
    )
    store = Store(cfg.database)
    store.seed_demo()
    router = FixedRouter()
    service = SupportService(store, cfg, load_taxonomy(ROOT / "configs/taxonomy.yaml"), router)
    with TestClient(create_app(Settings(root=ROOT, serving=cfg), service)) as client:
        client.headers["Authorization"] = "Bearer " + "a" * 32
        yield client, service, router
    service.close()


def request(client: TestClient, message: str, **kwargs: Any) -> dict[str, Any]:
    response = client.post(
        "/chat", json={"message": message, "language": "en", "request_id": str(uuid4()), **kwargs}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_learned_tool_decision_executes_and_passes_only_authorized_memory(
    runtime: tuple[TestClient, SupportService, FixedRouter],
) -> None:
    client, service, router = runtime
    first = request(client, "Inspect SP-1001; the display is blank.")
    assert first["delivery"] == "tool"
    assert "SP-1001" in first["response"]["answer"]
    assert "SP-1002" not in first["response"]["answer"]
    router.intent = "payment_problem"
    second = request(client, "What about that transaction?", session_id=first["session_id"])
    assert router.calls[-1] == {"order_id": "SP-1001"}
    assert second["delivery"] == "tool"
    assert "payment_status=paid" in second["response"]["answer"]
    assert len(service.sessions) == 1


def test_reference_followup_uses_pending_intent_without_learning_predicted_slots(
    runtime: tuple[TestClient, SupportService, FixedRouter],
) -> None:
    client, _service, router = runtime
    router.intent = "invoice"
    first = request(client, "Please inspect the fiscal document.")
    assert first["delivery"] == "clarification"
    second = request(client, "SP-1001", session_id=first["session_id"])
    assert second["delivery"] == "tool"
    assert "payment_status" in second["response"]["answer"]


@pytest.mark.parametrize(
    "message",
    [
        "Do not cancel SP-1001",
        "Never cancel SP-1001",
        "Cancel SP-1001 and check payment",
        "How do I cancel SP-1001?",
        "Inspect SP-1001",  # model cancellation label cannot authorize mutation
    ],
)
def test_confirmation_does_not_authorize_negated_policy_or_inferred_cancellation(
    runtime: tuple[TestClient, SupportService, FixedRouter],
    message: str,
) -> None:
    client, service, router = runtime
    router.intent = "cancel_order"
    result = request(client, message, confirm_cancellation=True)
    assert result["delivery"] != "tool"
    with service.store.transaction() as db:
        assert (
            db.execute("SELECT cancellation_requested FROM orders WHERE id='SP-1001'").fetchone()[0]
            == 0
        )


@pytest.mark.parametrize(
    "message",
    [
        "Stock TV-5501 again",
        "Stock TV-5501 for court games",
        "Stock TV-5501, a smoke detector",
        "Наличие TV-5501 повторно",
    ],
)
def test_ordinary_product_and_retry_words_do_not_trigger_critical_queue(
    runtime: tuple[TestClient, SupportService, FixedRouter],
    message: str,
) -> None:
    assert request(runtime[0], message)["delivery"] == "tool"


def test_policy_question_does_not_become_stock_lookup(
    runtime: tuple[TestClient, SupportService, FixedRouter],
) -> None:
    client, _service, router = runtime
    router.intent = "availability"
    result = request(client, "What are the stock rules for TV-5501?")
    assert result["delivery"] == "queued"
    assert result["response"]["facts_from_tools"] == []
    assert result["response"]["escalation_reason"] == "evidence_unavailable"


def test_fraud_queue_reason_survives_semantic_fallback(
    runtime: tuple[TestClient, SupportService, FixedRouter],
) -> None:
    client, _service, router = runtime
    first = request(client, "My account was hacked")
    router.intent, router.action = "account_help", Action.ANSWER_KB
    second = request(client, "Explain profile settings", session_id=first["session_id"])
    assert second["escalation_id"] == first["escalation_id"]
    assert client.get("/escalations").json()[0]["reason"] == "fraud_suspected"


def test_wrong_reference_type_never_reuses_a_carried_order_to_cancel(
    runtime: tuple[TestClient, SupportService, FixedRouter],
) -> None:
    client, service, _router = runtime
    first = request(client, "Track SP-1001")
    result = request(
        client, "Cancel TV-5501", session_id=first["session_id"], confirm_cancellation=True
    )
    assert result["delivery"] == "clarification"
    with service.store.transaction() as db:
        assert (
            db.execute("SELECT cancellation_requested FROM orders WHERE id='SP-1001'").fetchone()[0]
            == 0
        )
