from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

from supportpilot.service.manual_testing import InspectionRequest

ROOT = Path(__file__).resolve().parents[2]


class WorkbenchDouble:
    def __init__(self) -> None:
        self.requests: list[InspectionRequest] = []

    def inspect(self, request: InspectionRequest) -> dict[str, Any]:
        self.requests.append(request)
        return {
            "request": request.model_dump(),
            "attempt": str(len(self.requests)),
            "status": "COMPLETE",
            "raw": "<script>unmodified invalid raw</script>",
            "checks": {
                "candidate": None,
                "schema": "INVALID",
                "detected_reply_language": None,
                "deterministic_problems": ["invalid_schema"],
            },
            "context": {"kb_version": "kb", "index_version": "index", "evidence": []},
            "messages": [],
            "identity": {"selection_identity": "sealed"},
            "wall_seconds": 1.0,
        }


def test_ui_displays_raw_and_does_not_promote_or_call_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    double = WorkbenchDouble()
    monkeypatch.setattr("supportpilot.service.manual_testing.ManualWorkbench", lambda _: double)
    import streamlit as st

    get_option = st.get_option
    monkeypatch.setattr(
        st, "get_option", lambda name: "127.0.0.1" if name == "server.address" else get_option(name)
    )
    st.cache_resource.clear()
    try:
        app: Any = AppTest.from_file(str(ROOT / "apps/manual_testing.py")).run()
        assert not app.exception
        app.sidebar.selectbox[0].set_value("kk").run()
        app.chat_input[0].set_value("Где возврат? Қалай қайтаруға болады?").run(timeout=30)
        assert not app.exception
        assert double.requests[0].language == "kk"
        assert len(app.chat_message) == 2
        assert any("unmodified invalid raw" in x.value for x in app.text)
        assert any("NOT_EVALUATED" in x.value for x in app.warning)
        assert any("INVALID" in x.value for x in app.text)
        app.sidebar.button[0].click().run()
        assert not app.chat_message
        assert len(double.requests) == 1
    finally:
        st.cache_resource.clear()


def test_ui_refuses_nonloopback_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit as st

    get_option = st.get_option
    monkeypatch.setattr(
        st, "get_option", lambda name: "0.0.0.0" if name == "server.address" else get_option(name)
    )
    app: Any = AppTest.from_file(str(ROOT / "apps/manual_testing.py")).run()
    assert not app.exception
    assert app.error
    assert not app.chat_input
