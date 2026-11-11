from __future__ import annotations

import io
import json
import logging

import pytest

from supportpilot.common.config import LoggingSettings
from supportpilot.common.logging import REDACTED, configure_logging, redact, scrub


@pytest.fixture
def capture() -> io.StringIO:
    buf = io.StringIO()
    configure_logging(LoggingSettings(level="DEBUG", format="json"), stream=buf)
    return buf


def _last(buf: io.StringIO) -> dict[str, object]:
    return json.loads(buf.getvalue().strip().splitlines()[-1])  # type: ignore[no-any-return]


def test_json_line_with_extras(capture: io.StringIO) -> None:
    logging.getLogger("t").info("routed %s", "ok", extra={"intent": "track_order", "k": 3})
    rec = _last(capture)
    assert rec["msg"] == "routed ok"
    assert rec["level"] == "INFO"
    assert rec["intent"] == "track_order"
    assert rec["k"] == 3


def test_sensitive_keys_redacted(capture: io.StringIO) -> None:
    logging.getLogger("t").info(
        "auth", extra={"api_key": "abc", "headers": {"Authorization": "Bearer x", "ok": 1}}
    )
    rec = _last(capture)
    assert rec["api_key"] == REDACTED
    assert rec["headers"] == {"Authorization": REDACTED, "ok": 1}


def test_customer_text_hashed_by_default(capture: io.StringIO) -> None:
    logging.getLogger("t").info("chat", extra={"text": "Менің тапсырысым қайда?"})
    rec = _last(capture)
    assert rec["text"] == {"len": 23, "sha256_8": rec["text"]["sha256_8"]}  # type: ignore[index]


def test_customer_text_kept_when_enabled() -> None:
    buf = io.StringIO()
    configure_logging(LoggingSettings(format="json", log_text=True), stream=buf)
    logging.getLogger("t").info("chat", extra={"text": "где заказ?"})
    assert _last(buf)["text"] == "где заказ?"


@pytest.mark.parametrize(
    ("raw", "hidden"),
    [
        ("token hf_AbCdEfGhIjKlMnOp", "hf_AbCdEfGhIjKlMnOp"),
        ("gh gho_1234567890abcdef", "gho_1234567890abcdef"),
        ("key sk-proj-1234567890abcdefghij", "sk-proj-1234567890abcdefghij"),
        ("Authorization: Bearer eyJhbGciOi.x.y", "eyJhbGciOi"),
        ("mail aigerim.k@example.kz now", "aigerim.k@example.kz"),
        ("call +7 701 234 56 78", "701 234 56 78"),
        ("call 8(727)2345678", "2345678"),
        ("card 4111111111111111", "4111111111111111"),
        ("IIN 900101300123", "900101300123"),
    ],
)
def test_scrub_masks_secrets_and_pii(raw: str, hidden: str) -> None:
    assert hidden not in scrub(raw)


@pytest.mark.parametrize(
    "keep",
    ["2026-11-11 10:43:35", "order 100234 shipped", "price 12 990 KZT", "SKU-77812"],
)
def test_scrub_keeps_ordinary_values(keep: str) -> None:
    assert scrub(keep) == keep


def test_exception_text_scrubbed(capture: io.StringIO) -> None:
    try:
        raise RuntimeError("leaked hf_SECRETSECRET123")
    except RuntimeError:
        logging.getLogger("t").exception("boom")
    rec = _last(capture)
    assert "hf_SECRETSECRET123" not in json.dumps(rec)
    assert "RuntimeError" in str(rec["exc"])


def test_configure_is_idempotent() -> None:
    for _ in range(3):
        configure_logging(LoggingSettings(), stream=io.StringIO())
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_supportpilot_handler", False)]
    assert len(ours) == 1


def test_redact_nested_lists() -> None:
    out = redact({"items": [{"password": "x"}, "a@b.co"]}, log_text=False)
    assert out == {"items": [{"password": REDACTED}, "<email>"]}


def test_console_format(capture: io.StringIO) -> None:
    buf = io.StringIO()
    configure_logging(LoggingSettings(format="console"), stream=buf)
    logging.getLogger("t").warning("hi", extra={"token": "zzz"})
    line = buf.getvalue()
    assert "WARNING" in line
    assert "zzz" not in line


def test_default_handler_follows_current_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(LoggingSettings(format="json"))
    logging.getLogger("t").warning("after swap")
    assert "after swap" in capsys.readouterr().err
    assert logging.getLogger("httpx").level == logging.WARNING
