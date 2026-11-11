"""Structured logging (stdlib) with redaction of secrets and PII.

- Values under sensitive keys (password, token, api_key, ...) become `***`.
- Secret-looking tokens, e-mails, KZ/RU phone numbers and long digit runs (cards, IIN) are
  masked in every string, including the message and tracebacks.
- Customer/model text under TEXT_KEYS is replaced by `{len, sha256_8}` unless `log_text=True`.

Usage: `configure_logging(settings.logging)` once per process, then
`log.info("routed", extra={"intent": "track_order", "text": user_message})`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from supportpilot.common.config import LoggingSettings
from supportpilot.common.hashing import sha256_text

REDACTED = "***"
SENSITIVE_KEY_RE = re.compile(
    r"pass(word)?|passwd|secret|token|api[_-]?key|authorization|cookie|credential|private[_-]?key",
    re.IGNORECASE,
)
TEXT_KEYS = frozenset({"text", "user_text", "message_text", "prompt", "answer", "response"})

_SCRUB_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE), "Bearer <token>"),
    (re.compile(r"\b(?:hf|gh[pousr]|github_pat|xox[abprs])_[A-Za-z0-9_]{8,}"), "<token>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "<token>"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "<email>"),
    (
        re.compile(r"(?<![\w+])\+?[78][\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}(?!\d)"),
        "<phone>",
    ),
    (re.compile(r"(?<!\d)\d{12,19}(?!\d)"), "<number>"),
)

_STD_ATTRS = frozenset(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}
_OUR_HANDLER_FLAG = "_supportpilot_handler"


def scrub(text: str) -> str:
    for pattern, repl in _SCRUB_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def redact(value: Any, *, log_text: bool, key: str | None = None) -> Any:
    """Recursively redact a structured log value."""
    if key is not None and SENSITIVE_KEY_RE.search(key):
        return REDACTED
    if key is not None and key in TEXT_KEYS and isinstance(value, str) and not log_text:
        return {"len": len(value), "sha256_8": sha256_text(value)[:8]}
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, Mapping):
        return {str(k): redact(v, log_text=log_text, key=str(k)) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact(v, log_text=log_text) for v in value]
    if value is None or isinstance(value, bool | int | float):
        return value
    return scrub(str(value))


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _STD_ATTRS and not k.startswith("_")}


class JsonFormatter(logging.Formatter):
    def __init__(self, *, log_text: bool) -> None:
        super().__init__()
        self.log_text = log_text

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": scrub(record.getMessage()),
        }
        payload.update(redact(_extras(record), log_text=self.log_text))
        if record.exc_info:
            payload["exc"] = scrub(self.formatException(record.exc_info))
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self, *, log_text: bool) -> None:
        super().__init__()
        self.log_text = log_text

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=UTC).strftime("%H:%M:%S")
        line = f"{ts} {record.levelname:<7} {record.name}: {scrub(record.getMessage())}"
        extras = redact(_extras(record), log_text=self.log_text)
        if extras:
            line += " " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + scrub(self.formatException(record.exc_info))
        return line


class _CurrentStderrHandler(logging.StreamHandler):  # type: ignore[type-arg]
    """Writes to whatever `sys.stderr` is *now* (survives stream swaps by test runners)."""

    @property
    def stream(self) -> Any:
        return sys.stderr

    @stream.setter
    def stream(self, _value: Any) -> None:
        pass


# Third-party loggers that are chatty at INFO (one line per HTTP request).
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "alembic")


def configure_logging(settings: LoggingSettings, stream: Any = None) -> None:
    """Install one SupportPilot handler on the root logger (idempotent; logs go to stderr)."""
    root = logging.getLogger()
    for old in list(root.handlers):
        if getattr(old, _OUR_HANDLER_FLAG, False):
            root.removeHandler(old)
    handler: logging.Handler = (
        logging.StreamHandler(stream) if stream is not None else _CurrentStderrHandler()
    )
    formatter_cls = JsonFormatter if settings.format == "json" else ConsoleFormatter
    handler.setFormatter(formatter_cls(log_text=settings.log_text))
    setattr(handler, _OUR_HANDLER_FLAG, True)
    root.addHandler(handler)
    root.setLevel(settings.level)
    third_party = logging.DEBUG if settings.level == "DEBUG" else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
