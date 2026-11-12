"""Deterministic grounding checks for an AssistantResponse (DATA_CONTRACTS §6, EVALUATION §3).

Used by the SFT builder and gates (Stage 4) and, later, by the online validators (Stage 8):
- citations are a subset of the chunk ids that were in the context;
- every number in the answer (amounts, periods, dates, ids) occurs in the cited chunks, the
  tool results or the user's own messages;
- `facts_from_tools` values occur in a tool result;
- an answer_kb response either cites evidence or declares missing information (abstains).
Language is checked by the caller (it needs the language identifier).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from supportpilot.common.schemas import Action, AssistantResponse

_GROUPED_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ,]\d{3})+(?![\d,])")
_DIGITS_RE = re.compile(r"\d+")


def numbers_in(text: str) -> set[str]:
    """Numbers as digit strings: grouped amounts collapse ("15 000" / "15,000" -> "15000"),
    and every plain digit run counts on its own ("2026-11-14" -> 2026, 11, 14)."""
    out: set[str] = set()
    rest = text
    for m in _GROUPED_RE.finditer(text):
        out.add(re.sub(r"\D", "", m.group(0)))
    rest = _GROUPED_RE.sub(" ", rest)
    for d in _DIGITS_RE.findall(rest):
        out.add(d.lstrip("0") or "0")
    return out


def _evidence_numbers(texts: Iterable[str]) -> set[str]:
    nums: set[str] = set()
    for t in texts:
        nums |= numbers_in(t)
        # a grouped amount in the answer may be written plainly in JSON evidence and vice versa
        nums |= {d.lstrip("0") or "0" for d in _DIGITS_RE.findall(t)}
    return nums


@dataclass
class Evidence:
    chunks: Mapping[str, str]  # chunk_id -> text (everything shown in the context)
    tool_results: Sequence[Mapping[str, Any]] = field(default_factory=list)
    user_texts: Sequence[str] = field(default_factory=list)


def check_response(resp: AssistantResponse, ev: Evidence) -> list[str]:
    """Problems found (empty list = grounded)."""
    problems: list[str] = []
    unknown = [c for c in resp.citations if c not in ev.chunks]
    if unknown:
        problems.append(f"citation_not_in_context:{len(unknown)}")
    cited = [ev.chunks[c] for c in resp.citations if c in ev.chunks]
    tools = [json.dumps(t, ensure_ascii=False) for t in ev.tool_results]
    allowed = _evidence_numbers([*cited, *tools, *ev.user_texts])
    text = " ".join(x for x in (resp.answer, resp.clarification_question or "") if x)
    missing = sorted(numbers_in(text) - allowed)
    if missing:
        problems.append("unsupported_number:" + ",".join(missing[:5]))
    for fact in resp.facts_from_tools:
        value = fact.split("=", 1)[-1].strip()
        if not any(value in t for t in tools):
            problems.append(f"tool_fact_not_in_results:{fact}")
    if resp.facts_from_tools and not ev.tool_results:
        problems.append("tool_facts_without_tool_call")
    if (
        resp.action is Action.ANSWER_KB
        and not resp.citations
        and not resp.missing_information
        and not resp.needs_human
    ):
        problems.append("answer_without_evidence")
    return problems
