"""Versioned LLM-authored safety cases; never mixed into optimization data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.schemas import LANGUAGES, Action, AssistantResponse


def build_safety(source: Path, destination: Path) -> list[dict[str, Any]]:
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    cases = document["cases"]
    if len({case["id"] for case in cases}) != len(cases) or len(cases) < 50:
        raise ValueError("need at least 50 unique authored safety needs")
    rows = []
    for case in cases:
        for lang in LANGUAGES:
            action = Action(case["action"])
            response = AssistantResponse(
                language=lang,
                intent=case["intent"],
                action=action,
                answer="Evaluation label only; never supplied to the model.",
                needs_human=action is Action.ESCALATE,
                escalation_reason=case.get("reason"),
                clarification_question="?" if action is Action.CLARIFY else None,
            )
            scenario = "safety_" + case["id"]
            rows.append(
                {
                    "example_id": sha256_text(scenario + "|" + lang)[:16],
                    "scenario_id": scenario,
                    "family_id": "safety_" + case["category"],
                    "lang": lang,
                    "code_switch": False,
                    "task": case["category"],
                    "user_text": case[lang],
                    "split": "safety",
                    "review_status": "draft",
                    "target": response.model_dump(mode="json"),
                    "messages": [
                        {"role": "system", "content": "Store support safety evaluation."},
                        {"role": "user", "content": case[lang]},
                        {"role": "assistant", "content": response.model_dump_json()},
                    ],
                }
            )
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "safety.json", rows)
    write_json(
        destination / "manifest.json",
        {
            "schema_version": "llm_safety/1",
            "source_sha256": sha256_file(source),
            "data_sha256": sha256_file(destination / "safety.json"),
            "needs": len(cases),
            "rows": len(rows),
            "languages": list(LANGUAGES),
            "review_status": "draft",
            "independent_fluent_review": False,
            "categories": sorted({case["category"] for case in cases}),
        },
    )
    return rows


def context_variant(
    row: dict[str, Any],
    mode: str,
    *,
    required: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Controlled removal of explicit proof; topical proximity is never a positive label.

    This transformation changes context only, never the user question. It supplies an
    evaluation sufficiency label, not an oracle to the serving generation boundary.
    """
    result: dict[str, Any] = json.loads(json.dumps(row))
    messages = result["messages"]
    kb = [m for m in messages[:-1] if m.get("name") == "kb_search"]
    if len(kb) != 1:
        raise ValueError("context probe needs exactly one kb_search message")
    payload = json.loads(kb[0]["content"])
    cited = set(row["target"]["citations"])
    if not cited:
        raise ValueError("context probe needs a known complete cited answer")
    if mode == "absent":
        payload["results"] = []
    elif mode == "irrelevant":
        payload["results"] = [r for r in payload["results"] if r["chunk_id"] not in cited]
    elif mode == "partial":
        if len(cited) < 2:
            raise ValueError("partial probe requires multiple cited fragments")
        removed = sorted(cited)[-1]
        payload["results"] = [r for r in payload["results"] if r["chunk_id"] != removed]
    elif mode != "complete":
        raise ValueError("unsupported context intervention")
    kb[0]["content"] = json.dumps(payload, ensure_ascii=False)
    result["task"] = mode
    result["example_id"] = sha256_text(row["example_id"] + "|" + mode)[:16]
    result["parent_example_id"] = row["example_id"]
    result["required_support_groups"] = required or [[c] for c in sorted(cited)]
    packed = {r["chunk_id"] for r in payload["results"]}
    result["packed_sufficient"] = all(
        packed.intersection(group) for group in result["required_support_groups"]
    )
    return result
