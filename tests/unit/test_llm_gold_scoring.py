import json
from types import SimpleNamespace
from typing import Any

from supportpilot.common.schemas import Action, AssistantResponse
from supportpilot.llm.evaluate import score_output


def abstainer() -> AssistantResponse:
    return AssistantResponse(
        language="en",
        intent="return_policy",
        action=Action.ESCALATE,
        answer="I cannot confirm the complete answer.",
        needs_human=True,
        escalation_reason="unverified_evidence",
        missing_information=["complete support"],
    )


def row(task: str, *, complete: bool | None = None) -> dict[str, Any]:
    target = (
        abstainer()
        if task == "abstain"
        else AssistantResponse(
            language="en",
            intent="return_policy",
            action=Action.ANSWER_KB,
            answer="Evaluation label only",
        )
    )
    passages = []
    if task == "grounded":
        passages.append(
            {
                "chunk_id": "returns:aaaaaaaa:0000",
                "text": "Returns within 14 days.",
                "effective_from": "2026-11-13",
                "effective_to": None,
            }
        )
        if complete:
            passages.append(
                {
                    "chunk_id": "delivery:bbbbbbbb:0000",
                    "text": "Delivery takes 2 days.",
                    "effective_from": "2026-11-13",
                    "effective_to": None,
                }
            )
    result: dict[str, Any] = {
        "example_id": "x",
        "scenario_id": "scenario",
        "family_id": "family",
        "lang": "en",
        "task": task,
        "target": target.model_dump(mode="json"),
        "messages": [
            {"role": "user", "content": "Return window and delivery time?"},
            {"role": "tool", "name": "kb_search", "content": json.dumps({"results": passages})},
            {"role": "assistant", "content": target.model_dump_json()},
        ],
    }
    if complete is not None:
        result["packed_sufficient"] = complete
        result["support_scope"] = "required_proof_groups"
    return result


def test_gold_abstain_alias_keeps_contract_failures_in_the_denominator() -> None:
    lid: Any = SimpleNamespace(detect=lambda text: SimpleNamespace(lang="en"))
    invalid = score_output(row("abstain"), "invalid", lid)
    assert invalid["abstention_accuracy"] == 0
    assert invalid["negative_answer_attempt_rate"] == 1
    valid = score_output(row("abstain"), abstainer().model_dump_json(), lid)
    assert valid["abstention_accuracy"] == 1
    assert valid["negative_answer_attempt_rate"] == 0
    assert valid["semantic_support"] == "NOT_EVALUATED"


def test_explicit_required_proof_loss_is_not_a_generator_false_refusal() -> None:
    lid: Any = SimpleNamespace(detect=lambda text: SimpleNamespace(lang="en"))
    missing = score_output(row("grounded", complete=False), abstainer().model_dump_json(), lid)
    assert missing["abstention_scope"] == "known_support_missing_from_packed_context"
    assert missing["abstention_accuracy"] == 1
    assert "false_refusal_proxy" not in missing
    complete = score_output(row("grounded", complete=True), abstainer().model_dump_json(), lid)
    assert complete["abstention_scope"] == "draft_task_label"
    assert complete["false_refusal_proxy"] == 1
    assert complete["abstention_accuracy"] == 0
    assert complete["grounded_answer_accuracy"] is None


def test_accepted_citation_groups_do_not_establish_context_sufficiency() -> None:
    lid: Any = SimpleNamespace(detect=lambda text: SimpleNamespace(lang="en"))
    reference_only = row("grounded", complete=False)
    reference_only["support_scope"] = "accepted_citation_sections_only"
    result = score_output(reference_only, abstainer().model_dump_json(), lid)
    assert result["abstention_scope"] == "draft_task_label"
    assert result["false_refusal_proxy"] == 1
    assert result["grounded_answer_accuracy"] is None
