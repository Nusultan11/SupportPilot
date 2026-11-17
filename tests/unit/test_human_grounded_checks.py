"""Structural and novelty checks for grounded doc2dial targets (ADR-045)."""

from __future__ import annotations

import pytest

from supportpilot.data.humangrounded import novel_tokens, structural_problem

EVIDENCE = "You can renew your licence online if it expires within 60 days."


@pytest.mark.parametrize(
    ("target", "behaviour", "evidence", "problem"),
    [
        ("You can renew it online [3] within 60 days.", "answer", EVIDENCE, "markup_artifact"),
        ("You can renew it onlineThe fee is waived.", "answer", EVIDENCE, "fused_words"),
        ("Renew online.", "answer", EVIDENCE, "fragment"),
        ("Tell me whether your licence expired.", "clarify", EVIDENCE, "clarify_not_question"),
        ("Can you renew it online within 60 days?", "answer", EVIDENCE, "answer_is_question"),
        ("Yes, you can renew it online in time.", "answer", EVIDENCE, "unsupported_polarity"),
        ("Please see the renewal section of the page.", "answer", EVIDENCE, "referral_only"),
    ],
)
def test_structural_problem_flags_each_defect(
    target: str, behaviour: str, evidence: str, problem: str
) -> None:
    assert structural_problem(target, "how do I renew?", behaviour, evidence) == problem


def test_structural_problem_accepts_clean_targets() -> None:
    assert (
        structural_problem(
            "You can renew it online if it expires within 60 days.", "how?", "answer", EVIDENCE
        )
        is None
    )
    assert (
        structural_problem(
            "Has your licence expired more than 60 days ago?", "how?", "clarify", EVIDENCE
        )
        is None
    )
    no_ev = "No, you cannot renew a licence that expired over a year ago."
    assert structural_problem(no_ev, "can I renew?", "answer", "You cannot renew it.") is None


def test_novel_tokens_ignore_evidence_dialogue_and_inflection() -> None:
    target = "You can renew licences online; the fee is waived for veterans."
    assert novel_tokens(target, [EVIDENCE], ["is there a fee?"]) == ["waived", "veterans"]
    assert novel_tokens("Renewing online works.", [EVIDENCE], []) == ["works"]
