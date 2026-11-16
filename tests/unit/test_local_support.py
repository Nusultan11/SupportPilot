"""Local judge diagnostic accounting must not fabricate independent semantics."""

from pathlib import Path

from supportpilot.service.local_support import cases, parse, summary


def test_controlled_rows_are_conditions_not_new_independent_needs() -> None:
    rows = cases(Path(__file__).resolve().parents[2] / "authored/system/local_support.yaml")
    assert len(rows) == 192
    assert len({r["need_id"] for r in rows}) == 8
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(not r["evidence"] for r in rows if r["condition"] == "absent")


def test_verdict_requires_exact_schema_and_no_timeout_or_truncation() -> None:
    assert parse('{"verdict":"supported"}', {}) == ("supported", None)
    for text in (
        '```json\n{"verdict":"supported"}\n```',
        '{"verdict":true}',
        '{"verdict":"supported","reason":"yes"}',
    ):
        assert parse(text, {}) == (None, "invalid_verdict_schema")
    assert parse('{"verdict":"supported"}', {"token_limit_reached": True}) == (
        None,
        "output_token_limit",
    )
    assert parse('{"verdict":"supported"}', {"seconds": 15.1}) == (None, "time_budget_exceeded")


def test_failed_judgements_stay_in_denominator_not_correct_unknown() -> None:
    rows = [
        {
            "scope": "kk",
            "need_id": "one",
            "expected": "unknown",
            "condition": "absent",
            "prediction": None,
        }
    ]
    result = summary(rows)["kk"]
    assert result["rows"] == 1
    assert result["fixture_correct"] == 0
    assert result["always_unknown_correct"] == 1
    assert result["invalid_or_failed"] == 1
    assert result["paired_accuracy_delta_need_ci95"] is None
    assert result["semantic_citation_precision"] is None
    assert result["independent_semantic_status"] == "NOT_EVALUATED"


def test_enum_contract_is_explicit_and_never_repairs_other_outputs() -> None:
    assert parse('"supported"', {}, verdict_format="enum") == ("supported", None)
    assert parse('"supported"', {}) == (None, "invalid_verdict_schema")
    for raw in ("supported", '{"verdict":"supported"}', '"Supported"', '```"supported"```'):
        assert parse(raw, {}, verdict_format="enum") == (None, "invalid_verdict_schema")
