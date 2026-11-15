"""Capacity denominators must retain errors and use actual concurrent wall time."""

from supportpilot.service.workload import summarise


def test_expected_auth_rejections_are_not_business_success_throughput() -> None:
    rows = [
        {
            "status": 401,
            "kind": "authentication",
            "scope": "en",
            "outcome_correct": True,
            "seconds": 1.0,
        },
        {"status": 200, "kind": "stock", "scope": "en", "outcome_correct": True, "seconds": 1.0},
        {"status": 503, "kind": "kb", "scope": "kk", "outcome_correct": False, "seconds": 1.0},
        {
            "status": "transport_or_decode_error",
            "kind": "order",
            "scope": "mixed",
            "outcome_correct": False,
            "seconds": 1.0,
        },
        {"status": 200, "kind": "stock", "scope": "ru", "outcome_correct": False, "seconds": 1.0},
    ]
    result = summarise(rows, 2.0)
    assert result["attempts"] == 5
    assert result["correct_service_deliveries"] == 1
    assert result["http200"] == 2
    assert result["expected_auth_denials"] == 1
    assert result["incorrect_outcomes"] == 3
    assert result["successful_requests_per_second"] == 0.5
    assert result["attempts_per_second"] == 2.5
    assert result["all_attempts_latency"]["denominator"] == 5
