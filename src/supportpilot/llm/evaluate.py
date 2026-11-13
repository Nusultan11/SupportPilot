"""Raw-output metrics, strict diagnostics, explicit semantic UNKNOWN and clustered CIs."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
from pydantic import ValidationError

from supportpilot.common.grounding import Evidence, check_response
from supportpilot.common.schemas import INTENTS, AssistantResponse
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.sftgates import evidence_of


def score_output(row: dict[str, Any], raw: str | None, lid: LanguageIdentifier) -> dict[str, Any]:
    target = AssistantResponse.model_validate(row["target"])
    negative = row["task"] in {"abstain_no_evidence", "absent", "partial", "irrelevant"}
    positive = row["task"] in {"grounded_answer", "complete", "multi_document"}
    result: dict[str, Any] = {
        "example_id": row["example_id"],
        "scenario_id": row["scenario_id"],
        "family_id": row["family_id"],
        "lang": row["lang"],
        "code_switch": row.get("code_switch", False),
        "task": row["task"],
        "expected_intent": target.intent,
        "predicted_intent": "INVALID",
        "json_parseability": 0.0,
        "schema_validity": 0.0,
        "intent_accuracy": 0.0,
        "action_accuracy": 0.0,
        "language_compliance": 0.0,
        "escalation_accuracy": 0.0,
        "diagnostic_grounding": 0.0,
        "grounded_answer_accuracy": None,
        "citation_precision": None,
        "unsupported_claim_rate": None,
        "semantic_support": "NOT_EVALUATED",
        "raw": raw,
        "problems": [],
        "expected_escalation": target.needs_human,
        "predicted_escalation": False,
        "fallback_required": True,
    }
    if negative or positive:
        result["abstention_accuracy"] = 0.0
        result["delivered_abstention_accuracy"] = float(negative)
        if negative:
            result["negative_answer_attempt_rate"] = 1.0
        if positive:
            result["false_refusal_proxy"] = 1.0
    try:
        json.loads(raw or "")
        result["json_parseability"] = 1.0
        response = AssistantResponse.model_validate_json(raw or "", strict=True)
    except (ValueError, ValidationError):
        result["problems"] = ["invalid_schema"]
        return result
    evidence, _ = evidence_of(row["messages"][:-1])
    # Customer messages cannot authorize store facts; include only cited/tool evidence.
    strict = Evidence(chunks=evidence.chunks, tool_results=evidence.tool_results)
    problems = check_response(response, strict)
    result.update(
        {
            "schema_validity": 1.0,
            "predicted_intent": response.intent,
            "intent_accuracy": float(response.intent == target.intent),
            "action_accuracy": float(response.action == target.action),
            "language_compliance": float(
                response.language == target.language
                and lid.detect(response.answer).lang == target.language
            ),
            "escalation_accuracy": float(response.needs_human == target.needs_human),
            "predicted_escalation": response.needs_human,
            "diagnostic_grounding": float(not problems),
            "problems": problems,
            "citation_membership": float(set(response.citations) <= set(evidence.chunks)),
        }
    )
    abstained = (
        bool(response.missing_information or response.needs_human) and not response.citations
    )
    if negative or positive:
        result["abstention_accuracy"] = float(abstained == negative)
        if negative:
            # Behavioral attempt-to-answer proxy, not a semantic false-answer estimate.
            result["negative_answer_attempt_rate"] = float(not abstained)
        else:
            result["false_refusal_proxy"] = float(abstained)
    result["fallback_required"] = True  # semantic support remains unverified
    return result


def macro_f1(rows: list[dict[str, Any]]) -> float:
    if not rows:
        raise ValueError("no routing observations")
    scores = []
    for label in sorted(INTENTS):
        tp = sum(r["expected_intent"] == label == r["predicted_intent"] for r in rows)
        fp = sum(r["expected_intent"] != label == r["predicted_intent"] for r in rows)
        fn = sum(r["expected_intent"] == label != r["predicted_intent"] for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return float(np.mean(scores))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scopes = {lang: [r for r in rows if r["lang"] == lang] for lang in ("en", "ru", "kk")}
    scopes["ru_kk"] = [r for r in rows if r.get("code_switch")]
    result = {}
    names = (
        "json_parseability",
        "schema_validity",
        "intent_accuracy",
        "action_accuracy",
        "language_compliance",
        "escalation_accuracy",
        "diagnostic_grounding",
        "abstention_accuracy",
        "negative_answer_attempt_rate",
        "false_refusal_proxy",
    )
    for scope, selected in scopes.items():
        metrics = {}
        for name in names:
            values = [r[name] for r in selected if r.get(name) is not None]
            metrics[name] = {"n": len(values), "value": float(np.mean(values)) if values else None}
        timings = [r["seconds"] for r in selected if "seconds" in r]
        timed = [r for r in selected if r.get("seconds", 0) > 0]
        elapsed = sum(r["seconds"] for r in timed)
        resource_peaks = {
            name: max((r[name] for r in selected if r.get(name) is not None), default=None)
            for name in ("rss_bytes", "peak_vram_bytes", "reserved_vram_bytes")
        }
        escalation_rows = [
            r
            for r in selected
            if isinstance(r.get("expected_escalation"), bool)
            and isinstance(r.get("predicted_escalation"), bool)
        ]
        result[scope] = {
            "n": len(selected),
            "scenarios": len({r["scenario_id"] for r in selected}),
            "families": len({r["family_id"] for r in selected}),
            "metrics": metrics,
            "intent_macro_f1_35": macro_f1(selected) if selected else None,
            "intent_coverage": len({r["expected_intent"] for r in selected}),
            "tasks": dict(Counter(r["task"] for r in selected)),
            "latency_p50": float(np.percentile(timings, 50)) if timings else None,
            "latency_p95": float(np.percentile(timings, 95)) if timings else None,
            "resources": resource_peaks,
            "execution_errors": sum(bool(r.get("execution_error")) for r in selected),
            "escalation_observations": len(escalation_rows),
            "required_escalations": (
                sum(r["expected_escalation"] for r in escalation_rows) if escalation_rows else None
            ),
            "missed_required_escalations": sum(
                r["expected_escalation"] and not r["predicted_escalation"] for r in escalation_rows
            )
            if escalation_rows
            else None,
            "token_limit_reached": sum(bool(r.get("token_limit_reached")) for r in selected),
            "timed_observations": len(timed),
            "generation_tokens_per_second": (
                sum(r.get("output_tokens", 0) for r in timed) / elapsed if elapsed else None
            ),
            "throughput_scope": "timed model generation only; excludes loading and failures",
            "grounded_answer_accuracy": "NOT_EVALUATED",
            "citation_precision": "NOT_EVALUATED",
            "unsupported_claim_rate": "NOT_EVALUATED",
            "false_answer_rate": "NOT_EVALUATED",
        }
    return result


def paired_ci(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    metric: str,
    *,
    cluster: str = "scenario_id",
    seed: int = 20261113,
    samples: int = 1000,
) -> dict[str, Any]:
    left, right = ({r["example_id"]: r for r in rows} for rows in (control, candidate))
    if len(left) != len(control) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("paired CI needs unique identical example IDs")
    groups: dict[str, list[float]] = {}
    for key, a in left.items():
        b = right[key]
        if any(a[k] != b[k] for k in ("scenario_id", "family_id", "lang", "expected_intent")):
            raise ValueError("paired input identity mismatch")
        if a.get(metric) is None or b.get(metric) is None:
            continue
        groups.setdefault(a[cluster], []).append(float(b[metric]) - float(a[metric]))
    if not groups:
        return {"n": 0, "delta": None, "ci95": None}
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(samples, len(groups)))
    means = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {
        "n": int(counts.sum()),
        "clusters": len(groups),
        "delta": float(sums.sum() / counts.sum()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "between_family_uncertainty_identified": len(groups) > 1,
    }


def paired_macro_f1_ci(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    cluster: str = "scenario_id",
    seed: int = 20261113,
    samples: int = 1000,
) -> dict[str, Any]:
    """Recompute the nonlinear 35-class F1 for each paired cluster resample."""
    left = {r["example_id"]: r for r in control}
    right = {r["example_id"]: r for r in candidate}
    if len(left) != len(control) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("macro-F1 CI requires unique identical item IDs")
    if not left:
        return {"n": 0, "delta": None, "ci95": None}
    labels = sorted(INTENTS)
    groups = sorted({r[cluster] for r in control})
    table = {key: index for index, key in enumerate(groups)}
    counts = np.zeros((2, len(groups), 3, len(labels)), dtype=np.float64)
    for key, a in left.items():
        b = right[key]
        if any(a[k] != b[k] for k in ("scenario_id", "family_id", "lang", "expected_intent")):
            raise ValueError("paired macro-F1 identity mismatch")
        for model, row in enumerate((a, b)):
            group = table[row[cluster]]
            for label_index, label in enumerate(labels):
                expected = row["expected_intent"] == label
                predicted = row["predicted_intent"] == label
                counts[model, group, 0, label_index] += expected and predicted
                counts[model, group, 1, label_index] += not expected and predicted
                counts[model, group, 2, label_index] += expected and not predicted
    indices = np.random.default_rng(seed).integers(0, len(groups), size=(samples, len(groups)))
    f1 = []
    for matrix in counts:
        sampled = matrix[indices].sum(axis=1)
        denominator = 2 * sampled[:, 0] + sampled[:, 1] + sampled[:, 2]
        values = np.divide(
            2 * sampled[:, 0], denominator, out=np.zeros_like(denominator), where=denominator > 0
        )
        f1.append(values.mean(axis=1))
    return {
        "n": len(left),
        "clusters": len(groups),
        "delta": macro_f1(candidate) - macro_f1(control),
        "ci95": [float(v) for v in np.quantile(f1[1] - f1[0], [0.025, 0.975])],
        "between_family_uncertainty_identified": len(groups) > 1,
    }


def compare(control: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for language in ("en", "ru", "kk", "ru_kk"):

        def select(rows: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
            return (
                [r for r in rows if r.get("code_switch") or False]
                if scope == "ru_kk"
                else [r for r in rows if r["lang"] == scope]
            )

        a, b = select(control, language), select(candidate, language)
        report[language] = {}
        for cluster in ("scenario_id", "family_id"):
            report[language][cluster] = {
                metric: paired_ci(a, b, metric, cluster=cluster)
                for metric in (
                    "schema_validity",
                    "intent_accuracy",
                    "action_accuracy",
                    "language_compliance",
                    "diagnostic_grounding",
                    "escalation_accuracy",
                    "abstention_accuracy",
                )
            }
            report[language][cluster]["intent_macro_f1_35"] = paired_macro_f1_ci(
                a, b, cluster=cluster
            )
    return report
