"""Sufficiency diagnostics: relevance and answerability have different denominators."""

from __future__ import annotations

from typing import Any

from supportpilot.retrieval.evaluate import clustered_ci, query_metrics


def evidence_metrics(
    ranking: list[str], packed: list[str], groups: list[list[str]]
) -> dict[str, float]:
    """OR within a support group, AND across groups; never turn background into proof."""
    if not groups or any(not group for group in groups):
        raise ValueError("answerable needs require nonempty evidence groups")
    if len(ranking) != len(set(ranking)) or len(packed) != len(set(packed)):
        raise ValueError("duplicate citation IDs")
    if not set(packed) <= set(ranking):
        raise ValueError("packed evidence must come from the ranking")

    def coverage(ids: list[str]) -> float:
        selected = set(ids)
        return sum(bool(selected & set(group)) for group in groups) / len(groups)

    at5, at10, context = coverage(ranking[:5]), coverage(ranking[:10]), coverage(packed)
    return {
        "full_evidence5": float(at5 == 1),
        "full_evidence10": float(at10 == 1),
        "packed_coverage": context,
        "packed_sufficiency": float(context == 1),
    }


def measure_query(
    ranking: list[str],
    packed: list[str],
    grades: dict[str, int],
    groups: list[list[str]],
    *,
    answerable: bool,
) -> dict[str, float]:
    """Abstention measures empty packed context, not empty candidate generation."""
    if len(ranking) != len(set(ranking)) or len(packed) != len(set(packed)):
        raise ValueError("duplicate citation IDs")
    if not set(packed) <= set(ranking):
        raise ValueError("packed evidence must come from the ranking")
    returned = bool(packed)
    result = {"abstention_accuracy": float(returned == answerable)}
    if answerable:
        result.update(query_metrics(ranking, grades))
        result.update(evidence_metrics(ranking, packed, groups))
        result["answerable_fnr"] = float(not returned)
    else:
        if groups or any(grade > 0 for grade in grades.values()):
            raise ValueError("unanswerable need cannot have positive proof")
        result["unanswerable_fpr"] = float(returned)
    return result


def summarise_scale(
    rows: list[dict[str, Any]], *, seed: int, samples: int = 1000
) -> dict[str, Any]:
    """Cluster every metric, including FPR/FNR; absent cohorts remain null."""
    metrics = (
        "recall5",
        "mrr10",
        "ndcg10",
        "full_evidence5",
        "full_evidence10",
        "packed_coverage",
        "packed_sufficiency",
        "answerable_fnr",
        "unanswerable_fpr",
        "abstention_accuracy",
    )
    result: dict[str, Any] = {}
    scopes = sorted({scope for row in rows for scope in row["slices"]})
    for scope in scopes:
        selected = [row for row in rows if scope in row["slices"]]
        result[scope] = {
            "queries": len(selected),
            "scenarios": len({row["scenario_id"] for row in selected}),
            "families": len({row["family_id"] for row in selected}),
            "metrics": {
                metric: clustered_ci(
                    [row for row in selected if metric in row], metric, seed=seed, samples=samples
                )
                for metric in metrics
            },
            "family_sensitivity": {
                metric: clustered_ci(
                    [{**row, "scenario_id": row["family_id"]} for row in selected if metric in row],
                    metric,
                    seed=seed,
                    samples=samples,
                )
                for metric in metrics
            },
        }
    return result


def paired_scale(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    """Require identical phases/families and pair by query; never pair unrelated corpora."""
    left = {row["query_id"]: row for row in control}
    right = {row["query_id"]: row for row in candidate}
    if len(left) != len(control) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("paired comparison requires identical unique query IDs")
    rows = []
    for qid, row in left.items():
        other = right[qid]
        for key in (
            "scenario_id",
            "family_id",
            "phase",
            "answerable",
            "slices",
            "kb_version",
            "benchmark_version",
        ):
            if row.get(key) != other.get(key):
                raise ValueError("paired comparison populations disagree")
        metrics = {
            key: other[key] - value
            for key, value in row.items()
            if isinstance(value, float) and key in other
        }
        rows.append({**row, **metrics})
    return summarise_scale(rows, seed=seed)
