"""Query-level confidence gating; never infer store facts or relabel evaluation evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.retrieval.models import ScoredChunk

THRESHOLDS = (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def evidence_gate(rows: list[ScoredChunk], threshold: float | None) -> list[ScoredChunk]:
    if threshold is None or not rows:
        return rows
    if any(row.rerank_score is None for row in rows):
        raise ValueError("query evidence threshold requires reranker scores")
    best = max(float(row.rerank_score or 0) for row in rows)
    return rows if best >= threshold else []


def balanced_accuracy(rows: list[dict[str, Any]]) -> float:
    groups: dict[tuple[bool, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["phase"] == "development":
            correct = (row["returned"] > 0) == row["answerable"]
            groups[(row["answerable"], row["scenario_id"])].append(float(correct))
    means = {
        label: [float(np.mean(v)) for (answerable, _), v in groups.items() if answerable == label]
        for label in (True, False)
    }
    if not all(means.values()):
        raise ValueError("confidence calibration needs positive and negative development scenarios")
    return float((np.mean(means[True]) + np.mean(means[False])) / 2)


def preserves_multilingual_recall(dev: dict[str, Any], ungated: dict[str, Any]) -> bool:
    """Protect cross-lingual and mixed speech when calibrating a general query gate."""
    return all(dev[lang]["recall5"]["mean"] >= 0.85 for lang in ("ru", "kk", "en")) and all(
        dev[scope]["recall5"]["mean"] >= ungated[scope]["recall5"]["mean"] - 0.01
        for scope in ("cross_lingual", "code_switch")
    )


def calibrate(
    settings: Settings, chosen: dict[str, Any], queries: list[RetrievalQuery], root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from supportpilot.retrieval.experiments import store_evaluation

    params = chosen["params"]
    if not params.get("reranker"):
        return chosen, []
    ranking_file = root / "comparisons" / chosen["name"] / "observations.json"
    score_file = root / "rerank" / params["reranker"] / "scores.json"
    fingerprint = sha256_text(
        json.dumps(
            {
                "rankings": sha256_file(ranking_file),
                "scores": sha256_file(score_file),
                "thresholds": THRESHOLDS,
                "seed": settings.retrieval.seed,
                "code": sha256_file(Path(__file__)),
                "metrics_code": sha256_file(Path(__file__).with_name("evaluate.py")),
            },
            sort_keys=True,
        )
    )
    receipt = root / "abstention_manifest.json"
    output = root / "abstention_results.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["fingerprint"] == fingerprint:
            if sha256_file(output) != saved["results_sha256"]:
                raise ValueError("confidence-calibration result checksum mismatch")
            results = json.loads(output.read_text())
            return next(r for r in results if r["name"] == saved["selected"]), results
    base = json.loads(ranking_file.read_text())
    scores = json.loads(score_file.read_text())
    limit = params.get("rerank_k", params["options"]["candidate_k"])
    confidence = {
        qid: max((score for _, score in values[:limit]), default=0)
        for qid, values in scores.items()
    }
    results, decisions = [], []
    for threshold in THRESHOLDS:
        name = f"{chosen['name']}_evidence{threshold:g}"
        rankings = {
            row["query_id"]: row["ranking"] if confidence[row["query_id"]] >= threshold else []
            for row in base
        }
        gated_params = {
            **params,
            "evidence_threshold": threshold,
            "options": {**params["options"], "evidence_threshold": threshold},
        }
        result = store_evaluation(
            settings, name, gated_params, queries, rankings, control=chosen["run_id"]
        )
        rows = json.loads((root / "comparisons" / name / "observations.json").read_text())
        dev = result["summary"]["development"]
        ungated = chosen["summary"]["development"]
        eligible = preserves_multilingual_recall(dev, ungated)
        accuracy = balanced_accuracy(rows)
        result["gate_balanced_accuracy"] = accuracy
        result["gate_eligible"] = eligible
        results.append(result)
        decisions.append(
            {
                "threshold": threshold,
                "eligible": eligible,
                "development_grouped_balanced_accuracy": accuracy,
                "development_recall": {
                    lang: dev[lang]["recall5"]["mean"] for lang in ("ru", "kk", "en")
                },
            }
        )
    eligible_results = [r for r in results if r["gate_eligible"]]
    if not eligible_results:
        raise ValueError("no evidence threshold meets development Recall@5 requirements")
    selected = max(
        eligible_results,
        key=lambda r: (
            r["gate_balanced_accuracy"],
            min(
                r["summary"]["development"][lang]["recall5"]["mean"] for lang in ("ru", "kk", "en")
            ),
            -r["params"]["evidence_threshold"],
        ),
    )
    write_json(output, results)
    write_json(
        receipt,
        {
            "fingerprint": fingerprint,
            "results_sha256": sha256_file(output),
            "selected": selected["name"],
            "decisions": decisions,
            "rule": (
                "Development scenario-balanced answerability accuracy; "
                "each language Recall@5 >= 0.85; cross-lingual/code-switch recall loss <= 0.01; "
                "development metrics only. Protocol amended after initial holdout inspection; "
                "holdout is exploratory, not an untouched confirmation set"
            ),
        },
    )
    return selected, results
