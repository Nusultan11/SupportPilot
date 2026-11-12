"""Graded retrieval metrics with scenario-cluster bootstrap, never gold-based tuning."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.hashing import sha256_text
from supportpilot.common.manifest import read_model_json, verify_files
from supportpilot.common.schemas import DatasetManifest
from supportpilot.data.retrievalset import RetrievalQuery

METRICS = ("recall5", "mrr10", "ndcg10")


def load_benchmark(path: Path) -> tuple[DatasetManifest, list[RetrievalQuery]]:
    manifest = read_model_json(path / "manifest.json", DatasetManifest)
    if verify_files(path, manifest.files, exclude=frozenset({"manifest.json"})):
        raise ValueError("retrieval benchmark manifest mismatch")
    queries = [
        RetrievalQuery.model_validate_json(line)
        for line in (path / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    expected = [f"{q.query_id}\t0\t{r.chunk_id}\t{r.grade}" for q in queries for r in q.relevant]
    if (path / "qrels.tsv").read_text().splitlines() != expected:
        raise ValueError("qrels disagree with benchmark queries")
    return manifest, queries


def phase(scenario_id: str, seed: int) -> str:
    return (
        "holdout" if int(sha256_text(f"{seed}|{scenario_id}")[:8], 16) % 5 == 0 else "development"
    )


def query_metrics(
    ranking: list[str], grades: dict[str, int], *, minimum_grade: int = 1
) -> dict[str, float]:
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranking must contain unique canonical chunk IDs")
    positives = {cid for cid, g in grades.items() if g >= minimum_grade}
    if not positives:
        raise ValueError("ranking metrics are undefined for unanswerable queries")
    recall = len(set(ranking[:5]) & positives) / len(positives)
    rr = next((1 / (i + 1) for i, cid in enumerate(ranking[:10]) if cid in positives), 0.0)
    gains = [2 ** grades.get(cid, 0) - 1 if cid in positives else 0 for cid in ranking[:10]]
    ideal = sorted((2 ** grades[cid] - 1 for cid in positives), reverse=True)[:10]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return {"recall5": recall, "mrr10": rr, "ndcg10": float(dcg / idcg)}


def slices(query: RetrievalQuery) -> list[str]:
    result = [
        "all",
        f"query_{query.lang}",
        query.category,
        f"pair_{query.lang}_{query.corpus_lang}",
    ]
    if query.cross_lingual:
        result.append("cross_lingual")
    elif not query.code_switch:
        result.append(query.lang)
    if query.code_switch:
        result.append("code_switch")
    result.append(
        "semantic_near_sft" if "semantic_near_sft" in query.quality_flags else "semantic_clean"
    )
    return result


def observations(
    queries: list[RetrievalQuery], rankings: dict[str, list[str]], *, seed: int
) -> list[dict[str, Any]]:
    if set(rankings) != {q.query_id for q in queries}:
        raise ValueError("missing or extra benchmark query results")
    result = []
    for q in queries:
        row: dict[str, Any] = {
            "query_id": q.query_id,
            "scenario_id": q.scenario_id,
            "lang": q.lang,
            "phase": phase(q.scenario_id, seed),
            "slices": slices(q),
            "answerable": q.answerable,
            "returned": len(rankings[q.query_id]),
            "ranking": rankings[q.query_id],
        }
        if q.answerable:
            grades = {r.chunk_id: r.grade for r in q.relevant}
            row.update(query_metrics(rankings[q.query_id], grades))
            row["grade2"] = query_metrics(rankings[q.query_id], grades, minimum_grade=2)
        result.append(row)
    return result


def clustered_ci(
    rows: list[dict[str, Any]], metric: str, *, seed: int, samples: int = 1000
) -> dict[str, Any]:
    if not rows:
        return {"mean": None, "low": None, "high": None, "queries": 0, "scenarios": 0}
    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        groups[r["scenario_id"]].append(float(r[metric]))
    values = list(groups.values())
    sums = np.array([sum(v) for v in values])
    counts = np.array([len(v) for v in values])
    indices = np.random.default_rng(seed).integers(0, len(values), size=(samples, len(values)))
    bootstrap = sums[indices].sum(1) / counts[indices].sum(1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(sums.sum() / counts.sum()),
        "low": float(low),
        "high": float(high),
        "queries": len(rows),
        "scenarios": len(values),
    }


def summarise(rows: list[dict[str, Any]], *, seed: int, samples: int = 1000) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for subset in ("full", "development", "holdout"):
        selected = [r for r in rows if subset == "full" or r["phase"] == subset]
        scopes = sorted({s for r in selected for s in r["slices"]})
        result = {}
        for scope in scopes:
            scoped = [r for r in selected if scope in r["slices"]]
            answerable = [r for r in scoped if r["answerable"]]
            unanswerable = [r for r in scoped if not r["answerable"]]
            result[scope] = {
                metric: clustered_ci(answerable, metric, seed=seed, samples=samples)
                for metric in METRICS
            }
            result[scope]["unanswerable"] = {
                "queries": len(unanswerable),
                "abstention_rate": sum(r["returned"] == 0 for r in unanswerable) / len(unanswerable)
                if unanswerable
                else None,
                "false_positive_rate": sum(r["returned"] > 0 for r in unanswerable)
                / len(unanswerable)
                if unanswerable
                else None,
            }
            abstention_rows = [
                {
                    "scenario_id": r["scenario_id"],
                    "abstention": float(r["returned"] == 0),
                    "false_positive": float(r["returned"] > 0),
                }
                for r in unanswerable
            ]
            result[scope]["unanswerable"]["abstention_ci"] = clustered_ci(
                abstention_rows, "abstention", seed=seed, samples=samples
            )
            result[scope]["unanswerable"]["false_positive_ci"] = clustered_ci(
                abstention_rows, "false_positive", seed=seed, samples=samples
            )
            if answerable:
                result[scope]["grade2_sensitivity"] = {
                    metric: clustered_ci(
                        [{"scenario_id": r["scenario_id"], **r["grade2"]} for r in answerable],
                        metric,
                        seed=seed,
                        samples=samples,
                    )
                    for metric in METRICS
                }
        summary[subset] = result
    return summary


def paired_comparison(
    control: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int,
    subset: str = "holdout",
    scope: str = "all",
) -> dict[str, Any]:
    other = {r["query_id"]: r for r in candidate}
    if set(other) != {r["query_id"] for r in control}:
        raise ValueError("paired comparison requires identical query IDs")
    diffs = [
        {"scenario_id": r["scenario_id"], **{m: other[r["query_id"]][m] - r[m] for m in METRICS}}
        for r in control
        if r["answerable"] and scope in r["slices"] and (subset == "full" or r["phase"] == subset)
    ]
    return {m: clustered_ci(diffs, m, seed=seed) for m in METRICS}
